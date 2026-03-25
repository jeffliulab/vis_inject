"""
VisInject Web Demo (Gradio)
============================
Bilingual (EN/CN) web interface for the VisInject pipeline.

Two modes:
  - Quick mode: reuse cached UniversalAttack checkpoint (instant)
  - Full mode: run UniversalAttack optimization from scratch (shows progress)

Usage:
    python web_demo.py
    python web_demo.py --port 7861 --share
"""

import argparse
import glob
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    UNIVERSAL_ATTACK_CONFIG, ATTACK_TARGETS, ANYATTACK_CONFIG, OUTPUT_CONFIG,
)

# ── Bilingual text ────────────────────────────────────────────────

TEXT = {
    "en": {
        "title": "VisInject: Adversarial Prompt Injection for VLMs",
        "subtitle": "Generate adversarial images that hijack multimodal LLM responses",
        "tab_generate": "Generate",
        "tab_evaluate": "Evaluate",
        "target_phrase": "Target phrase",
        "target_phrase_info": "The phrase the VLM will be forced to say",
        "clean_image": "Clean image",
        "clean_image_info": "The image to inject adversarial semantics into",
        "num_steps": "Optimization steps",
        "num_steps_info": "More steps = better attack but slower (3000 recommended for multi-model)",
        "eps": "Perturbation budget (eps)",
        "eps_info": "L-inf noise budget. Higher = stronger attack but more visible",
        "target_models": "Target VLMs",
        "generate_btn": "Generate Adversarial Image",
        "result_image": "Adversarial Image",
        "result_info": "Generation Info",
        "universal_image": "Universal Image (intermediate)",
        "psnr": "PSNR",
        "noise_linf": "Noise L-inf",
        "status": "Status",
        "quick_mode": "Quick mode (use cached universal image if available)",
        "eval_vlm": "Evaluation VLM",
        "eval_btn": "Evaluate",
        "eval_result": "Evaluation Result",
        "adv_image_eval": "Adversarial image to evaluate",
        "lang_switch": "Language / 语言",
    },
    "cn": {
        "title": "VisInject: 多模态大模型对抗性 Prompt 注入",
        "subtitle": "生成能劫持多模态大模型回答的对抗图片",
        "tab_generate": "生成",
        "tab_evaluate": "评估",
        "target_phrase": "目标短语",
        "target_phrase_info": "VLM 将被迫说出的短语",
        "clean_image": "干净图片",
        "clean_image_info": "需要注入对抗语义的图片",
        "num_steps": "优化步数",
        "num_steps_info": "步数越多攻击效果越好但越慢（多模型推荐 3000）",
        "eps": "扰动预算 (eps)",
        "eps_info": "L-inf 噪声预算。越大攻击越强但越可见",
        "target_models": "目标 VLM",
        "generate_btn": "生成对抗图片",
        "result_image": "对抗图片",
        "result_info": "生成信息",
        "universal_image": "通用对抗图（中间产物）",
        "psnr": "PSNR",
        "noise_linf": "噪声 L-inf",
        "status": "状态",
        "quick_mode": "快速模式（如有缓存的通用对抗图则直接复用）",
        "eval_vlm": "评估用 VLM",
        "eval_btn": "评估",
        "eval_result": "评估结果",
        "adv_image_eval": "待评估的对抗图片",
        "lang_switch": "Language / 语言",
    },
}


def find_cached_universal() -> str | None:
    """Find any cached universal image."""
    pattern = os.path.join(OUTPUT_CONFIG["universal_dir"], "universal_*.png")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def run_generate(
    clean_image_path: str,
    target_phrase: str,
    num_steps: int,
    eps: float,
    target_models: list[str],
    quick_mode: bool,
    progress=None,
):
    """Run the VisInject pipeline and return results."""
    from pipeline import run_universal_attack, run_anyattack_fusion
    from utils import load_image, compute_psnr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: UniversalAttack
    universal_path = None
    if quick_mode:
        universal_path = find_cached_universal()

    if universal_path is None:
        ua_config = dict(UNIVERSAL_ATTACK_CONFIG)
        ua_config["num_steps"] = num_steps
        universal_path = run_universal_attack(
            target_phrase=target_phrase,
            target_models=target_models,
            config=ua_config,
            output_dir=OUTPUT_CONFIG["universal_dir"],
            checkpoint_dir=OUTPUT_CONFIG["checkpoint_dir"],
            device=device,
        )

    # Step 2: AnyAttack fusion
    aa_config = dict(ANYATTACK_CONFIG)
    aa_config["eps"] = eps
    adv_paths = run_anyattack_fusion(
        universal_image_path=universal_path,
        clean_image_paths=[clean_image_path],
        config=aa_config,
        output_dir=OUTPUT_CONFIG["adversarial_dir"],
        device=device,
    )

    # Compute metrics
    clean = load_image(clean_image_path, 224)
    adv = load_image(adv_paths[0], 224)
    psnr = compute_psnr(clean, adv)
    noise_linf = (adv - clean).abs().max().item()

    info = (
        f"Universal image: {os.path.basename(universal_path)}\n"
        f"PSNR: {psnr:.1f} dB\n"
        f"Noise L-inf: {noise_linf:.4f}\n"
        f"Target models: {', '.join(target_models)}\n"
        f"Steps: {num_steps}"
    )

    return adv_paths[0], universal_path, info


def run_eval(adv_image_path: str, eval_vlm: str, target_phrase: str):
    """Run ASR evaluation on a single image."""
    from evaluate import evaluate_asr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = evaluate_asr(
        adv_image_path=adv_image_path,
        target_phrase=target_phrase,
        target_models=[eval_vlm],
        num_adversarial=10,
        num_safe=5,
        device=device,
    )

    model_result = results.get(eval_vlm, {})
    asr = model_result.get("asr", 0)
    details = model_result.get("details", [])

    output_lines = [f"ASR: {asr:.1f}%\n"]
    for d in details[:10]:
        tag = "ADV" if d["is_adversarial"] else "SAFE"
        status = "HIT" if d["attack_success"] else "MISS"
        output_lines.append(f"[{tag}][{status}] Q: {d['question'][:60]}")
        output_lines.append(f"  A: {d['response'][:80]}\n")

    return "\n".join(output_lines)


def build_ui(lang: str = "en"):
    """Build the Gradio interface."""
    import gradio as gr

    t = TEXT[lang]

    model_choices = ATTACK_TARGETS

    with gr.Blocks(title=t["title"]) as demo:
        gr.Markdown(f"# {t['title']}\n{t['subtitle']}")

        with gr.Tabs():
            # ── Generate tab ──
            with gr.Tab(t["tab_generate"]):
                with gr.Row():
                    with gr.Column():
                        clean_img = gr.Image(
                            label=t["clean_image"], type="filepath",
                            sources=["upload", "clipboard"],
                        )
                        target_phrase = gr.Textbox(
                            label=t["target_phrase"],
                            value=UNIVERSAL_ATTACK_CONFIG["target_phrase"],
                            info=t["target_phrase_info"],
                        )
                        target_models = gr.CheckboxGroup(
                            choices=model_choices,
                            value=model_choices[:2],
                            label=t["target_models"],
                        )
                        num_steps = gr.Slider(
                            100, 5000, value=3000, step=100,
                            label=t["num_steps"], info=t["num_steps_info"],
                        )
                        eps = gr.Slider(
                            0.01, 0.2, value=ANYATTACK_CONFIG["eps"], step=0.005,
                            label=t["eps"], info=t["eps_info"],
                        )
                        quick_mode = gr.Checkbox(
                            value=True, label=t["quick_mode"],
                        )
                        gen_btn = gr.Button(t["generate_btn"], variant="primary")

                    with gr.Column():
                        result_img = gr.Image(label=t["result_image"], type="filepath")
                        universal_img = gr.Image(label=t["universal_image"], type="filepath")
                        result_info = gr.Textbox(label=t["result_info"], lines=6)

                gen_btn.click(
                    fn=run_generate,
                    inputs=[clean_img, target_phrase, num_steps, eps, target_models, quick_mode],
                    outputs=[result_img, universal_img, result_info],
                )

            # ── Evaluate tab ──
            with gr.Tab(t["tab_evaluate"]):
                with gr.Row():
                    with gr.Column():
                        eval_img = gr.Image(
                            label=t["adv_image_eval"], type="filepath",
                            sources=["upload", "clipboard"],
                        )
                        eval_vlm = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0] if model_choices else None,
                            label=t["eval_vlm"],
                        )
                        eval_phrase = gr.Textbox(
                            label=t["target_phrase"],
                            value=UNIVERSAL_ATTACK_CONFIG["target_phrase"],
                        )
                        eval_btn = gr.Button(t["eval_btn"], variant="primary")

                    with gr.Column():
                        eval_result = gr.Textbox(label=t["eval_result"], lines=20)

                eval_btn.click(
                    fn=run_eval,
                    inputs=[eval_img, eval_vlm, eval_phrase],
                    outputs=[eval_result],
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="VisInject Web Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "cn"])
    args = parser.parse_args()

    demo = build_ui(lang=args.lang)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
