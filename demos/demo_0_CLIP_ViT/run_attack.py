"""
VisInject Demo C1 - 跨模态特征对齐攻击（Cross-modal Embedding Alignment）

与demo1/2/3的端到端PGD攻击不同，本方法在CLIP共享嵌入空间中对齐视觉与文本向量。
攻击后的图片在CLIP语义空间中等同于攻击指令，可迁移影响多种MLLM。

运行示例：
  python run_attack.py --image sample/cat.png --target "chicken dinner"
  python run_attack.py --image sample/cat.png --preset chicken --iterations 1000
  python run_attack.py --image sample/cat.png --preset personal_info --compare
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MALICIOUS_PROMPTS_EN, CLIP_CONFIG, ATTACK_CONFIG
from clip_embedding_attack import CLIPEmbeddingAttack
from utils import (
    load_image, pil_to_tensor, tensor_to_pil,
    calculate_psnr, calculate_ssim,
    visualize_clip_attack, save_image
)

RUN_DIR = None
logger = None


def setup_run_dir(base_dir="logs_and_outputs"):
    global RUN_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(base_dir, timestamp)
    for sub in ['adversarial', 'visualizations', 'temp']:
        os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)
    return RUN_DIR


def setup_logging(run_dir):
    log_file = os.path.join(run_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)
    return logging.getLogger('clip_attack_demo')


def parse_args():
    parser = argparse.ArgumentParser(description='VisInject C1 - CLIP Embedding Alignment Attack')
    parser.add_argument('--image', type=str, default=None, help='输入图像路径')
    parser.add_argument('--preset', type=str, default='chicken',
                        choices=list(MALICIOUS_PROMPTS_EN.keys()),
                        help='预设攻击指令')
    parser.add_argument('--target', type=str, default=None,
                        help='自定义攻击指令文本（覆盖preset）')
    parser.add_argument('--epsilon', type=float, default=ATTACK_CONFIG['epsilon'])
    parser.add_argument('--alpha', type=float, default=ATTACK_CONFIG['alpha'])
    parser.add_argument('--iterations', type=int, default=ATTACK_CONFIG['num_iterations'])
    parser.add_argument('--loss-type', type=str, default=ATTACK_CONFIG['loss_type'],
                        choices=['cosine', 'l2'])
    parser.add_argument('--clip-model', type=str, default=CLIP_CONFIG['model_name'])
    parser.add_argument('--device', type=str, default=CLIP_CONFIG['device'])
    parser.add_argument('--sample-dir', type=str, default='sample')
    parser.add_argument('--output-dir', type=str, default='logs_and_outputs')
    parser.add_argument('--no-qaa', action='store_true', help='禁用量化感知')
    parser.add_argument('--compare', action='store_true',
                        help='运行多组对比实验（不同eps、loss类型）')
    return parser.parse_args()


def run_single_attack(attacker, image_path, target_text, args, output_dirs,
                      eps_override=None, loss_type_override=None, mode_name="default"):
    """执行单次攻击"""
    image_name = os.path.basename(image_path)
    logger.info("=" * 80)
    logger.info(f"模式: {mode_name} | 图片: {image_name} | 目标: '{target_text}'")

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  模式: {mode_name}")
    print(f"  图片: {image_name}")
    print(f"  目标: \"{target_text}\"")
    print(f"{'='*60}")

    original_pil = load_image(image_path)
    original_pil_resized = original_pil.resize((224, 224))
    image_tensor = pil_to_tensor(original_pil_resized, device=args.device)

    if eps_override is not None:
        attacker.epsilon = eps_override
    if loss_type_override is not None:
        attacker.loss_type = loss_type_override

    print(f"  执行CLIP嵌入对齐攻击 (eps={attacker.epsilon:.4f}, loss={attacker.loss_type})...")
    adv_image, perturbation, losses, attack_metrics = attacker.attack(
        image_tensor, target_text, verbose=True
    )

    if adv_image is None:
        logger.error(f"攻击失败: {mode_name}")
        print("  攻击失败！")
        return None

    elapsed = time.time() - start_time

    adv_pil = tensor_to_pil(adv_image)
    save_filename = f"adv_{mode_name}_{image_name}"
    save_path = os.path.join(output_dirs['adversarial'], save_filename)
    adv_pil.save(save_path)

    img_psnr = calculate_psnr(original_pil_resized, adv_pil)
    img_ssim = calculate_ssim(original_pil_resized, adv_pil)

    attack_metrics['psnr'] = img_psnr
    attack_metrics['ssim'] = img_ssim

    print(f"  PSNR: {img_psnr:.2f} dB, SSIM: {img_ssim:.4f}")

    reference_texts = [
        target_text,
        "a photo of a cat",
        "a photo of an animal",
        "What do you see in this image?",
    ]
    print(f"\n  CLIP相似度对比:")
    sim_results = attacker.compute_similarity_matrix(adv_image, reference_texts)
    for text, sim in sim_results.items():
        marker = " <<<" if text == target_text else ""
        print(f"    \"{text[:50]}\" → {sim:.4f}{marker}")
    attack_metrics['similarity_matrix'] = sim_results

    vis_path = os.path.join(output_dirs['visualizations'], f"vis_{mode_name}_{image_name}")
    visualize_clip_attack(
        original_pil_resized, adv_pil, perturbation, vis_path,
        target_text=target_text, metrics=attack_metrics, loss_curve=losses
    )

    print(f"  耗时: {elapsed:.1f}s")
    print(f"  保存: {save_path}")

    return {
        'mode': mode_name,
        'image_name': image_name,
        'target_text': target_text,
        'clip_similarity': attack_metrics['final_similarity'],
        'psnr': img_psnr,
        'ssim': img_ssim,
        'perturbation_linf': attack_metrics['perturbation_linf'],
        'save_path': save_path,
        'time': elapsed,
        'metrics': attack_metrics,
    }


def main():
    args = parse_args()
    run_dir = setup_run_dir(args.output_dir)
    output_dirs = {
        'adversarial': os.path.join(run_dir, 'adversarial'),
        'visualizations': os.path.join(run_dir, 'visualizations'),
        'temp': os.path.join(run_dir, 'temp'),
    }

    global logger
    logger = setup_logging(run_dir)

    target_text = args.target if args.target else MALICIOUS_PROMPTS_EN[args.preset]

    print("\n" + "=" * 60)
    print("  VisInject C1 - 跨模态特征对齐攻击")
    print("  (Cross-modal Embedding Alignment)")
    print(f"  目标: \"{target_text}\"")
    print("=" * 60)

    attacker = CLIPEmbeddingAttack(
        model_name=args.clip_model,
        device=args.device,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iterations=args.iterations,
        random_start=True,
        loss_type=args.loss_type,
        quantize_aware=not args.no_qaa,
    )

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            [str(p) for p in Path(args.sample_dir).glob("*.png")] +
            [str(p) for p in Path(args.sample_dir).glob("*.jpg")]
        )

    if not image_paths:
        print("  错误：未找到输入图像！请指定 --image 或放图片到 sample/ 目录")
        return

    if args.compare:
        attack_modes = [
            ('cosine_eps16', 16/255, 'cosine'),
            ('cosine_eps32', 32/255, 'cosine'),
            ('cosine_eps64', 64/255, 'cosine'),
            ('l2_eps16', 16/255, 'l2'),
            ('l2_eps32', 32/255, 'l2'),
        ]
    else:
        attack_modes = [('default', args.epsilon, args.loss_type)]

    all_results = []

    for mode_name, eps, loss_type in attack_modes:
        for path in image_paths:
            result = run_single_attack(
                attacker, path, target_text, args, output_dirs,
                eps_override=eps, loss_type_override=loss_type,
                mode_name=mode_name
            )
            if result:
                all_results.append(result)

            if args.device == 'cuda':
                torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print(f"{'模式':<20} {'图片':<12} {'CLIP Sim':<10} {'PSNR':<8} {'SSIM':<8} {'L∞':<8} {'耗时'}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['mode']:<20} {r['image_name']:<12} "
            f"{r['clip_similarity']:<10.4f} {r['psnr']:<8.2f} "
            f"{r['ssim']:<8.4f} {r['perturbation_linf']:<8.4f} "
            f"{r['time']:.1f}s"
        )
    print("=" * 90)

    results_file = os.path.join(run_dir, 'results.json')
    serializable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != 'metrics'}
        entry['metrics'] = {
            k: (v if not isinstance(v, dict) else v)
            for k, v in r['metrics'].items()
        }
        serializable.append(entry)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n  结果保存在: {run_dir}")
    print(f"  对抗图像: {output_dirs['adversarial']}")
    print(f"  可视化: {output_dirs['visualizations']}")


if __name__ == "__main__":
    main()
