"""
VisInject Demo - DeepSeek-VL 1.3B 端到端对抗攻击
运行示例：
python simple_demo.py --compare --image sample/cat.png
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RUN_DIR = None
logger = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MALICIOUS_PROMPTS_EN, PGD_CONFIG, MODEL_CONFIG, DEFAULT_QUESTION, VISION_CONFIG
from model_loader import load_model
from pgd_attack import pgd_attack_wrapper, test_attack_success
from utils import (
    load_image, pil_to_tensor, tensor_to_pil,
    calculate_psnr, calculate_ssim,
    visualize_comparison, print_banner
)

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
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)
    return logging.getLogger('simple_demo')

def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepSeek-VL 1.3B Attack Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--preset', type=str, default='chicken', choices=list(MALICIOUS_PROMPTS_EN.keys()))
    parser.add_argument('--custom-prompt', type=str, default=None)
    parser.add_argument('--question', type=str, default=DEFAULT_QUESTION)
    parser.add_argument('--epsilon', type=float, default=PGD_CONFIG['epsilon'])
    parser.add_argument('--iterations', type=int, default=PGD_CONFIG['num_iterations'])
    parser.add_argument('--model-name', type=str, default=MODEL_CONFIG['model_name'])
    parser.add_argument('--device', type=str, default=MODEL_CONFIG['device'])
    parser.add_argument('--sample-dir', type=str, default='sample')
    parser.add_argument('--output-dir', type=str, default='logs_and_outputs')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--compare', action='store_true', help='运行对比实验')
    return parser.parse_args()

def test_single_image(model, image_path, question, target_text, args, output_dirs,
                      quantize_aware=False, eps_override=None, mode_name="custom"):
    image_name = os.path.basename(image_path)
    logger.info("=" * 80)
    logger.info(f"模式: {mode_name} | 图片: {image_name}")
    logger.info(f"目标: '{target_text}'")

    target_size = VISION_CONFIG['image_size']
    print(f"  1. 加载图像并 Resize ({target_size}x{target_size})...")

    original_pil = load_image(image_path)
    original_pil = original_pil.resize((target_size, target_size), Image.Resampling.BICUBIC)
    image_tensor = pil_to_tensor(original_pil, device=args.device)

    use_eps = eps_override if eps_override is not None else args.epsilon
    print(f"     攻击模式: {mode_name} (eps={use_eps:.4f})...")

    try:
        adv_image, perturbation, losses = pgd_attack_wrapper(
            model=model,
            image_tensor=image_tensor,
            question=question,
            target_text=target_text,
            epsilon=use_eps,
            alpha=use_eps / 50,
            num_iterations=args.iterations,
            quantize_aware=quantize_aware,
            verbose=True
        )
    except Exception as e:
        logger.error(f"攻击崩溃: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    print("  3. 验证攻击效果...")
    save_filename = f"adv_{mode_name}_{image_name}"
    save_path = os.path.join(output_dirs['adversarial'], save_filename)

    success, adversarial_output = test_attack_success(
        model, adv_image, question, target_text,
        temp_dir=output_dirs['temp']
    )

    print(f"     对抗输出: {adversarial_output}")
    status = "SUCCESS" if success else "FAILED"
    print(f"     结果: {status}")

    adv_pil = tensor_to_pil(adv_image)
    adv_pil.save(save_path)
    logger.info(f"[保存] 对抗图像 → {save_path}")

    metrics = {
        'psnr': calculate_psnr(original_pil, adv_pil),
        'ssim': calculate_ssim(original_pil, adv_pil),
        'success': success
    }
    visualize_comparison(
        original_pil, adv_pil, perturbation,
        os.path.join(output_dirs['visualizations'], f"vis_{mode_name}_{image_name}"),
        "[Ref]", adversarial_output, metrics
    )

    return {
        'mode': mode_name,
        'image_name': image_name,
        'output': adversarial_output,
        'psnr': metrics['psnr'],
        'success': success
    }

def main():
    args = parse_args()

    run_dir = setup_run_dir(args.output_dir)
    output_dirs = {
        'adversarial': os.path.join(run_dir, 'adversarial'),
        'visualizations': os.path.join(run_dir, 'visualizations'),
        'temp': os.path.join(run_dir, 'temp')
    }

    global logger
    logger = setup_logging(run_dir)
    target_text = args.custom_prompt if args.custom_prompt else MALICIOUS_PROMPTS_EN[args.preset]

    print_banner(f"DeepSeek-VL Attack - {MODEL_CONFIG['model_name']}")
    print(f"  Image Size: {VISION_CONFIG['image_size']}x{VISION_CONFIG['image_size']} (SigLIP-L fixed)")
    print(f"  Vision Tokens: {VISION_CONFIG['num_patches']}")
    print()

    try:
        model = load_model(model_name=args.model_name, device=args.device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([str(p) for p in Path(args.sample_dir).glob("*.png")] +
                             [str(p) for p in Path(args.sample_dir).glob("*.jpg")])

    if not image_paths:
        print("未找到图片，请检查 --sample-dir 或 --image")
        return

    if args.compare:
        attack_modes = [
            ('Standard',     False, args.epsilon),
            ('QAA',          True,  args.epsilon),
            ('QAA_High',     True,  64/255),
        ]
    else:
        attack_modes = [('Custom', True, args.epsilon)]

    results = []

    for mode_name, qaa_flag, eps in attack_modes:
        print(f"\n>>> 运行模式: {mode_name}")
        for path in image_paths:
            res = test_single_image(
                model, path, args.question, target_text, args, output_dirs,
                quantize_aware=qaa_flag, eps_override=eps, mode_name=mode_name
            )
            if res:
                results.append(res)

            if args.device == 'cuda':
                torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"{'Mode':<12} {'Image':<15} {'Success':<8} {'PSNR':<8} {'Output Snippet'}")
    print("-" * 80)
    for r in results:
        status = "YES" if r['success'] else "NO"
        print(f"{r['mode']:<12} {r['image_name']:<15} {status:<8} {r['psnr']:<8.2f} {r['output'][:40].replace(chr(10), ' ')}...")
    print("="*80)
    print(f"Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
