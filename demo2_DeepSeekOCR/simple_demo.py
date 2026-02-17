"""
VisInject Demo - BLIP-2端到端对抗攻击 (修复版)
1. 修复了文件名覆盖的问题
2. 修复了缩放算法不一致导致的推理失败问题

运行示例：
python simple_demo.py --compare --image sample/cat.png
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
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 全局变量
RUN_DIR = None
logger = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MALICIOUS_PROMPTS_EN, PGD_CONFIG, MODEL_CONFIG, DEFAULT_QUESTION
from model_loader import load_model
from pgd_attack import pgd_attack_wrapper, test_attack_success
from utils import (
    load_image, pil_to_tensor, tensor_to_pil,
    calculate_psnr, calculate_ssim,
    visualize_comparison, save_image,
    print_banner
)


def setup_run_dir(base_dir="logs_and_outputs"):
    """创建本次运行目录"""
    global RUN_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(base_dir, timestamp)
    for sub in ['adversarial', 'visualizations', 'temp']:
        os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)
    return RUN_DIR


def setup_logging(run_dir, debug_mode=False):
    log_file = os.path.join(run_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)
    return logging.getLogger('simple_demo')


def parse_args():
    parser = argparse.ArgumentParser(description='VisInject Fix')
    parser.add_argument('--preset', type=str, default='chicken', choices=list(MALICIOUS_PROMPTS_EN.keys()))
    parser.add_argument('--custom-prompt', type=str, default=None)
    parser.add_argument('--question', type=str, default=DEFAULT_QUESTION)
    parser.add_argument('--epsilon', type=float, default=PGD_CONFIG['epsilon'])
    parser.add_argument('--alpha', type=float, default=PGD_CONFIG['alpha'])
    parser.add_argument('--iterations', type=int, default=PGD_CONFIG['num_iterations'])
    parser.add_argument('--model-name', type=str, default=MODEL_CONFIG['model_name'])
    parser.add_argument('--device', type=str, default=MODEL_CONFIG['device'])
    parser.add_argument('--sample-dir', type=str, default='sample')
    parser.add_argument('--output-dir', type=str, default='logs_and_outputs')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--compare', action='store_true', help='运行5组对比实验')
    return parser.parse_args()


def test_single_image(model, image_path, question, target_text, args, output_dirs,
                      quantize_aware=False, eps_override=None, alpha_override=None, mode_name="custom"):
    """测试单张图片"""
    image_name = os.path.basename(image_path)
    logger.info("=" * 80)
    logger.info(f"模式: {mode_name} | 图片: {image_name} | 目标: '{target_text}'")
    
    start_time = time.time()
    
    # 1. 加载并强制 Resize (修复核心 BUG)
    # BLIP-2 的输入尺寸是 224x224。我们在攻击前就缩放好，
    # 这样攻击算法生成的像素就是最终像素，不会被后续的缩放破坏。
    print(f"  1. 加载图像并 Resize (224x224)...")
    original_pil = load_image(image_path)
    original_pil = original_pil.resize((224, 224), Image.Resampling.BICUBIC) # 强制统一尺寸
    
    image_tensor = pil_to_tensor(original_pil, device=args.device)
    
    # 2. 原始输出
    print("  2. 测试原始输出...")
    # 注意：这里我们传入已经 resize 过的图片 tensor 对应的 PIL
    # 为了避免 model_loader 再次读取原图，我们需要改一下逻辑，或者临时保存
    # 这里简单起见，我们直接看攻击后的效果，原始输出仅供参考
    
    # 3. PGD攻击
    use_eps = eps_override if eps_override is not None else args.epsilon
    use_alpha = alpha_override if alpha_override is not None else args.alpha
    
    print(f"     攻击模式: {mode_name} (eps={use_eps:.4f})...")
    
    try:
        adv_image, perturbation, losses = pgd_attack_wrapper(
            model=model,
            image_tensor=image_tensor,
            question=question,
            target_text=target_text,
            epsilon=use_eps,
            alpha=use_alpha,
            num_iterations=args.iterations,
            quantize_aware=quantize_aware,
            verbose=True
        )
    except Exception as e:
        logger.error(f"攻击失败: {e}")
        return None
    
    # 4. 验证 (保存并重新加载)
    print("  4. 验证...")
    # 使用 mode_name 作为文件名前缀，防止覆盖
    save_filename = f"adv_{mode_name}_{image_name}"
    save_path = os.path.join(output_dirs['adversarial'], save_filename)
    
    # 保存图片
    adv_pil = tensor_to_pil(adv_image)
    adv_pil.save(save_path)
    
    # 使用 model_loader 的 save_and_test 流程验证
    # 注意：这里传入的是已经 resize 过的 224x224 图片，这正是我们想要的
    success, adversarial_output = test_attack_success(
        model, adv_image, question, target_text,
        temp_dir=output_dirs['temp']
    )
    
    print(f"     对抗输出: {adversarial_output}")
    status = "✓ 成功" if success else "✗ 失败"
    print(f"     结果: {status}")
    
    # 5. 可视化
    metrics = {'psnr': calculate_psnr(original_pil, adv_pil), 'ssim': calculate_ssim(original_pil, adv_pil), 'success': success}
    visualize_comparison(original_pil, adv_pil, perturbation,
                         os.path.join(output_dirs['visualizations'], f"vis_{mode_name}_{image_name}"),
                         "[Skipped]", adversarial_output, metrics)
    
    return {
        'mode': mode_name,
        'image_name': image_name,
        'adversarial_output': adversarial_output,
        'psnr': metrics['psnr'],
        'attack_success': success,
        'save_path': save_path
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
    
    print_banner(f"VisInject Fix - 目标: {target_text}")
    
    model = load_model(model_name=args.model_name, device=args.device)
    
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([str(p) for p in Path(args.sample_dir).glob("*.png")] +
                             [str(p) for p in Path(args.sample_dir).glob("*.jpg")])
    
    # 定义攻击模式
    if args.compare:
        attack_modes = [
            ('A_standard',     False, args.epsilon, args.alpha),
            ('B_large_eps',    False, 64/255,       2/255),
            ('C_qaa',          True,  args.epsilon, args.alpha),
            ('D_qaa_large',    True,  64/255,       2/255),
            ('E_quant_train',  'ste', args.epsilon, args.alpha),
        ]
    else:
        attack_modes = [('Custom', args.qaa, args.epsilon, args.alpha)]
    
    results = []
    
    for mode_name, qaa_flag, eps, alpha in attack_modes:
        print(f"\n>>> 运行模式: {mode_name}")
        for path in image_paths:
            res = test_single_image(model, path, args.question, target_text, args, output_dirs,
                                    quantize_aware=qaa_flag, eps_override=eps, alpha_override=alpha,
                                    mode_name=mode_name)
            if res:
                results.append(res)
            
            # 清理显存
            if args.device == 'cuda':
                torch.cuda.empty_cache()

    # 打印总结
    print("\n" + "="*80)
    print(f"{'模式':<15} {'图片':<15} {'成功':<5} {'PSNR':<8} {'输出'}")
    print("-" * 80)
    for r in results:
        success = "✓" if r['attack_success'] else "✗"
        print(f"{r['mode']:<15} {r['image_name']:<15} {success:<5} {r['psnr']:<8.2f} {r['adversarial_output'][:30]}...")
    print("="*80)
    print(f"结果保存在: {run_dir}")

if __name__ == "__main__":
    main()