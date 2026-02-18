# Qwen2.5-VL-3B 独立推理脚本
# 用法:
#   python test_inference.py --image path/to/image.png
#   python test_inference.py --image logs_and_outputs/xxx/adversarial/adv_xxx.png
#   python test_inference.py --image sample/cat.png --mode both

import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, VISION_CONFIG

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B 独立推理验证")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--question", type=str, default="Describe this image.",
                        help="提问内容")
    parser.add_argument("--device", type=str, default=MODEL_CONFIG['device'])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["native", "manual", "both"],
                        help="推理模式: native=Processor, manual=手动pixel_values, both=两者对比")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"{RED}错误: 找不到文件 {args.image}{RESET}")
        return

    print(f"\n{'='*60}")
    print(f"  Qwen2.5-VL-3B 推理测试")
    print(f"{'='*60}")
    print(f"  图片:     {args.image}")
    print(f"  问题:     {args.question}")
    print(f"  模式:     {args.mode}")
    print()

    from model_loader import load_model
    print(f"加载模型: {MODEL_CONFIG['model_name']} ...")
    model = load_model(device=args.device)
    print()

    if args.mode in ("native", "both"):
        print(f"{CYAN}[Native] Processor 推理...{RESET}")
        try:
            result_native = model.generate(args.image, args.question)
            print(f"{GREEN}输出:{RESET} {result_native}")
        except Exception as e:
            print(f"{RED}Native 推理失败: {e}{RESET}")
        print()

    if args.mode in ("manual", "both"):
        print(f"{CYAN}[Manual] 手动 pixel_values 推理...{RESET}")
        try:
            from PIL import Image
            from utils import pil_to_tensor

            img = Image.open(args.image).convert('RGB')
            img = img.resize(
                (VISION_CONFIG['image_size'], VISION_CONFIG['image_size']),
                Image.Resampling.BICUBIC,
            )
            image_tensor = pil_to_tensor(img, device=args.device)
            result_manual = model.generate_from_tensor(image_tensor, args.question)
            print(f"{GREEN}输出:{RESET} {result_manual}")
        except Exception as e:
            print(f"{RED}Manual 推理失败: {e}{RESET}")
            import traceback
            traceback.print_exc()
        print()

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
