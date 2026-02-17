"""
工具函数：图像处理、指标计算、可视化
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免显示问题
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os


def load_image(image_path):
    """加载图像并转换为PIL Image对象"""
    return Image.open(image_path).convert('RGB')


def pil_to_tensor(image, device='cuda'):
    """将PIL Image转换为tensor [1, 3, H, W]，范围[0, 1]"""
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor


def tensor_to_pil(tensor):
    """将tensor [1, 3, H, W]转换为PIL Image"""
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    image_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def calculate_psnr(original, adversarial):
    """
    计算PSNR（峰值信噪比）
    Args:
        original: PIL Image或numpy array
        adversarial: PIL Image或numpy array
    Returns:
        psnr_value: float
    """
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(adversarial, Image.Image):
        adversarial = np.array(adversarial)
    
    # 适配 scikit-image 0.26.0
    return psnr(original, adversarial, data_range=255)


def calculate_ssim(original, adversarial):
    """
    计算SSIM（结构相似度）
    Args:
        original: PIL Image或numpy array
        adversarial: PIL Image或numpy array
    Returns:
        ssim_value: float
    """
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(adversarial, Image.Image):
        adversarial = np.array(adversarial)
    
    # 适配 scikit-image 0.26.0 - channel_axis 参数
    return ssim(original, adversarial, channel_axis=2, data_range=255)


def visualize_comparison(original_img, adversarial_img, perturbation, save_path, 
                         original_text="", adversarial_text="", metrics=None):
    """
    可视化对比图：原图、对抗图、扰动
    Args:
        original_img: PIL Image
        adversarial_img: PIL Image
        perturbation: torch.Tensor [1, 3, H, W]，范围[-epsilon, epsilon]
        save_path: str
        original_text: str, 模型对原图的输出
        adversarial_text: str, 模型对对抗图的输出
        metrics: dict, 包含PSNR、SSIM等指标
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原图
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 对抗图
    axes[0, 1].imshow(adversarial_img)
    axes[0, 1].set_title('Adversarial Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 扰动（放大显示）
    perturbation_np = perturbation.squeeze(0).cpu().numpy()
    perturbation_np = np.transpose(perturbation_np, (1, 2, 0))
    # 放大10倍以便可视化
    perturbation_display = np.clip((perturbation_np * 10 + 0.5), 0, 1)
    axes[1, 0].imshow(perturbation_display)
    axes[1, 0].set_title('Perturbation (10x amplified)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 差异高亮图
    diff = np.abs(np.array(adversarial_img).astype(float) - np.array(original_img).astype(float))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    axes[1, 1].imshow(diff_normalized)
    axes[1, 1].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 添加文本信息
    info_text = f"Original Output:\n{original_text[:100]}...\n\n"
    info_text += f"Adversarial Output:\n{adversarial_text[:100]}...\n\n"
    if metrics:
        info_text += f"PSNR: {metrics['psnr']:.2f} dB\n"
        info_text += f"SSIM: {metrics['ssim']:.4f}\n"
        info_text += f"Attack Success: {'✓' if metrics.get('success', False) else '✗'}"
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_image(image, save_path):
    """保存PIL Image到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    image.save(save_path)


def check_attack_success(output_text, target_text):
    """
    检查攻击是否成功（目标文本是否出现在输出中）
    Args:
        output_text: str, 模型输出
        target_text: str, 目标注入文本
    Returns:
        success: bool
    """
    # 简单的子串匹配
    return target_text.lower() in output_text.lower()


def print_banner(text):
    """打印带边框的标题"""
    width = len(text) + 4
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_result(image_name, original_output, adversarial_output, 
                 psnr, ssim, success, time_cost):
    """打印单张图片的测试结果"""
    print(f"\n{'='*80}")
    print(f"图片: {image_name}")
    print(f"{'-'*80}")
    print(f"原始输出: {original_output}")
    print(f"对抗输出: {adversarial_output}")
    print(f"{'-'*80}")
    status = "✓ 攻击成功" if success else "✗ 攻击失败"
    print(f"状态: {status}")
    print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | 耗时: {time_cost:.1f}s")
    print(f"{'='*80}")
