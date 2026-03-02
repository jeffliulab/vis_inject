"""
工具函数：图像处理、指标计算、可视化
"""

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def pil_to_tensor(image, device='cuda'):
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor


def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    image_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def calculate_psnr(original, adversarial):
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(adversarial, Image.Image):
        adversarial = np.array(adversarial)
    return psnr(original, adversarial, data_range=255)


def calculate_ssim(original, adversarial):
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(adversarial, Image.Image):
        adversarial = np.array(adversarial)
    return ssim(original, adversarial, channel_axis=2, data_range=255)


def visualize_clip_attack(original_img, adversarial_img, perturbation, save_path,
                          target_text="", metrics=None, loss_curve=None):
    """
    可视化CLIP对齐攻击结果：原图、对抗图、扰动、loss曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(adversarial_img)
    axes[0, 1].set_title('Adversarial Image (CLIP-Aligned)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    perturbation_np = perturbation.squeeze(0).cpu().numpy()
    perturbation_np = np.transpose(perturbation_np, (1, 2, 0))
    perturbation_display = np.clip((perturbation_np * 10 + 0.5), 0, 1)
    axes[1, 0].imshow(perturbation_display)
    axes[1, 0].set_title('Perturbation (10x amplified)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    if loss_curve and len(loss_curve) > 1:
        axes[1, 1].plot(loss_curve, color='#e74c3c', linewidth=1.5)
        axes[1, 1].set_title('Alignment Loss Curve', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        diff = np.abs(np.array(adversarial_img).astype(float) - np.array(original_img).astype(float))
        if diff.max() > 0:
            diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff_normalized = diff.astype(np.uint8)
        axes[1, 1].imshow(diff_normalized)
        axes[1, 1].set_title('Absolute Difference', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

    info_lines = [f"Target Text: \"{target_text}\""]
    if metrics:
        info_lines.append(
            f"CLIP Similarity: {metrics.get('original_similarity', 0):.4f} -> "
            f"{metrics.get('final_similarity', 0):.4f}"
        )
        info_lines.append(
            f"PSNR: {metrics.get('psnr', 0):.2f} dB  |  "
            f"SSIM: {metrics.get('ssim', 0):.4f}  |  "
            f"L_inf: {metrics.get('perturbation_linf', 0):.4f}"
        )

    info_text = "\n".join(info_lines)
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    image.save(save_path)
