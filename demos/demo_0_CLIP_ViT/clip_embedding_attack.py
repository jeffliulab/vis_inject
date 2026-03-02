"""
跨模态特征对齐攻击（Cross-modal Embedding Alignment）

核心思想：
  不通过端到端的MLLM梯度攻击，而是在CLIP的共享嵌入空间中，
  让图像经过视觉编码器后的向量 逼近 攻击指令文本的向量。

数学目标：
  min_δ || E_v(x + δ) - E_t(target_text) ||^2
  s.t. ||δ||_∞ ≤ ε,  x + δ ∈ [0, 1]

其中 E_v 是CLIP视觉编码器, E_t 是CLIP文本编码器。

效果：
  图片视觉上仍像原图，但在CLIP嵌入空间中语义等同于攻击指令。
  任何使用CLIP或类CLIP视觉编码器的MLLM都可能受影响。
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CLIPEmbeddingAttack:
    """基于CLIP嵌入空间对齐的对抗攻击"""

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda",
                 epsilon=16/255, alpha=1/255, num_iterations=1000,
                 random_start=True, loss_type="cosine", quantize_aware=True):
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.random_start = random_start
        self.loss_type = loss_type
        self.quantize_aware = quantize_aware

        logger.info(f"加载CLIP模型: {model_name}")
        print(f"加载CLIP模型: {model_name} ...")

        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        self.image_size = self.clip_processor.image_processor.size.get(
            "shortest_edge",
            self.clip_processor.image_processor.size.get("height", 224)
        )
        self.mean = torch.tensor(
            self.clip_processor.image_processor.image_mean,
            device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self.clip_processor.image_processor.image_std,
            device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  CLIP模型加载完成，显存: {mem:.1f}GB")

    def _normalize_for_clip(self, image_tensor):
        """CLIP标准化预处理: resize + normalize"""
        resized = F.interpolate(
            image_tensor, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        )
        return (resized - self.mean) / self.std

    def _quantize_snap(self, image):
        """模拟PNG uint8量化"""
        return (image * 255).round() / 255.0

    def _encode_text(self, text):
        """编码攻击指令文本 → CLIP文本嵌入"""
        text_inputs = self.clip_processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs.get('attention_mask'),
            )
            pooled_output = text_outputs.pooler_output
            text_features = self.clip_model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_image(self, pixel_values):
        """编码图像 → CLIP视觉嵌入（保持梯度流）"""
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = self.clip_model.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _compute_loss(self, image_features, text_features):
        """
        计算对齐损失：
        - cosine: 1 - cosine_similarity (让向量方向对齐)
        - l2:     ||E_v - E_t||^2 (让向量距离最小)
        """
        if self.loss_type == "cosine":
            similarity = F.cosine_similarity(image_features, text_features, dim=-1)
            return 1.0 - similarity.mean()
        else:
            return F.mse_loss(image_features, text_features)

    def attack(self, image_tensor, target_text, verbose=True):
        """
        执行跨模态嵌入对齐攻击

        Args:
            image_tensor: 原始图像 [1, 3, H, W]，范围 [0, 1]，float32
            target_text: 攻击指令文本
            verbose: 是否显示进度

        Returns:
            adv_image: 对抗图像 tensor [1, 3, H, W]
            perturbation: 扰动 tensor
            losses: loss变化记录
            metrics: 攻击过程指标
        """
        logger.info("=" * 70)
        logger.info(f"[CLIP-Align] 跨模态特征对齐攻击")
        logger.info(f"[CLIP-Align] 目标文本: '{target_text}'")
        logger.info(f"[CLIP-Align] Loss类型: {self.loss_type}")
        logger.info(f"[CLIP-Align] 参数: eps={self.epsilon:.4f}, alpha={self.alpha:.4f}, iters={self.num_iterations}")
        logger.info(f"[CLIP-Align] 量化感知: {self.quantize_aware}")

        image_tensor = image_tensor.to(self.device).float()
        original_image = image_tensor.clone().detach()

        text_features = self._encode_text(target_text)
        logger.info(f"[CLIP-Align] 文本embedding shape: {text_features.shape}")

        with torch.no_grad():
            orig_normalized = self._normalize_for_clip(original_image)
            orig_features = self._encode_image(orig_normalized)
            orig_sim = F.cosine_similarity(orig_features, text_features, dim=-1).item()
            logger.info(f"[CLIP-Align] 原始相似度: {orig_sim:.4f}")
            print(f"  原始图像与目标文本的CLIP相似度: {orig_sim:.4f}")

        if self.random_start:
            perturbation = torch.zeros_like(image_tensor).uniform_(
                -self.epsilon, self.epsilon
            ).to(self.device)
        else:
            perturbation = torch.zeros_like(image_tensor).to(self.device)

        if self.quantize_aware:
            adv_init = self._quantize_snap(torch.clamp(original_image + perturbation, 0, 1))
            perturbation = (adv_init - original_image).clone()

        perturbation.requires_grad = True

        losses = []
        similarities = []
        best_loss = float('inf')
        best_perturbation = perturbation.clone().detach()
        best_similarity = orig_sim

        iterator = tqdm(range(self.num_iterations), desc="CLIP-Align") if verbose else range(self.num_iterations)

        for i in iterator:
            adv_image = torch.clamp(original_image + perturbation, 0, 1)

            if self.quantize_aware:
                adv_quantized = self._quantize_snap(adv_image)
                adv_for_loss = adv_image + (adv_quantized - adv_image).detach()
            else:
                adv_for_loss = adv_image

            normalized = self._normalize_for_clip(adv_for_loss)
            image_features = self._encode_image(normalized)
            loss = self._compute_loss(image_features, text_features)

            loss_val = loss.item()
            losses.append(loss_val)

            with torch.no_grad():
                sim = F.cosine_similarity(image_features, text_features, dim=-1).item()
                similarities.append(sim)

            if loss_val < best_loss:
                best_loss = loss_val
                best_perturbation = perturbation.clone().detach()
                best_similarity = sim

            if i in [0, 1, 5, 10, 25, 50, 100, 200, 300, 500, 750, self.num_iterations - 1]:
                logger.info(
                    f"[Step {i}] Loss={loss_val:.6f}, "
                    f"Cosine Sim={sim:.4f}, Best Sim={best_similarity:.4f}"
                )

            if perturbation.grad is not None:
                perturbation.grad.zero_()

            loss.backward()

            if perturbation.grad is None:
                if i == 0:
                    logger.error("[Step 0] 梯度为None！")
                    print("  致命错误：梯度为None")
                return None, None, [], {}

            grad = perturbation.grad.detach()

            if i == 0:
                grad_norm = grad.norm().item()
                logger.info(f"[Step 0] 梯度范数: {grad_norm:.6f}")
                if grad_norm < 1e-10:
                    logger.error("[Step 0] 梯度为零！")
                else:
                    print(f"  梯度正常 (norm={grad_norm:.6f})")

            perturbation = perturbation.detach() - self.alpha * grad.sign()

            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            perturbation = torch.clamp(original_image + perturbation, 0, 1) - original_image

            if self.quantize_aware:
                adv_snapped = self._quantize_snap(
                    torch.clamp(original_image + perturbation, 0, 1)
                )
                perturbation = (adv_snapped - original_image).clone()

            perturbation.requires_grad = True

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'sim': f'{sim:.4f}',
                    'best': f'{best_similarity:.4f}'
                })

            if i % 100 == 0:
                torch.cuda.empty_cache()

        adv_image = torch.clamp(original_image + best_perturbation, 0, 1)

        if self.quantize_aware:
            adv_image = self._quantize_snap(adv_image)
            best_perturbation = adv_image - original_image

        with torch.no_grad():
            final_normalized = self._normalize_for_clip(adv_image)
            final_features = self._encode_image(final_normalized)
            final_sim = F.cosine_similarity(final_features, text_features, dim=-1).item()

        metrics = {
            'original_similarity': orig_sim,
            'final_similarity': final_sim,
            'best_similarity': best_similarity,
            'initial_loss': losses[0] if losses else 0,
            'final_loss': best_loss,
            'perturbation_linf': best_perturbation.abs().max().item(),
            'perturbation_l2': best_perturbation.norm().item(),
        }

        logger.info("=" * 70)
        logger.info(f"[CLIP-Align] 攻击完成")
        logger.info(f"[CLIP-Align] 相似度: {orig_sim:.4f} → {final_sim:.4f}")
        logger.info(f"[CLIP-Align] Loss: {losses[0]:.6f} → {best_loss:.6f}")
        logger.info(f"[CLIP-Align] 扰动 L_inf: {metrics['perturbation_linf']:.4f}")

        print(f"\n  攻击完成:")
        print(f"    CLIP相似度: {orig_sim:.4f} → {final_sim:.4f}")
        print(f"    扰动 L∞: {metrics['perturbation_linf']:.4f}")

        return adv_image, best_perturbation, losses, metrics

    def compute_similarity(self, image_tensor, text):
        """计算图像与文本的CLIP相似度（用于验证）"""
        image_tensor = image_tensor.to(self.device).float()
        with torch.no_grad():
            normalized = self._normalize_for_clip(image_tensor)
            image_features = self._encode_image(normalized)
            text_features = self._encode_text(text)
            similarity = F.cosine_similarity(image_features, text_features, dim=-1)
        return similarity.item()

    def compute_similarity_matrix(self, image_tensor, texts):
        """计算图像与多段文本的相似度矩阵"""
        image_tensor = image_tensor.to(self.device).float()
        results = {}
        with torch.no_grad():
            normalized = self._normalize_for_clip(image_tensor)
            image_features = self._encode_image(normalized)
            for text in texts:
                text_features = self._encode_text(text)
                sim = F.cosine_similarity(image_features, text_features, dim=-1).item()
                results[text] = sim
        return results
