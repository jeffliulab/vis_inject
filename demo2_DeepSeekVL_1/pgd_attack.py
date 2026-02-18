"""
PGD对抗攻击 - DeepSeek-VL 端到端方案
支持量化感知攻击(QAA)以提升PNG保存鲁棒性
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class PGDAttack:
    """PGD对抗攻击 - 端到端梯度攻击"""
    
    def __init__(self, model, epsilon=32/255, alpha=1/255, num_iterations=500,
                 random_start=True, quantize_aware=False):
        """
        quantize_aware可以是:
        - False: 标准攻击
        - True: QAA（每步snap到网格）
        - 'ste': 量化鲁棒性训练（STE，forward量化backward直通）
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.random_start = random_start
        self.quantize_aware = quantize_aware
        self.device = model.device
    
    def _quantize_snap(self, image):
        """将图像snap到uint8量化网格（模拟PNG保存/加载）"""
        return (image * 255).round() / 255.0
    
    def attack(self, image_tensor, question, target_text, verbose=True):
        """执行PGD攻击"""
        if self.quantize_aware == 'ste':
            mode_str = "STE(量化鲁棒性训练)"
        elif self.quantize_aware:
            mode_str = "QAA(量化感知)"
        else:
            mode_str = "标准"
        logger.info(f"[PGD] ========== {mode_str} PGD攻击 ==========")
        logger.info(f"[PGD] 目标: '{target_text}'")
        logger.info(f"[PGD] 量化感知: {self.quantize_aware}")
        logger.info(f"[PGD] 参数: eps={self.epsilon:.4f}, alpha={self.alpha:.4f}, iters={self.num_iterations}")
        
        original_image = image_tensor.clone().detach()
        
        # 初始化扰动
        if self.random_start:
            perturbation = torch.zeros_like(image_tensor).uniform_(-self.epsilon, self.epsilon).to(self.device)
        else:
            perturbation = torch.zeros_like(image_tensor).to(self.device)
        
        # 量化感知：初始化时就snap到网格
        if self.quantize_aware:
            adv_init = self._quantize_snap(torch.clamp(original_image + perturbation, 0, 1))
            perturbation = (adv_init - original_image).clone()
        
        perturbation.requires_grad = True
        
        losses = []
        best_loss = float('inf')
        best_perturbation = perturbation.clone().detach()
        
        desc = f"PGD{'(QAA)' if self.quantize_aware else ''}"
        iterator = tqdm(range(self.num_iterations), desc=desc) if verbose else range(self.num_iterations)
        
        for i in iterator:
            # 生成对抗样本
            adv_image = torch.clamp(original_image + perturbation, 0, 1)
            
            # 量化处理
            if self.quantize_aware == 'ste':
                # 方法E：量化鲁棒性训练(STE)
                # 每步都在量化后的图像上计算loss，梯度通过STE直通
                adv_quantized = self._quantize_snap(adv_image)
                adv_for_loss = adv_image + (adv_quantized - adv_image).detach()
            elif self.quantize_aware:
                # 方法C/D：QAA（每步snap后计算）
                adv_quantized = self._quantize_snap(adv_image)
                adv_for_loss = adv_image + (adv_quantized - adv_image).detach()
            else:
                # 方法A/B：标准（不做量化处理）
                adv_for_loss = adv_image
            
            # 计算loss
            try:
                loss, pred_text = self.model.compute_attack_loss(adv_for_loss, target_text)
            except Exception as e:
                if i == 0:
                    logger.error(f"[PGD-Step0] Loss计算失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    print(f"\n攻击失败: {e}")
                    return None, None, []
                continue
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            # 记录最佳
            if loss_val < best_loss:
                best_loss = loss_val
                best_perturbation = perturbation.clone().detach()
            
            # 日志
            if i in [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, self.num_iterations - 1]:
                logger.info(f"[PGD-Step{i}] Loss={loss_val:.4f}, Pred='{pred_text[:50]}'")
            
            if i == 0:
                logger.info(f"[PGD-Step0] Loss grad_fn: {loss.grad_fn}")
            
            # 清零梯度
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            
            # 反向传播
            loss.backward()
            
            if perturbation.grad is None:
                if i == 0:
                    logger.error("[PGD-Step0] 梯度为None！")
                    print("\n致命错误：梯度为None")
                return None, None, []
            
            grad = perturbation.grad.detach()
            
            if i == 0:
                grad_norm = grad.norm().item()
                logger.info(f"[PGD-Step0] 梯度范数: {grad_norm:.4f}")
                if grad_norm < 1e-10:
                    logger.error("[PGD-Step0] 梯度为0！")
                else:
                    logger.info(f"[PGD-Step0] ✓ 梯度非零")
                    print(f"  ✓ 梯度正常 (norm={grad_norm:.4f})")
            
            # FGSM更新
            perturbation = perturbation.detach() - self.alpha * grad.sign()
            
            # 投影
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            perturbation = torch.clamp(original_image + perturbation, 0, 1) - original_image
            
            # 量化snap（QAA和STE方法都做）
            if self.quantize_aware:
                adv_snapped = self._quantize_snap(torch.clamp(original_image + perturbation, 0, 1))
                perturbation = (adv_snapped - original_image).clone()
            
            perturbation.requires_grad = True
            
            # 进度条
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': f'{loss_val:.3f}', 'best': f'{best_loss:.3f}'})
            
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        # 最终对抗样本
        adv_image = torch.clamp(original_image + best_perturbation, 0, 1)
        
        # 量化感知：最终也snap
        if self.quantize_aware:
            adv_image = self._quantize_snap(adv_image)
            best_perturbation = (adv_image - original_image)
        
        logger.info(f"[PGD] ========== 攻击完成 ==========")
        logger.info(f"[PGD] Loss: {losses[0]:.4f} → {best_loss:.4f}")
        logger.info(f"[PGD] 扰动: L_inf={best_perturbation.abs().max().item():.4f}")
        
        return adv_image, best_perturbation, losses


def pgd_attack_wrapper(model, image_tensor, question, target_text,
                       epsilon=32/255, alpha=1/255, num_iterations=500,
                       quantize_aware=False, verbose=True):
    """PGD攻击包装函数"""
    attacker = PGDAttack(
        model=model,
        epsilon=epsilon,
        alpha=alpha,
        num_iterations=num_iterations,
        random_start=True,
        quantize_aware=quantize_aware
    )
    return attacker.attack(image_tensor, question, target_text, verbose=verbose)


def test_attack_success(model, adv_image, question, target_text, temp_dir="temp"):
    """测试攻击是否成功"""
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_adv.png")
    
    response = model.save_adversarial_image_and_test(adv_image, question, temp_path)
    
    target_lower = target_text.lower()
    response_lower = response.lower()
    full_match = target_lower in response_lower
    
    keywords = [w for w in target_text.split() if len(w) > 2]
    keyword_hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    keyword_ratio = keyword_hits / len(keywords) if keywords else 0
    
    min_ratio = 1.0 if len(keywords) <= 2 else 0.5
    success = full_match or keyword_ratio >= min_ratio
    
    logger.info(f"[验证] 输出: '{response}'")
    logger.info(f"[验证] 目标: '{target_text}'")
    logger.info(f"[验证] 匹配: full={full_match}, kw={keyword_hits}/{len(keywords)}")
    logger.info(f"[验证] 判定: {'成功' if success else '失败'}")
    
    return success, response
