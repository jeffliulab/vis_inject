"""
模型加载器：DeepSeek-VL 1.3B 端到端攻击 + 验证
架构: SigLIP-L (视觉编码) + MLP Aligner (投影) + LLaMA-1.3B (语言模型)
梯度路径: image → SigLIP → MLP → LLaMA → loss (无 torch.no_grad 阻断)
"""

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import logging
import warnings
import os

warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, VISION_CONFIG

logger = logging.getLogger(__name__)


class DeepSeekVLAttackModel:
    """DeepSeek-VL 1.3B 统一模型：攻击 + 验证"""

    def __init__(self, model_name=MODEL_CONFIG["model_name"], device="cuda"):
        self.device = device
        self.dtype = torch.bfloat16 if MODEL_CONFIG["dtype"] == "bf16" else torch.float16
        self.image_size = VISION_CONFIG["image_size"]
        self.num_patches = VISION_CONFIG["num_patches"]

        logger.info(f"加载 DeepSeek-VL 模型: {model_name} ({self.dtype})")
        print(f"加载 DeepSeek-VL 模型: {model_name}...")

        # 1. 加载 VLChatProcessor (tokenizer + image_processor)
        self.processor = VLChatProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        # 2. 加载模型
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

        # 冻结所有参数（只优化图像像素）
        self.model.requires_grad_(False)

        # 3. 获取组件引用
        self.vision_model = self.model.vision_model
        self.aligner = self.model.aligner
        self.language_model = self.model.language_model
        self.embed_tokens = self.language_model.get_input_embeddings()

        # 4. 预计算 prompt 模板的 token IDs
        # DeepSeek-VL 使用 "deepseek" 对话模板:
        #   "{system}\n\nUser: <image_placeholder>{question}\n\nAssistant:"
        # 我们将 prompt 分为 image 前后两部分并缓存 token IDs
        self._cache_prompt_tokens()

        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"模型显存占用: {mem:.1f}GB")
            print(f"模型加载完成，显存: {mem:.1f}GB")
        else:
            print(f"模型加载完成")

    def _cache_prompt_tokens(self):
        """预计算 prompt 模板各段的 token IDs"""
        system_prompt = self.processor.system_prompt
        # 构造: "{system}\n\nUser: " + [VISION_576] + "{question}\n\nAssistant:"
        self.prompt_before = system_prompt + "\n\nUser: "
        self.t_before = self.tokenizer.encode(
            self.prompt_before, add_special_tokens=True
        )

    def _compute_vision_embeds(self, image_tensor):
        """
        计算视觉 Embedding (带梯度传播)
        CLIPVisionTower.forward() 内部自带 Normalize(mean=0.5, std=0.5)，
        因此输入必须是 [0, 1] 范围，不要额外归一化。

        Args:
            image_tensor: [1, 3, 384, 384] 范围 [0, 1]

        Returns:
            vision_embeds: [1, 576, 2048] 可直接拼入 embedding 序列
        """
        vision_features = self.vision_model(image_tensor.to(self.dtype))
        return self.aligner(vision_features)

    def compute_attack_loss(self, image_tensor, target_text, question="Describe this image."):
        """
        计算攻击 Loss (端到端梯度)

        构造序列: [prompt_before] + [VISION_576] + [question + "\\n\\nAssistant:"] + [TARGET]
        Labels:   [-100 .....................................................] + [TARGET_IDS]

        Args:
            image_tensor: [1, 3, 384, 384] 范围 [0, 1]
            target_text: 注入目标文本
            question: 用户问题

        Returns:
            loss: 标量
            pred_text: 当前预测文本 (用于监控)
        """
        # 1. 计算视觉 Embedding (vision tower 内部自带归一化)
        vision_embeds = self._compute_vision_embeds(image_tensor)  # [1, 576, 2048]

        # 2. 构造文本各段 token IDs
        prompt_after = question + "\n\nAssistant:"
        t_before = torch.tensor(
            [self.t_before], dtype=torch.long, device=self.device
        )
        t_after = torch.tensor(
            [self.tokenizer.encode(prompt_after, add_special_tokens=False)],
            dtype=torch.long, device=self.device,
        )
        t_target = torch.tensor(
            [self.tokenizer.encode(target_text, add_special_tokens=False)],
            dtype=torch.long, device=self.device,
        )

        # 3. 文本 Embedding
        embeds_before = self.embed_tokens(t_before)
        embeds_after = self.embed_tokens(t_after)
        embeds_target = self.embed_tokens(t_target)

        # 4. 拼接完整 Embedding 序列
        inputs_embeds = torch.cat([
            embeds_before,
            vision_embeds,        # [1, 576, 2048]
            embeds_after,
            embeds_target,
        ], dim=1)

        # 5. Labels: 只计算 target 部分的 loss
        prefix_len = t_before.shape[1] + self.num_patches + t_after.shape[1]
        target_len = t_target.shape[1]

        labels = torch.cat([
            torch.full((1, prefix_len), -100, dtype=torch.long, device=self.device),
            t_target,
        ], dim=1)

        # 6. Attention mask
        attention_mask = torch.ones(
            1, inputs_embeds.shape[1], dtype=torch.long, device=self.device
        )

        # 7. 前向传播 (直接调用 language_model，绕过 MultiModalityCausalLM 的视觉处理)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        # 8. 监控预测
        with torch.no_grad():
            logits = outputs.logits
            pred_logits = logits[0, prefix_len - 1: prefix_len - 1 + target_len]
            pred_ids = torch.argmax(pred_logits, dim=-1)
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

        return loss, pred_text

    def generate(self, image_path, question="Describe this image."):
        """
        标准生成 (用于验证攻击效果)
        使用 VLChatProcessor + prepare_inputs_embeds + language_model.generate()
        """
        try:
            conversation = [
                {"role": "User", "content": f"<image_placeholder>{question}", "images": [image_path]},
                {"role": "Assistant", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(self.device, dtype=self.dtype)

            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=200,
                    do_sample=False,
                    use_cache=True,
                )

            answer = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"[Generate] 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_from_tensor(self, image_tensor, question="Describe this image.", max_new_tokens=100):
        """
        直接从 tensor 生成 (不经过 PNG 保存/加载)
        手动构建 embedding 序列，使用 language_model.generate()
        LLaMA 标准 KV cache 可正常工作。
        """
        with torch.no_grad():
            vision_embeds = self._compute_vision_embeds(image_tensor)

            prompt_after = question + "\n\nAssistant:"
            t_before = torch.tensor(
                [self.t_before], dtype=torch.long, device=self.device
            )
            t_after = torch.tensor(
                [self.tokenizer.encode(prompt_after, add_special_tokens=False)],
                dtype=torch.long, device=self.device,
            )

            embeds_before = self.embed_tokens(t_before)
            embeds_after = self.embed_tokens(t_after)

            inputs_embeds = torch.cat([
                embeds_before,
                vision_embeds,
                embeds_after,
            ], dim=1)

            attention_mask = torch.ones(
                1, inputs_embeds.shape[1], dtype=torch.long, device=self.device
            )

            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        return self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        ).strip()

    def save_adversarial_image_and_test(self, adv_image_tensor, question, save_path):
        """保存对抗图像并验证"""
        from utils import tensor_to_pil

        adv_pil = tensor_to_pil(adv_image_tensor)
        adv_pil.save(save_path)
        logger.info(f"[保存] PNG: {save_path}")

        # 验证1: 直接 tensor (无损基准)
        try:
            result_direct = self.generate_from_tensor(adv_image_tensor, question)
            logger.info(f"[验证-direct] '{result_direct}'")
        except Exception as e:
            logger.warning(f"[验证-direct] 失败: {e}")
            result_direct = "(direct验证失败)"

        # 验证2: PNG 重载 (模拟真实场景)
        try:
            result_png = self.generate(save_path, question)
            logger.info(f"[验证-png] '{result_png}'")
        except Exception as e:
            logger.warning(f"[验证-png] 失败: {e}")

        return result_direct


def load_model(model_name=MODEL_CONFIG["model_name"], device="cuda"):
    return DeepSeekVLAttackModel(model_name, device)
