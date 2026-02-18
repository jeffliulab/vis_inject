"""
模型加载器：Qwen2.5-VL-3B 端到端攻击 + 验证
架构: ViT (视觉编码) + PatchMerger (投影) + Qwen2.5-3B (语言模型)
梯度路径: image → pixel_values → ViT → Merger → Qwen2.5 → loss
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import warnings

warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, VISION_CONFIG

logger = logging.getLogger(__name__)


class QwenVLAttackModel:
    """Qwen2.5-VL-3B 统一模型：攻击 + 验证"""

    def __init__(self, model_name=MODEL_CONFIG["model_name"], device="cuda"):
        self.device = device
        self.dtype = torch.bfloat16 if MODEL_CONFIG["dtype"] == "bf16" else torch.float16
        self.image_size = VISION_CONFIG["image_size"]
        self.num_merged_tokens = VISION_CONFIG["num_merged_tokens"]
        self.patch_size = VISION_CONFIG["patch_size"]
        self.merge_size = VISION_CONFIG["merge_size"]
        self.temporal_patch_size = VISION_CONFIG["temporal_patch_size"]

        logger.info(f"加载 Qwen2.5-VL 模型: {model_name} ({self.dtype})")
        print(f"加载 Qwen2.5-VL 模型: {model_name}...")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=self.image_size * self.image_size,
            max_pixels=self.image_size * self.image_size,
            use_fast=False,
        )
        self.tokenizer = self.processor.tokenizer

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.img_mean = torch.tensor(
            VISION_CONFIG["image_mean"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std = torch.tensor(
            VISION_CONFIG["image_std"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        self._cache_attack_template()

        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"模型显存占用: {mem:.1f}GB")
            print(f"模型加载完成，显存: {mem:.1f}GB")
        else:
            print(f"模型加载完成")

    def _cache_attack_template(self):
        """预构建攻击用 input_ids 模板（不含 target，每次攻击时拼接）"""
        grid_h = self.image_size // self.patch_size
        grid_w = self.image_size // self.patch_size
        self.grid_thw = torch.tensor(
            [[1, grid_h, grid_w]], dtype=torch.long, device=self.device
        )

    def _image_to_pixel_values(self, image_tensor):
        """
        将 [1, 3, H, W] 的 [0,1] 图像转换为 Qwen2.5-VL 的 pixel_values 格式。
        全程可微分，支持梯度回传。

        Returns:
            pixel_values: [N_patches, C*T*pH*pW] 格式
        """
        normalized = (image_tensor - self.img_mean) / self.img_std
        normalized = normalized.to(self.dtype)

        PS = self.patch_size
        MS = self.merge_size
        T = self.temporal_patch_size
        C = 3
        H, W = self.image_size, self.image_size
        h_merge = H // (PS * MS)
        w_merge = W // (PS * MS)

        x = normalized.reshape(1, C, h_merge, MS, PS, w_merge, MS, PS)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, T, -1, -1)
        pixel_values = x.reshape(-1, C * T * PS * PS)

        return pixel_values

    def _build_attack_ids(self, question, target_text):
        """
        构建攻击用 input_ids 和 labels。
        模板: <|im_start|>system ... <|im_end|>
              <|im_start|>user <vision_start><image_pad>*N<vision_end>QUESTION<|im_end|>
              <|im_start|>assistant TARGET_TEXT<|im_end|>\n

        通过定位最后一个 <|im_end|> 来确定 target tokens 的位置，
        避免被尾部 \n 等额外 token 干扰。

        Returns:
            input_ids: [1, seq_len]
            labels: [1, seq_len]  (只有 target 部分有有效 label)
            target_len: int
            target_start: int  (target 在 input_ids 中的起始位置)
        """
        n_image_tokens = self.num_merged_tokens

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"<|vision_start|>{'<|image_pad|>' * n_image_tokens}<|vision_end|>{question}"},
            {"role": "assistant", "content": target_text},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        target_ids = self.tokenizer.encode(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        target_len = target_ids.shape[1]

        im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        seq = input_ids[0].tolist()
        im_end_pos = len(seq) - 1 - seq[::-1].index(im_end_id)

        target_start = im_end_pos - target_len
        labels = torch.full_like(input_ids, -100)
        labels[0, target_start: im_end_pos] = input_ids[0, target_start: im_end_pos]

        return input_ids, labels, target_len, target_start

    def compute_attack_loss(self, image_tensor, target_text, question="Describe this image."):
        """
        计算攻击 Loss (端到端梯度)
        直接使用 model.forward()，模型内部处理 position_ids / masked_scatter 等。

        Args:
            image_tensor: [1, 3, H, W] 范围 [0, 1]
            target_text: 注入目标文本
            question: 用户问题

        Returns:
            loss: 标量
            pred_text: 当前预测文本
        """
        pixel_values = self._image_to_pixel_values(image_tensor)
        input_ids, labels, target_len, target_start = self._build_attack_ids(question, target_text)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=self.grid_thw,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        with torch.no_grad():
            logits = outputs.logits
            # logits[i] predicts input_ids[i+1], so to predict target_start we need logits[target_start - 1]
            pred_start = target_start - 1
            pred_logits = logits[0, pred_start: pred_start + target_len]
            pred_ids = torch.argmax(pred_logits, dim=-1)
            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

        return loss, pred_text

    def generate(self, image_path, question="Describe this image."):
        """
        标准生成 (用于验证攻击效果)
        使用 processor + model.generate() 完整流程
        """
        try:
            from PIL import Image
            messages = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(image_path).convert('RGB')},
                {"type": "text", "text": question},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=None,
                padding=True, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                )
                trimmed = generated_ids[0, inputs['input_ids'].shape[1]:]
                answer = self.tokenizer.decode(trimmed, skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            logger.error(f"[Generate] 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_from_tensor(self, image_tensor, question="Describe this image.", max_new_tokens=100):
        """
        直接从 tensor 生成 (不经过 PNG 保存/加载)
        手动构建 pixel_values 并调用 model.generate()
        """
        with torch.no_grad():
            pixel_values = self._image_to_pixel_values(image_tensor)

            n_image_tokens = self.num_merged_tokens
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<|vision_start|>{'<|image_pad|>' * n_image_tokens}<|vision_end|>{question}"},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=self.grid_thw,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            trimmed = generated_ids[0, input_ids.shape[1]:]
            answer = self.tokenizer.decode(trimmed, skip_special_tokens=True)
        return answer.strip()

    def save_adversarial_image_and_test(self, adv_image_tensor, question, save_path):
        """保存对抗图像并验证"""
        from utils import tensor_to_pil

        adv_pil = tensor_to_pil(adv_image_tensor)
        adv_pil.save(save_path)
        logger.info(f"[保存] PNG: {save_path}")

        try:
            result_direct = self.generate_from_tensor(adv_image_tensor, question)
            logger.info(f"[验证-direct] '{result_direct}'")
        except Exception as e:
            logger.warning(f"[验证-direct] 失败: {e}")
            result_direct = "(direct验证失败)"

        try:
            result_png = self.generate(save_path, question)
            logger.info(f"[验证-png] '{result_png}'")
        except Exception as e:
            logger.warning(f"[验证-png] 失败: {e}")

        return result_direct


def load_model(model_name=MODEL_CONFIG["model_name"], device="cuda"):
    return QwenVLAttackModel(model_name, device)
