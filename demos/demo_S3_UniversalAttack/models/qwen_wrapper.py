"""
Qwen2.5-VL wrapper for Universal Adversarial Attack.

Implements the MLLMWrapper interface for Qwen2-VL / Qwen2.5-VL models.
Uses the processor to build correct input tokens (including the right
number of image placeholders), then replaces pixel_values with a
differentiable tensor for gradient-based optimization.
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


class QwenWrapper(MLLMWrapper):
    """Wrapper for Qwen2-VL / Qwen2.5-VL models."""

    def load(self):
        from transformers import AutoProcessor

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        is_qwen2_5 = "Qwen2.5" in hf_id or "qwen2.5" in hf_id.lower()
        if is_qwen2_5:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelCls

        print(f"Loading {info['short_name']} ({hf_id})...")
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = ModelCls.from_pretrained(
            hf_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.dtype = dtype
        self.img_size = info["img_size"]
        print(f"  Loaded. VRAM: ~{info['vram_bf16_gb']} GB")

    def _tensor_to_pil(self, image: torch.Tensor) -> Image.Image:
        img_np = (image[0].detach().cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)

    def _build_inputs(self, image: torch.Tensor, question: str,
                      target_answer: Optional[str] = None):
        """Build inputs using the processor for correct image token counts,
        then swap pixel_values with a differentiable tensor."""
        pil_img = self._tensor_to_pil(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if target_answer:
            text_with_answer = text + target_answer
        else:
            text_with_answer = text

        # Use full processor (with image) to get correct input_ids
        # including the right number of <|image_pad|> tokens
        inputs = self.processor(
            text=[text_with_answer],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Build labels: mask everything except target answer tokens
        labels = None
        if target_answer:
            prompt_inputs = self.processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
                padding=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100

        # Build differentiable pixel_values matching the processor output shape.
        # Qwen2-VL processor outputs (num_patches, flattened) where
        # flattened = temporal * 3 * patch_h * patch_w (e.g. 2*3*14*14 = 1176)
        proc_pv = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]  # (1, 3) -> [t, h, w]
        t_grid = grid_thw[0, 0].item()
        h_grid = grid_thw[0, 1].item()
        w_grid = grid_thw[0, 2].item()

        patch_size = 14
        temporal = 2
        target_h = h_grid * patch_size
        target_w = w_grid * patch_size

        img = image.to(self.device, dtype=self.dtype)
        if img.shape[2:] != (target_h, target_w):
            img = F.interpolate(img, size=(target_h, target_w),
                                mode="bilinear", align_corners=False)

        info = get_model_info(self.model_key)
        mean = torch.tensor(info["norm_mean"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(info["norm_std"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        img_normed = (img - mean) / std

        img_t = img_normed.expand(temporal, -1, -1, -1)  # (T, 3, H, W)
        patches = img_t.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # (T, 3, h_grid, w_grid, 14, 14)
        patches = patches.permute(2, 3, 0, 1, 4, 5).contiguous()
        # (h_grid, w_grid, T, 3, 14, 14)
        num_patches = h_grid * w_grid
        diff_pv = patches.reshape(num_patches, temporal * 3 * patch_size * patch_size)

        inputs["pixel_values"] = diff_pv.to(proc_pv.dtype)
        inputs["labels"] = labels

        return inputs

    def compute_masked_ce_loss(self, image: torch.Tensor,
                               question: str, target_answer: str) -> torch.Tensor:
        inputs = self._build_inputs(image, question, target_answer)
        labels = inputs.pop("labels")
        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        inputs = self._build_inputs(image, question)
        inputs.pop("labels", None)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
