"""
LLaVA wrapper for Universal Adversarial Attack.

Supports LLaVA-1.5-7B via LlavaForConditionalGeneration.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


class LlavaWrapper(MLLMWrapper):
    """Wrapper for LLaVA-1.5 models."""

    def load(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        print(f"Loading {info['short_name']} ({hf_id})...")
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.dtype = dtype
        self.img_size = info["img_size"]
        print(f"  Loaded. VRAM: ~{info['vram_bf16_gb']} GB")

    def _build_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        img = image.to(self.device, dtype=self.dtype)
        if img.shape[2:] != (self.img_size, self.img_size):
            img = F.interpolate(img, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        info = get_model_info(self.model_key)
        mean = torch.tensor(info["norm_mean"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(info["norm_std"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        return (img - mean) / std

    def compute_masked_ce_loss(self, image: torch.Tensor,
                               question: str, target_answer: str) -> torch.Tensor:
        pixel_values = self._build_pixel_values(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        text_with_answer = text + target_answer

        text_inputs = self.processor.tokenizer(
            text_with_answer, return_tensors="pt", padding=True
        ).to(self.device)

        prompt_only = self.processor.tokenizer(text, return_tensors="pt")
        prompt_len = prompt_only["input_ids"].shape[1]

        labels = text_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        outputs = self.model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        pixel_values = self._build_pixel_values(image).to(self.dtype)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        text_inputs = self.processor.tokenizer(
            text, return_tensors="pt", padding=True
        ).to(self.device)

        output_ids = self.model.generate(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[:, text_inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
