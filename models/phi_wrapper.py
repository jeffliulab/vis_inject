"""
Phi-3.5-Vision wrapper for Universal Adversarial Attack.

Uses AutoModelForCausalLM with trust_remote_code for the custom Phi vision arch.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


class PhiWrapper(MLLMWrapper):
    """Wrapper for Phi-3.5-Vision-Instruct."""

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        print(f"Loading {info['short_name']} ({hf_id})...")
        self.processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, trust_remote_code=True,
            _attn_implementation="eager",
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

    def _get_differentiable_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
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
        pil_img = self._tensor_to_pil(image)

        prompt = f"<|image_1|>\n{question}"
        full_text = f"{prompt}\n{target_answer}"

        inputs = self.processor(
            text=full_text,
            images=[pil_img],
            return_tensors="pt",
        ).to(self.device)

        prompt_inputs = self.processor(
            text=prompt,
            images=[pil_img],
            return_tensors="pt",
        ).to(self.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        # Swap in differentiable pixel values
        diff_pv = self._get_differentiable_pixel_values(image)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            target_shape = inputs["pixel_values"].shape
            if diff_pv.shape != target_shape:
                diff_pv = diff_pv.expand(target_shape)
            inputs["pixel_values"] = diff_pv

        outputs = self.model(
            **{k: v for k, v in inputs.items() if k != "labels"},
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        pil_img = self._tensor_to_pil(image)
        prompt = f"<|image_1|>\n{question}"

        inputs = self.processor(
            text=prompt,
            images=[pil_img],
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
