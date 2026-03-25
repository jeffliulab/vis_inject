"""
BLIP-2 / InstructBLIP wrapper for Universal Adversarial Attack.

Supports:
  - Blip2ForConditionalGeneration  (blip2_opt_2_7b, blip2_flan_t5_xl)
  - InstructBlipForConditionalGeneration  (instructblip_vicuna_7b)
"""

from typing import Optional

import torch
import torch.nn.functional as F

from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


class Blip2Wrapper(MLLMWrapper):
    """Wrapper for BLIP-2 and InstructBLIP models."""

    def load(self):
        from transformers import AutoProcessor

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        self.is_instructblip = "instructblip" in hf_id.lower()
        if self.is_instructblip:
            from transformers import InstructBlipForConditionalGeneration as ModelCls
        else:
            from transformers import Blip2ForConditionalGeneration as ModelCls

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

    def _build_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        """Resize and normalize image tensor for BLIP-2 vision encoder."""
        img = image.to(self.device, dtype=self.dtype)
        if img.shape[2:] != (self.img_size, self.img_size):
            img = F.interpolate(img, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)

        info = get_model_info(self.model_key)
        mean = torch.tensor(info["norm_mean"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(info["norm_std"], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        img = (img - mean) / std
        return img

    def compute_masked_ce_loss(self, image: torch.Tensor,
                               question: str, target_answer: str) -> torch.Tensor:
        pixel_values = self._build_pixel_values(image)

        if self.is_instructblip:
            inputs = self.processor(text=question, return_tensors="pt").to(self.device)
            prompt_ids = inputs["input_ids"]
            prompt_mask = inputs["attention_mask"]
            qformer_ids = inputs.get("qformer_input_ids")
            qformer_mask = inputs.get("qformer_attention_mask")
        else:
            inputs = self.processor(text=question, return_tensors="pt").to(self.device)
            prompt_ids = inputs["input_ids"]
            prompt_mask = inputs["attention_mask"]
            qformer_ids = None
            qformer_mask = None

        answer_tokens = self.processor.tokenizer(
            target_answer, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        answer_ids = answer_tokens["input_ids"]

        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        full_mask = torch.ones_like(full_ids)

        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100

        kwargs = {
            "pixel_values": pixel_values,
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "labels": labels,
        }
        if self.is_instructblip and qformer_ids is not None:
            kwargs["qformer_input_ids"] = qformer_ids
            kwargs["qformer_attention_mask"] = qformer_mask

        outputs = self.model(**kwargs)
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        pixel_values = self._build_pixel_values(image).to(self.dtype)

        if self.is_instructblip:
            inputs = self.processor(text=question, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=question, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
