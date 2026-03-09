"""
DeepSeek-VL wrapper for Universal Adversarial Attack.

Requires: pip install deepseek-vl

DeepSeek-VL uses a custom 'multi_modality' architecture not in standard
transformers, so we load via the deepseek_vl package.
"""

import os
import sys
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


def _check_deepseek_vl():
    try:
        import deepseek_vl  # noqa: F401
        return True
    except ImportError:
        return False


@contextmanager
def _patch_meta_tensor_item():
    """Make Tensor.item() safe for meta tensors during model construction.

    deepseek_vl's siglip_vit.py calls .item() in __init__, which fails
    on meta tensors used by transformers >=5.x for lazy initialization.
    """
    _orig = torch.Tensor.item

    def _safe_item(self):
        if self.device.type == "meta":
            return 0.0
        return _orig(self)

    torch.Tensor.item = _safe_item
    try:
        yield
    finally:
        torch.Tensor.item = _orig


class DeepSeekWrapper(MLLMWrapper):
    """Wrapper for DeepSeek-VL models.

    Requires the deepseek_vl package:
        pip install deepseek-vl
    """

    def load(self):
        if not _check_deepseek_vl():
            raise ImportError(
                "DeepSeek-VL requires the 'deepseek_vl' package.\n"
                "Install with:  pip install deepseek-vl\n"
                "Or:  pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git"
            )

        from deepseek_vl.models import VLChatProcessor
        from transformers import AutoModelForCausalLM

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        print(f"Loading {info['short_name']} ({hf_id})...")
        self.vl_processor = VLChatProcessor.from_pretrained(hf_id)
        self.tokenizer = self.vl_processor.tokenizer
        self.processor = self.vl_processor

        # Patch deepseek_vl model class for transformers >=5.x compatibility
        from deepseek_vl.models.modeling_vlm import MultiModalityCausalLM
        if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
            MultiModalityCausalLM.all_tied_weights_keys = {}

        with _patch_meta_tensor_item():
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_id, dtype=dtype, trust_remote_code=True,
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

    def _prepare_conversation(self, question: str, pil_img: Image.Image,
                              target_answer: str = None):
        """Build DeepSeek-VL conversation and tokenize."""
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}",
                "images": [pil_img],
            },
        ]
        if target_answer:
            conversation.append({
                "role": "Assistant",
                "content": target_answer,
            })
        else:
            conversation.append({
                "role": "Assistant",
                "content": "",
            })

        prepared = self.vl_processor(
            conversations=conversation,
            images=[pil_img],
            force_batchify=True,
        ).to(self.device)
        return prepared

    def compute_masked_ce_loss(self, image: torch.Tensor,
                               question: str, target_answer: str) -> torch.Tensor:
        pil_img = self._tensor_to_pil(image)

        # Prepare with target answer
        inputs = self._prepare_conversation(question, pil_img, target_answer)

        # Prepare without target to find prompt length
        inputs_prompt = self._prepare_conversation(question, pil_img)
        prompt_len = inputs_prompt["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        # Replace pixel_values with differentiable tensor
        # DeepSeek-VL expects (batch, num_images, channels, height, width)
        diff_pv = self._get_differentiable_pixel_values(image)  # (1, 3, H, W)
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            target_shape = inputs.pixel_values.shape
            # Match dimensions: add num_images dim if needed
            while diff_pv.ndim < len(target_shape):
                diff_pv = diff_pv.unsqueeze(1)
            if diff_pv.shape != target_shape:
                diff_pv = diff_pv.expand(target_shape)
            inputs.pixel_values = diff_pv

        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        outputs = self.model.language_model(
            attention_mask=inputs.attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        pil_img = self._tensor_to_pil(image)
        inputs = self._prepare_conversation(question, pil_img)

        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        output_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
