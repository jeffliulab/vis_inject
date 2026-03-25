"""
CLIP Image Encoder wrapper for AnyAttack.

Uses open_clip for loading CLIP ViT-B/32. The encoder is always frozen
and used as a surrogate model for self-supervised adversarial training.
"""

import torch
import torch.nn as nn
import open_clip


class CLIPEncoder(nn.Module):
    """Frozen CLIP image encoder used as surrogate for adversarial training."""

    CLIP_MODELS = {
        "ViT-B/32": ("ViT-B-32", "openai"),
        "ViT-B/16": ("ViT-B-16", "openai"),
        "ViT-L/14": ("ViT-L-14", "openai"),
    }

    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__()
        if model_name not in self.CLIP_MODELS:
            raise ValueError(f"Unsupported CLIP model: {model_name}. "
                             f"Available: {list(self.CLIP_MODELS.keys())}")

        arch, pretrained = self.CLIP_MODELS[model_name]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.normalize = open_clip.image_transform(
            self.model.visual.image_size[0]
            if hasattr(self.model.visual, "image_size")
            else 224,
            is_train=False,
        ).transforms[-1]  # extract Normalize transform

    @torch.no_grad()
    def encode_img(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to CLIP embedding space.

        Args:
            images: (B, 3, H, W) tensor in [0, 1] range.

        Returns:
            (B, embed_dim) float tensor of image embeddings.
        """
        images = self._normalize(images)
        return self.model.encode_image(images)

    def encode_img_with_grad(self, images: torch.Tensor) -> torch.Tensor:
        """Same as encode_img but allows gradient flow (for adversarial noise)."""
        images = self._normalize(images)
        return self.model.encode_image(images)

    @torch.no_grad()
    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        """Encode text strings to CLIP embedding space."""
        tokens = open_clip.tokenize(texts).to(device)
        return self.model.encode_text(tokens)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization (ImageNet CLIP mean/std)."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std
