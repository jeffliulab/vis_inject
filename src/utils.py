"""
VisInject Utilities
====================
Shared functions for image loading, decoder loading, and metric computation.
"""

import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.clip_encoder import CLIPEncoder
from models.decoder import Decoder


def load_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load image as (1, 3, H, W) tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def load_decoder(path: str, embed_dim: int = 512, device: torch.device = None) -> Decoder:
    """Load AnyAttack Decoder with state dict key remapping for official weights."""
    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("decoder_state_dict", ckpt)
    remapped = {}
    for k, v in state.items():
        k = k.removeprefix("module.")
        k = k.replace("upsample_blocks.", "blocks.")
        k = k.replace("final_conv.", "head.")
        remapped[k] = v
    decoder.load_state_dict(remapped)
    return decoder


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10 * torch.log10(torch.tensor(mse)).item()


def compute_clip_similarities(
    clip_encoder,
    images: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute pairwise CLIP cosine similarities between named images.

    Args:
        clip_encoder: CLIPEncoder instance.
        images: dict of name -> (1, 3, H, W) tensor.

    Returns:
        dict of "name1_vs_name2" -> cosine similarity.
    """
    embeddings = {}
    for name, img in images.items():
        emb = clip_encoder.encode_img(img)
        embeddings[name] = F.normalize(emb, p=2, dim=1)

    names = list(embeddings.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            sim = (embeddings[names[i]] * embeddings[names[j]]).sum().item()
            results[key] = sim
    return results
