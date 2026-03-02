"""
Dataset utilities for AnyAttack training and evaluation.

Supports:
  - LAION-Art WebDataset (pre-training): streaming from .tar files
  - COCO (fine-tuning): target images for embedding extraction
  - ImageNet / generic folder (fine-tuning): clean images for perturbation
  - Single image loading (demo / evaluation)
"""

import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

try:
    import webdataset as wds
except ImportError:
    wds = None


# ── Transforms ────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ── LAION-Art WebDataset (pre-training) ──────────────────────────

def make_laion_dataloader(tar_dir: str, batch_size: int,
                          num_workers: int = 8,
                          max_shards: int = None) -> "wds.WebLoader":
    """
    Create a streaming DataLoader from LAION-Art WebDataset tar files.

    Args:
        tar_dir: Directory containing .tar shard files.
        batch_size: Per-GPU batch size.
        num_workers: DataLoader workers.
        max_shards: If set, only use the first N tar shards (useful when
                    the dataset is still downloading).

    Returns:
        WebLoader yielding (images_tensor, text_list) batches.
    """
    if wds is None:
        raise ImportError("webdataset is required for LAION loading. "
                          "Install with: pip install webdataset")

    tar_files = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    total_shards = len(tar_files)
    if max_shards is not None:
        tar_files = tar_files[:max_shards]
    print(f"Using {len(tar_files)} / {total_shards} shards")
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {tar_dir}")

    def handle_sample(sample):
        image, text = sample
        return TRAIN_TRANSFORM(image), text

    dataset = (
        wds.WebDataset(tar_files, resampled=True, shardshuffle=True,
                       handler=wds.warn_and_continue)
        .shuffle(5000)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg;png;webp", "txt")
        .map(handle_sample)
        .batched(batch_size)
        .with_epoch(len(tar_files))
    )

    return wds.WebLoader(dataset, batch_size=None, num_workers=num_workers,
                         pin_memory=True)


# ── ImageFolder datasets (fine-tuning) ───────────────────────────

def make_imagefolder_dataloader(root_dir: str, batch_size: int,
                                train: bool = True,
                                num_workers: int = 4) -> DataLoader:
    """
    Create DataLoader from an ImageFolder directory (ImageNet, etc.).
    Returns (images_tensor, labels) batches.
    """
    transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM
    dataset = torchvision.datasets.ImageFolder(root_dir, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True, drop_last=train,
    )


# ── COCO dataset via collate ─────────────────────────────────────

def coco_collate_fn(batch):
    """Collate function for COCO dataset from LAVIS."""
    images = []
    for item in batch:
        image = item["image"]
        if isinstance(image, Image.Image):
            image = TRAIN_TRANSFORM(image)
        images.append(image)
    return {"image": torch.stack(images)}


# ── Single image loading ─────────────────────────────────────────

def load_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load a single image as a (1, 3, H, W) tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)
