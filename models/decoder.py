"""
AnyAttack Decoder Network.

Takes a CLIP embedding (512-dim for ViT-B/32) and generates an adversarial
noise image (3 x 224 x 224). The noise is clamped externally to [-eps, eps].

Architecture:
  FC(512 -> 256*14*14) -> 4x(ResBlock + UpBlock) -> Conv(16->3)
  ResBlocks include EfficientAttention for spatial self-attention.

Adapted from: https://github.com/jiamingzhang94/AnyAttack/blob/master/models/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    """Linear-complexity spatial self-attention (O(N*C^2) instead of O(N^2*C))."""

    def __init__(self, in_channels: int, key_channels: int,
                 head_count: int, value_channels: int):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.size()
        keys = self.keys(x).reshape(n, self.key_channels, h * w)
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(x).reshape(n, self.value_channels, h * w)

        head_key_ch = self.key_channels // self.head_count
        head_val_ch = self.value_channels // self.head_count

        attended = []
        for i in range(self.head_count):
            k = F.softmax(keys[:, i * head_key_ch:(i + 1) * head_key_ch, :], dim=2)
            q = F.softmax(queries[:, i * head_key_ch:(i + 1) * head_key_ch, :], dim=1)
            v = values[:, i * head_val_ch:(i + 1) * head_val_ch, :]
            context = k @ v.transpose(1, 2)
            out = (context.transpose(1, 2) @ q).reshape(n, head_val_ch, h, w)
            attended.append(out)

        aggregated = torch.cat(attended, dim=1)
        return self.reprojection(aggregated) + x


class ResBlock(nn.Module):
    """Residual block with EfficientAttention."""

    def __init__(self, in_ch: int, out_ch: int,
                 key_ch: int, head_count: int, val_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.attention = EfficientAttention(out_ch, key_ch, head_count, val_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        return self.act(out + residual)


class UpBlock(nn.Module):
    """2x spatial upsampling with conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(self.up(x))))


class Decoder(nn.Module):
    """
    AnyAttack noise generator: CLIP embedding -> adversarial noise image.

    Args:
        embed_dim: Input embedding dimension (512 for ViT-B/32, 1024 for ViT-L/14).
        img_channels: Output image channels (3 for RGB).
        img_size: Output spatial resolution (224).
    """

    def __init__(self, embed_dim: int = 512, img_channels: int = 3, img_size: int = 224):
        super().__init__()
        self.init_size = img_size // 16  # 14 for 224

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256 * self.init_size ** 2)
        )

        self.blocks = nn.ModuleList([
            ResBlock(256, 256, 64, 8, 256),
            UpBlock(256, 128),
            ResBlock(128, 128, 32, 8, 128),
            UpBlock(128, 64),
            ResBlock(64, 64, 16, 8, 64),
            UpBlock(64, 32),
            ResBlock(32, 32, 8, 8, 32),
            UpBlock(32, 16),
            ResBlock(16, 16, 4, 8, 16),
        ])

        self.head = nn.Conv2d(16, img_channels, 3, 1, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate noise from CLIP embedding.

        Args:
            embedding: (B, embed_dim) CLIP image embedding.

        Returns:
            (B, 3, img_size, img_size) raw noise (NOT clamped to [-eps, eps]).
        """
        out = self.fc(embedding.float().view(embedding.size(0), -1))
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        for block in self.blocks:
            out = block(out)
        return self.head(out)
