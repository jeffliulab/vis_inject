"""
Base MLLM wrapper for Universal Adversarial Attack.

Provides a unified interface for loading multimodal LLMs and computing
masked cross-entropy loss on target answer tokens.

Subclasses implement model-specific loading and input construction.
"""

import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLLMWrapper(abc.ABC):
    """
    Abstract wrapper for a multimodal LLM.

    All models must support:
    1. Loading with gradients through the vision encoder
    2. Computing masked CE loss for a (image, question) -> target_answer pair
    3. Generating text responses for evaluation
    """

    def __init__(self, model_key: str, device: torch.device):
        self.model_key = model_key
        self.device = device
        self.model = None
        self.processor = None

    @abc.abstractmethod
    def load(self):
        """Load model and processor. Must enable grad through vision encoder."""
        ...

    @abc.abstractmethod
    def compute_masked_ce_loss(
        self,
        image: torch.Tensor,
        question: str,
        target_answer: str,
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss on target answer tokens only.

        Args:
            image: (1, 3, H, W) adversarial image tensor in [0, 1].
            question: Text question/prompt.
            target_answer: Target answer string to force.

        Returns:
            Scalar loss tensor with grad.
        """
        ...

    @abc.abstractmethod
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        question: str,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate a text response for evaluation."""
        ...

    def unload(self):
        """Free model memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
