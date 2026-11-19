"""Train-only class weighting and multiclass focal loss."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def class_weights(labels):
    """Inverse-frequency weights of the three classes, clipped to [0.3, 3] (train labels only)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3)
    if (counts == 0).any():
        raise ValueError("All three classes must be present in train")
    return np.clip(len(labels) / (3 * counts), 0.3, 3).astype(np.float32)


class FocalLoss(nn.Module):
    """Multiclass focal loss with per-class weights ``alpha`` and focusing parameter ``gamma``."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("Focal gamma must be nonnegative")
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        """Weighted focal loss averaged over the batch."""
        logpt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        weight = self.alpha[target]
        return (-weight * (1 - logpt.exp()).pow(self.gamma) * logpt).sum() / weight.sum().clamp_min(1e-8)


def make_loss(name, labels, gamma=2.0):
    """Create the loss named ``ce``, ``weighted_ce`` or ``focal``; returns (loss, class weights)."""
    weights = class_weights(labels)
    if name == "ce":
        return nn.CrossEntropyLoss(), np.ones(3, np.float32)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=torch.from_numpy(weights)), weights
    if name == "focal":
        return FocalLoss(weights, gamma), weights
    raise ValueError(f"Unknown loss {name}")
