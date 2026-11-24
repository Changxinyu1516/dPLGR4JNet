from __future__ import annotations

import torch
from torch import nn


class NSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_2d = pred.squeeze(-1).transpose(0, 1)
        target_2d = target.squeeze(-1).transpose(0, 1)
        centered = target_2d - target_2d.mean(dim=1, keepdim=True)
        numerator = torch.sum((pred_2d - target_2d) ** 2, dim=1)
        denominator = torch.sum(centered**2, dim=1) + self.eps
        nse = 1.0 - numerator / denominator
        return 1.0 - nse.mean()


class StabilizedNSELoss(nn.Module):
    """The torchhydro runoff loss used by the water-balance experiment."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = ~target.isnan()
        observed = target[valid]
        if observed.numel() == 0:
            return pred[valid].sum()
        squared_error = (pred[valid] - observed).square().sum()
        total_variation = (observed - observed.mean()).square().sum()
        return squared_error / (total_variation.sqrt() + 0.1).square()
