from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def direction_geometry(
    codebook: torch.Tensor, temperature: float, epsilon: float
) -> torch.Tensor:
    centered = codebook - codebook.mean(dim=0, keepdim=True)
    directions = F.normalize(centered, dim=1, eps=epsilon)
    mask = ~torch.eye(len(codebook), device=codebook.device, dtype=torch.bool)
    similarities = (directions @ directions.T)[mask]
    return temperature * (
        torch.logsumexp(similarities / temperature, dim=0)
        - math.log(similarities.numel())
    )


def align_codebook(
    book: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    temperature: float,
    direction_epsilon: float,
    scale_epsilon: float,
) -> None:
    before = book.detach().clone()
    center = before.mean(0)
    scale = (before - center).square().sum(1).mean().sqrt()
    if not torch.isfinite(before).all() or float(scale) <= scale_epsilon:
        raise ValueError("Invalid/degenerate codebook")
    y = torch.nn.Parameter((before - center) / scale)
    optimizer = torch.optim.Adam([y], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = direction_geometry(y, temperature, direction_epsilon / float(scale))
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(y.grad).all():
            raise ValueError("Non-finite geometry update")
        optimizer.step()
    with torch.no_grad():
        updated = center + scale * y
        if not torch.isfinite(updated).all():
            raise ValueError("Non-finite codebook writeback")
        book.copy_(updated)
