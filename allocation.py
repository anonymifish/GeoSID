from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


@torch.no_grad()
def conditional_targets(
    reference: torch.Tensor, books: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        len(books) != 3
        or reference.ndim != 2
        or not len(reference)
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("Conditional targets require finite embeddings and three codebooks")
    residual = reference.detach().clone()
    prefix = []
    for book in books[:2]:
        labels = torch.cdist(residual, book).argmin(1)
        prefix.append(labels)
        residual -= book[labels]
    prefix = torch.stack(prefix, dim=1)
    cost = torch.cdist(residual, books[2]).square().double().cpu().numpy()
    if not np.isfinite(cost).all():
        raise ValueError("Non-finite conditional allocation cost")
    native = cost.argmin(1)
    target = native.copy()
    active = np.zeros(len(reference), dtype=bool)
    prefixes = prefix.cpu().numpy()
    _, inverse = np.unique(prefixes, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    for ids in groups:
        if len(np.unique(native[ids])) == len(ids):
            continue
        capacity = math.ceil(len(ids) / len(books[2]))
        slots = np.repeat(np.arange(len(books[2])), capacity)
        rows, cols = linear_sum_assignment(cost[ids][:, slots])
        assert np.array_equal(rows, np.arange(len(ids)))
        target[ids] = slots[cols]
        if np.bincount(target[ids], minlength=len(books[2])).max() > capacity:
            raise RuntimeError("Conditional allocation capacity invariant failed")
        active[ids] = True
    return (
        prefix,
        torch.as_tensor(target, device=reference.device),
        torch.as_tensor(active, device=reference.device),
    )


def assignment_loss(
    z: torch.Tensor,
    book: torch.Tensor,
    assigned: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = -torch.cdist(z, book).square() / temperature
    return F.cross_entropy(logits, assigned) * temperature / z.shape[1]


@torch.no_grad()
def capture_prefix_reference(
    reference: torch.Tensor,
    books: Sequence[torch.Tensor],
    *,
    temperature_fraction: float,
    scale_epsilon: float,
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    fixed_books = [book.detach().clone() for book in books[:2]]
    residual = reference.detach()
    labels = []
    for book in fixed_books:
        indices = torch.cdist(residual, book).argmin(1)
        labels.append(indices)
        residual = residual - book[indices]
    temperatures = [
        max(
            float((book - book.mean(0)).square().sum(1).mean())
            * temperature_fraction,
            scale_epsilon,
        )
        for book in fixed_books
    ]
    return torch.stack(labels, dim=1), fixed_books, temperatures


def prefix_adaptation_loss(
    z: torch.Tensor,
    books: Sequence[torch.Tensor],
    temperatures: Sequence[float],
) -> torch.Tensor:
    fixed_books = [book.detach() for book in books[:2]]
    with torch.no_grad():
        first = torch.cdist(z, fixed_books[0]).argmin(1)
        second = torch.cdist(z - fixed_books[0][first], fixed_books[1]).argmin(1)
    return (
        assignment_loss(z, fixed_books[0], first, temperatures[0])
        + assignment_loss(
            z - fixed_books[0][first], fixed_books[1], second, temperatures[1]
        )
    ) / 2


def continuous_conditional_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    ids: torch.Tensor,
    *,
    target_prefix: torch.Tensor,
    target_last: torch.Tensor,
    active: torch.Tensor,
    temperature: float,
    reference_prefix: torch.Tensor,
    reference_books: Sequence[torch.Tensor],
    reference_temperatures: Sequence[float],
    collision_weight: float,
) -> torch.Tensor:
    z = model.encoder(inputs)
    with torch.no_grad():
        sid = model.quantizer.encode(z)
        allowed = active[ids] & (sid[:, :2] == target_prefix[ids]).all(1)
        books = [book.detach() for book in model.quantizer.codebooks]
        prefix = books[0][sid[:, 0]] + books[1][sid[:, 1]]
    logits = -torch.cdist(z - prefix, books[2]).square() / temperature
    terms = F.cross_entropy(logits, target_last[ids], reduction="none")
    collision_loss = temperature / z.shape[1] * (terms * allowed).mean()
    labels = reference_prefix[ids]
    fixed_books = [book.detach() for book in reference_books]
    prefix_loss = (
        assignment_loss(z, fixed_books[0], labels[:, 0], reference_temperatures[0])
        + assignment_loss(
            z - fixed_books[0][labels[:, 0]],
            fixed_books[1],
            labels[:, 1],
            reference_temperatures[1],
        )
    ) / 2
    return collision_weight * collision_loss + prefix_loss
