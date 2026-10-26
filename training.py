from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from .allocation import (
    capture_prefix_reference,
    conditional_targets,
    continuous_conditional_loss,
    prefix_adaptation_loss,
)
from .geometry import align_codebook


@torch.no_grad()
def _encode_latents(
    model: torch.nn.Module, inputs: torch.Tensor, batch_size: int
) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.cat([
        model.encoder(batch.to(device)) for batch in inputs.split(batch_size)
    ])


def train_geosid(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    calibration_epochs: Sequence[int],
    geometry_layers: Sequence[int],
    geometry_steps: int,
    geometry_learning_rate: float,
    geometry_temperature: float,
    adaptation_steps: int,
    adaptation_learning_rate: float,
    adaptation_batch_size: int,
    recovery_epochs: int,
    collision_weight: float,
    assignment_temperature_fraction: float,
    gradient_clip_norm: float,
    seed: int,
    direction_epsilon: float = 1e-8,
    scale_epsilon: float = 1e-12,
) -> tuple[torch.nn.Module, torch.Tensor]:
    if (
        inputs.ndim != 2 or not len(inputs)
        or not inputs.is_floating_point() or not torch.isfinite(inputs).all()
    ):
        raise ValueError("Item embeddings must be a nonempty finite floating-point matrix")
    if any(type(value) is not int or value < 1 for value in (
        epochs, batch_size, geometry_steps, adaptation_steps, adaptation_batch_size
    )):
        raise ValueError("Epoch, batch, and update counts must be positive integers")
    if type(warmup_epochs) is not int or not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be an integer in [0, epochs)")
    if type(recovery_epochs) is not int or recovery_epochs < 0:
        raise ValueError("recovery_epochs must be a nonnegative integer")
    calibration_epochs = tuple(calibration_epochs)
    if (
        not calibration_epochs
        or any(type(epoch) is not int or not 1 <= epoch < epochs
               for epoch in calibration_epochs)
        or tuple(sorted(set(calibration_epochs))) != calibration_epochs
    ):
        raise ValueError("Calibration epochs must be sorted, unique, and precede the final epoch")
    geometry_layers = tuple(geometry_layers)
    if (
        not geometry_layers
        or any(type(layer) is not int or layer not in (1, 2, 3)
               for layer in geometry_layers)
        or tuple(sorted(set(geometry_layers))) != geometry_layers
    ):
        raise ValueError("Geometry layers must be sorted, unique one-based indices in [1, 3]")
    if any(not math.isfinite(value) or value <= 0 for value in (
        learning_rate, geometry_learning_rate, geometry_temperature,
        adaptation_learning_rate, assignment_temperature_fraction,
        gradient_clip_norm, direction_epsilon, scale_epsilon,
    )):
        raise ValueError("Learning rates, temperatures, clipping, and epsilons must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (
        weight_decay, collision_weight
    )):
        raise ValueError("Weight decay and collision weight must be nonnegative")
    books = list(model.quantizer.codebooks)
    if len(books) != 3 or any(
        book.ndim != 2 or len(book) < 2 or book.shape != books[0].shape
        for book in books
    ):
        raise ValueError("GeoSID requires three equally sized codebooks with at least two codes")
    if any(getattr(model.quantizer, "sinkhorn_epsilons", ())):
        raise ValueError("GeoSID requires native nearest-codeword quantization")
    if min(batch_size, len(inputs)) < len(books[0]):
        raise ValueError("The initialization batch must contain at least one item per code")

    torch.manual_seed(seed)
    tensor = inputs.detach().cpu()
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(tensor, torch.arange(len(tensor))),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.requires_grad_(True)
    model.train()
    model.initialize_codebooks(next(iter(loader))[0].to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    warmup_steps = warmup_epochs * len(loader)
    total_steps = epochs * len(loader)

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    reference_prefix = reference_books = reference_temperatures = None
    last_temperature = None
    recovery_remaining = 0

    for epoch in range(1, epochs + 1):
        targets = None
        if epoch > calibration_epochs[0]:
            model.eval()
            targets = conditional_targets(_encode_latents(model, tensor, batch_size), books)
        model.train()
        for batch, ids in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = model.loss(batch)
            if targets is not None:
                target_prefix, target_last, active = targets
                loss = loss + continuous_conditional_loss(
                    model, batch, ids.to(device),
                    target_prefix=target_prefix,
                    target_last=target_last,
                    active=active,
                    temperature=last_temperature,
                    reference_prefix=reference_prefix,
                    reference_books=reference_books,
                    reference_temperatures=reference_temperatures,
                    collision_weight=collision_weight,
                )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if recovery_remaining > 0:
                for book in books:
                    book.grad = None
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
        recovery_remaining = max(0, recovery_remaining - 1)

        if epoch in calibration_epochs:
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            for layer in geometry_layers:
                align_codebook(
                    books[layer - 1], steps=geometry_steps,
                    learning_rate=geometry_learning_rate,
                    temperature=geometry_temperature,
                    direction_epsilon=direction_epsilon,
                    scale_epsilon=scale_epsilon,
                )
            temperatures = [
                max(
                    float((book.detach() - book.detach().mean(0)).square().sum(1).mean())
                    * assignment_temperature_fraction,
                    scale_epsilon,
                )
                for book in books
            ]
            parameters = list(model.parameters())
            flags = [parameter.requires_grad for parameter in parameters]
            encoder_parameters = list(model.encoder.parameters())
            adaptation_optimizer = torch.optim.Adam(
                encoder_parameters, lr=adaptation_learning_rate
            )
            event_generator = torch.Generator().manual_seed(
                (seed + 1_000_003 * epoch) % (2**63 - 1)
            )
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                for parameter in encoder_parameters:
                    parameter.requires_grad_(True)
                for _ in range(adaptation_steps):
                    ids = torch.randperm(len(tensor), generator=event_generator)[
                        :adaptation_batch_size
                    ]
                    adaptation_optimizer.zero_grad(set_to_none=True)
                    loss = prefix_adaptation_loss(
                        model.encoder(tensor[ids].to(device)), books, temperatures
                    )
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite adaptation loss at epoch {epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        encoder_parameters, gradient_clip_norm, error_if_nonfinite=True
                    )
                    adaptation_optimizer.step()
            finally:
                adaptation_optimizer.zero_grad(set_to_none=True)
                for parameter, flag in zip(parameters, flags):
                    parameter.requires_grad_(flag)
            reference_prefix, reference_books, reference_temperatures = capture_prefix_reference(
                _encode_latents(model, tensor, batch_size), books,
                temperature_fraction=assignment_temperature_fraction,
                scale_epsilon=scale_epsilon,
            )
            last_temperature = temperatures[-1]
            recovery_remaining = recovery_epochs

    model.eval()
    with torch.no_grad():
        codes = torch.cat([
            model.quantizer.encode(model.encoder(batch.to(device))).cpu()
            for batch in tensor.split(batch_size)
        ])
    return model, codes
