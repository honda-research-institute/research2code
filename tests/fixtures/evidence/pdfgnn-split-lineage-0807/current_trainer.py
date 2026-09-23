"""Reduced current trainer with a private 90-percent-like fitting bound."""

from __future__ import annotations

import torch


def train_model(
    model,
    demand_matrix,
    *,
    val_split: float = 0.1,
    time_stride: int = 1,
):
    """Fit and select the model while independently inventing both ranges."""
    context_length = model.context_length
    forecast_horizon = model.forecast_horizon
    number_of_series, total_steps = demand_matrix.shape

    min_window = context_length + forecast_horizon
    number_of_validation_steps = max(
        int(total_steps * val_split),
        min_window,
    )
    train_end = total_steps - number_of_validation_steps
    if train_end < min_window:
        train_end = total_steps

    demand_tensor = torch.from_numpy(demand_matrix).float()
    optimizer = torch.optim.AdamW(model.parameters())

    t = context_length
    while t + forecast_horizon <= train_end:
        article_indices = torch.randperm(number_of_series)
        batch_target = demand_tensor[
            article_indices,
            t:t + forecast_horizon,
        ]
        loss = model.loss(batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += time_stride

    val_loss = 0.0
    number_of_selection_windows = 0
    t = context_length
    while t + forecast_horizon <= total_steps:
        val_target = demand_tensor[:, t:t + forecast_horizon]
        val_loss = val_loss + model.loss(val_target)
        number_of_selection_windows += 1
        t += time_stride

    best_score = float("inf")
    best_state = None
    if number_of_selection_windows > 0:
        val_loss = val_loss / number_of_selection_windows
        if val_loss < best_score:
            best_score = val_loss
            best_state = {
                name: value.clone()
                for name, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
