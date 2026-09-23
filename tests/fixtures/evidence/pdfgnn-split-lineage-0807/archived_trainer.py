"""Reduced archived `_11` randperm, alias, and min-bound split leak."""

from __future__ import annotations

import torch


def train_model(
    model,
    demand_matrix,
    context_length: int,
    forecast_horizon: int,
    val_split_start: int,
    *,
    max_epochs: int = 2,
):
    """Mutate ``model`` and return the archived loss-history shape."""
    number_of_series, total_steps = demand_matrix.shape
    num_train_steps = val_split_start - context_length
    demand_log = torch.log1p(demand_matrix)
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    for _epoch in range(max_epochs):
        window_indices = torch.randperm(num_train_steps)
        epoch_loss = 0.0

        for window_idx in window_indices:
            t_start = window_idx
            t_end = t_start + context_length
            t_target_end = min(t_end + forecast_horizon, total_steps)
            article_indices = torch.arange(number_of_series)
            demand_target = demand_log[
                article_indices,
                t_end:t_target_end,
            ]

            loss = model.loss(demand_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss)

        val_end = val_split_start + forecast_horizon
        val_demand_target = demand_log[:, val_split_start:val_end]
        val_loss = model.loss(val_demand_target)
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: value.clone()
                for name, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"train_loss": train_losses, "val_loss": val_losses}
