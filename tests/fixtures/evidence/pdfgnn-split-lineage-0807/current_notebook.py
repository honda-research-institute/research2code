"""Reduced current notebook call site and boolean-mask metric lineage."""

from __future__ import annotations

import numpy as np


def run_notebook_fragment(model, demand_matrix, train_model, forecast):
    """Run the same call site with either reduced current trainer variant."""
    total_steps = demand_matrix.shape[1]
    context_length = model.context_length
    forecast_horizon = model.forecast_horizon

    train_end = int(total_steps * 0.8)
    trained_model = train_model(
        model,
        demand_matrix=demand_matrix,
        val_split=0.1,
        time_stride=1,
    )

    history = demand_matrix[
        :,
        train_end - context_length:train_end,
    ]
    actual = demand_matrix[
        :,
        train_end:train_end + forecast_horizon,
    ]
    result = forecast(trained_model, history=history)

    finite_mask = np.isfinite(actual)
    actual_finite = actual[finite_mask]
    mean_finite = result.mean[finite_mask]

    if len(actual_finite) == 0:
        return None

    rmse = np.sqrt(np.mean((actual_finite - mean_finite) ** 2))
    return rmse
