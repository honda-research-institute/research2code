"""Reduced archived `_11` notebook shape for the R2C-077 adapter."""

demand_matrix = opaque_loader()
last_valid_idx = opaque_last_valid_index()
T_useful = last_valid_idx + 1
demand_filled = np.nan_to_num(demand_matrix)
demand_tensor = torch.as_tensor(demand_filled)

P = cfg["context_length"]
K = cfg["forecast_horizon"]
val_split_start = T_useful - K
model = build_model(context_length=P, forecast_horizon=K)

train_model(
    model=model,
    demand_matrix=demand_tensor,
    context_length=P,
    forecast_horizon=K,
    val_split_start=val_split_start,
    max_epochs=2,
)

history = demand_filled[:, val_split_start - P:val_split_start]
actual = demand_filled[:, val_split_start:val_split_start + K]
result = forecast(model=model, history=history)
finite_mask = isfinite(actual)
actual_finite = actual[finite_mask]
forecast_finite = result.mean[finite_mask]
score = sqrt(mean((actual_finite - forecast_finite) ** 2))
