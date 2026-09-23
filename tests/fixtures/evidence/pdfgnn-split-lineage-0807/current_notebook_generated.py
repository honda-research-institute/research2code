"""Reduced generated-notebook shape for the R2C-077 producer adapter."""

T_total = len(dates)
demand_matrix = zeros((N_articles, T_total))

trailing_zeros = opaque_trailing_zeros()
if trailing_zeros > 0:
    nonzero_cols = opaque_nonzero_columns()
    T_usable = nonzero_cols[-1] + 1
    demand_matrix = demand_matrix[:, :T_usable]
    T_total = T_usable

P = cfg["context_length"]
K = cfg["forecast_horizon"]
model = build_model(context_length=P, forecast_horizon=K)

train_end = int(T_total * 0.8)
trained_model = train_model(
    model,
    demand_matrix=demand_matrix,
    static_features=static_features,
    time_varying_matrix=time_varying_matrix,
    edge_index=edge_index,
    in_degree=in_degree,
    val_split=0.1,
    time_stride=1,
)

history = demand_matrix[:, train_end - P:train_end]
actual = demand_matrix[:, train_end:train_end + K]
result = forecast(trained_model, history=history)
finite_mask = isfinite(actual)
actual_finite = actual[finite_mask]
mean_finite = result.mean[finite_mask]
rmse = sqrt(mean((actual_finite - mean_finite) ** 2))
