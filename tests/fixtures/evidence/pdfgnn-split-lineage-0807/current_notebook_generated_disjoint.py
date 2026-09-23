"""Generated-notebook control with every temporal target role disjoint."""

T_total = len(dates)
demand_matrix = zeros((N_articles, T_total))
P = cfg["context_length"]
K = cfg["forecast_horizon"]
model = build_model(context_length=P, forecast_horizon=K)

train_end = int(T_total * 0.8)
training_input = demand_matrix[:, :train_end]
trained_model = train_model(
    model,
    demand_matrix=training_input,
    val_split=0.1,
    time_stride=1,
)

history = demand_matrix[:, train_end - P:train_end]
actual = demand_matrix[:, train_end:train_end + K]
result = trained_model.predict(history)
finite_mask = isfinite(actual)
actual_finite = actual[finite_mask]
mean_finite = result.mean[finite_mask]
score = sqrt(mean((actual_finite - mean_finite) ** 2))
