import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

from src import (
    WINDOW_SIZE,
    Predictor,
    State,
    evaluate,
    preprocess,
    save_model,
    train_model,
)

# Configuration
FEATURES = [
    "spread_pct",
    "depth_ratio",
    "deep_depth_ratio",
    "imbalance",
    "mid_return",
    "volatility",
    "skewness",
    "kurtosis",
    "row_total_qty",
    "imbalance_lag1",
    "mid_return_lag1",
    "spread_pct_lag1",
    "imbalance_lag2",
    "mid_return_lag2",
    "spread_pct_lag2",
    "ofi",
    "mid_price_velocity",
    "mid_price_volatility",
]
DATES = [
    "19082019",
    "20082019",
    "21082019",
    "22082019",
    "23082019",
    "26082019",
    "27082019",
    "28082019",
    "29082019",
    "30082019",
]
MODEL_PATH = "models/multitask_lstm_final.pt"
SCALER_PATH = "models/feature_scaler_final.pkl"

# To store results for plotting
history = []

model = None
optimizer = None
scaler = StandardScaler()

print("Starting 10-Day Walk-Forward Simulation...")

for i in range(len(DATES) - 1):
    train_date = DATES[i]
    test_date = DATES[i + 1]

    print(
        f"\n{'='*20}\nIteration {i+1}: Train {train_date} -> Predict {test_date}\n{'='*20}"
    )

    # 1. Prepare Training Data (Day N)
    df_train = pd.read_csv(f"LOB/LOB_{train_date}.csv")
    df_train = preprocess(df_train)

    # Fit scaler only on the first day, transform thereafter
    if i == 0:
        X_train = scaler.fit_transform(df_train[FEATURES].fillna(0))
        joblib.dump(scaler, SCALER_PATH)
    else:
        X_train = scaler.transform(df_train[FEATURES].fillna(0))

    train_dataset = State(
        X_train,
        df_train["next_participated"],
        df_train["next_side"].fillna(0),
        df_train["next_price_delta"].fillna(0),
        df_train["next_qty_log"].fillna(0),
        window_size=WINDOW_SIZE,
    )

    # 2. Train/Fine-tune
    model, optimizer = train_model(
        train_dataset, FEATURES, model=model, optimizer=optimizer
    )
    save_model(model, optimizer, FEATURES, scaler, MODEL_PATH, SCALER_PATH)

    # 3. Predict Day N+1
    df_test = pd.read_csv(f"LOB/LOB_{test_date}.csv")
    df_test = preprocess(df_test)

    DF_TRUE = f"True/True_{test_date}.csv"
    DF_PRED = f"Pred/Pred_{test_date}.csv"

    df_test.to_csv(DF_TRUE, index=False)

    predictor = Predictor(DF_PRED, model, FEATURES, scaler)
    predictor.run(df_test)

    # 4. Evaluate and Track Metrics
    daily_metrics = evaluate(DF_TRUE, DF_PRED)
    daily_metrics["date"] = test_date
    history.append(daily_metrics)

# --- 5. PLOT IMPROVEMENT ---
perf_df = pd.DataFrame(history)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(perf_df["date"], perf_df["side_f1"], marker="o", label="Side F1")
plt.title("Side Prediction Accuracy (F1)")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(
    perf_df["date"], perf_df["price_rmse"], marker="o", color="red", label="Price RMSE"
)
plt.title("Price Delta Error (RMSE)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("improvement_plot.png")
plt.show()
