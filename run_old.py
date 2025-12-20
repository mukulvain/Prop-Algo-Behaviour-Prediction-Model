import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import (
    WINDOW_SIZE,
    Predictor,
    State,
    evaluate,
    load_model,
    preprocess,
    save_model,
    train_model,
)

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
MODEL_PATH = "models/multitask_lstm.pt"
SCALER_PATH = "models/feature_scaler.pkl"
DF_TRUE = "LOB/LOB_True_20082019.csv"
DF_PRED = "LOB/LOB_Predicted_20082019.csv"

df = pd.read_csv("LOB/LOB_19082019.csv")
df = preprocess(df)

# Scaling is mandatory for LSTMs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES].fillna(0))

dataset = State(
    X_scaled,
    df["next_participated"],
    df["next_side"].fillna(0),
    df["next_price_delta"].fillna(0),
    df["next_qty_log"].fillna(0),
    window_size=WINDOW_SIZE,
)
model, optimizer = train_model(dataset, FEATURES)
save_model(model, optimizer, FEATURES, scaler, MODEL_PATH, SCALER_PATH)
model, scaler = load_model(MODEL_PATH, SCALER_PATH)

df_test = pd.read_csv("LOB/LOB_20082019.csv")
df_test = preprocess(df_test)

pd.DataFrame(df_test).to_csv(DF_TRUE, index=False)
predictor = Predictor(DF_PRED, model, FEATURES, scaler)
predictor.run(df_test)

evaluate(DF_TRUE, DF_PRED)

