import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import State, preprocess, save_model, train_model

df = pd.read_csv("LOB/LOB_19082019.csv")
df = preprocess(df)

features = [
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

# Scaling is mandatory for LSTMs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features].fillna(0))

dataset = State(
    X_scaled,
    df["next_participated"],
    df["next_side"].fillna(0),
    df["next_price_delta"].fillna(0),
    df["next_qty_log"].fillna(0),
    window_size=10,
)
model, optimizer = train_model(dataset, features)
save_model(model, optimizer, features, scaler)
