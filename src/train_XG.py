import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from preprocess import preprocess

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

X = df[features]
y_part = df["next_participated"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_part, test_size=0.2, shuffle=False
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model_part = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)

model_part.fit(X_train, y_train)

y_proba = model_part.predict_proba(X_val)[:, 1]

print("\n=== PARTICIPATION MODEL ===")
print("ROC AUC :", roc_auc_score(y_val, y_proba))
print("PR  AUC :", average_precision_score(y_val, y_proba))

# ---- Threshold selection (balanced) ----
THRESHOLD = 0.7
y_pred = (y_proba > THRESHOLD).astype(int)

print(f"\nThreshold = {THRESHOLD}")
print(classification_report(y_val, y_pred, zero_division=0))

fi = pd.DataFrame(
    {"feature": features, "importance": model_part.feature_importances_}
).sort_values("importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(fi)

df_prop = df[df["next_participated"] == 1].copy()
X_prop = df_prop[features]

# ---- A. SIDE MODEL ----
y_side = df_prop["next_side"].astype(int)

model_side = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model_side.fit(X_prop, y_side)

y_side_pred = model_side.predict(X_prop)

print("\n=== SIDE MODEL ===")
print(classification_report(y_side, y_side_pred))

# ---- B. PRICE IMPACT MODEL ----
y_price = df_prop["next_price_delta"]

model_price = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

model_price.fit(X_prop, y_price)

y_price_pred = model_price.predict(X_prop)

print("\n=== PRICE IMPACT MODEL ===")
print("MAE :", mean_absolute_error(y_price, y_price_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_price, y_price_pred)))

# ---- C. QUANTITY MODEL ----
y_qty = df_prop["next_qty_log"]

model_qty = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

model_qty.fit(X_prop, y_qty)

y_qty_pred = model_qty.predict(X_prop)

print("\n=== QUANTITY MODEL ===")
print("MAE :", mean_absolute_error(y_qty, y_qty_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_qty, y_qty_pred)))

print("\nTraining Complete for August 19th.")
