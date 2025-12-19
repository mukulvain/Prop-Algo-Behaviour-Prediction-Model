import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    auc,
)


def evaluate(true_path, pred_path):
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)

    # EVALUATE PARTICIPATION (The Gatekeeper)
    y_true_part = df_true["next_participated"].astype(int)
    y_pred_part = df_pred["next_participated"].astype(int)

    print("=== PARTICIPATION METRICS ===")
    print(classification_report(y_true_part, y_pred_part))

    # EVALUATE SIDE (Conditional on True Positives)
    mask = (y_true_part == 1) & (y_pred_part == 1)

    if mask.sum() > 0:
        print("\n=== CONDITIONAL METRICS (Correctly Predicted Trades) ===")

        # Side Accuracy
        y_true_side = df_true.loc[mask, "next_side"].astype(int)
        y_pred_side = df_pred.loc[mask, "next_side"].astype(int)
        side_f1 = f1_score(y_true_side, y_pred_side)
        print(f"Side F1-Score: {side_f1:.4f}")

        # Price Delta RMSE
        y_true_price = df_true.loc[mask, "next_price_delta"]
        y_pred_price = df_pred.loc[mask, "next_price_delta"]
        price_rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
        print(f"Price Delta RMSE: {price_rmse:.6f}")

        # Quantity RMSE (on log scale)
        y_true_qty = df_true.loc[mask, "next_qty_log"]
        y_pred_qty = df_pred.loc[mask, "next_qty_log"]
        qty_rmse = np.sqrt(mean_squared_error(y_true_qty, y_pred_qty))
        print(f"Quantity (log) RMSE: {qty_rmse:.4f}")
    else:
        print("\nNo overlapping trades found for conditional evaluation.")

    # MARKET STATE RMSE (Global)
    # How much did the mid-price in your simulation drift from reality?
    mid_rmse = np.sqrt(mean_squared_error(df_true["mid_price"], df_pred["mid_price"]))
    print(f"\n=== GLOBAL MARKET IMPACT ===")
    print(f"Mid-Price RMSE: {mid_rmse:.4f}")
