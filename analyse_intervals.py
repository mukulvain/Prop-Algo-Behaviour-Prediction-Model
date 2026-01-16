import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, mean_squared_error

stock = "MINDTREE"

DATES = [
    "20082019",
    "21082019",
    "22082019",
    "23082019",
    "26082019",
    "27082019",
    "28082019",
]

INTERVALS = {
    "09:15 - 10:30": (0, 4500),
    "10:30 - 11:30": (4500, 8100),
    "11:30 - 12:30": (8100, 11700),
    "12:30 - 13:30": (11700, 15300),
    "13:30 - 14:30": (15300, 18900),
    "14:30 - 15:30": (18900, 22500),
}


def calculate_metrics(y_true_part, y_pred_part, df_true_bucket, df_pred_bucket):
    part_f1_macro = f1_score(y_true_part, y_pred_part, average="macro")
    cls_report = classification_report(y_true_part, y_pred_part, output_dict=True)
    mask = (y_true_part == 1) & (y_pred_part == 1)

    side_f1, price_rmse, qty_rmse = np.nan, np.nan, np.nan
    overlap_count = mask.sum()

    if overlap_count > 0:
        # Side F1 Score
        y_true_side = df_true_bucket.loc[mask, "next_side"].astype(int)
        y_pred_side = df_pred_bucket.loc[mask, "next_side"].astype(int)
        side_f1 = f1_score(y_true_side, y_pred_side, average="binary")

        # Price Delta RMSE
        y_true_price = df_true_bucket.loc[mask, "next_price_delta"]
        y_pred_price = df_pred_bucket.loc[mask, "next_price_delta"]
        price_rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))

        # Quantity RMSE (log scale)
        y_true_qty = df_true_bucket.loc[mask, "next_qty_log"]
        y_pred_qty = df_pred_bucket.loc[mask, "next_qty_log"]
        qty_rmse = np.sqrt(mean_squared_error(y_true_qty, y_pred_qty))

    return {
        "Participation F1": part_f1_macro,
        "Precision": cls_report["1"]["precision"] if "1" in cls_report else 0,
        "Recall": cls_report["1"]["recall"] if "1" in cls_report else 0,
        "Side F1": side_f1,
        "Price RMSE": price_rmse,
        "Qty RMSE": qty_rmse,
        "Trades": overlap_count,
    }


def eval_intervals(true_files, pred_files):

    agg_data = {k: {"true": [], "pred": []} for k in INTERVALS.keys()}
    print(f"Processing {len(true_files)} days of data...")

    for true_path, pred_path in zip(true_files, pred_files):
        df_t = pd.read_csv(true_path)
        df_p = pd.read_csv(pred_path)

        for interval_name, (start, end) in INTERVALS.items():
            mask_t = (df_t["period"] >= start) & (df_t["period"] < end)
            agg_data[interval_name]["true"].append(df_t[mask_t].copy())
            agg_data[interval_name]["pred"].append(df_p[mask_t].copy())

    results = []

    for interval_name, data in agg_data.items():
        full_true = pd.concat(data["true"], axis=0).reset_index(drop=True)
        full_pred = pd.concat(data["pred"], axis=0).reset_index(drop=True)

        y_true_part = full_true["next_participated"].astype(int)
        y_pred_part = full_pred["next_participated"].astype(int)

        metrics = calculate_metrics(y_true_part, y_pred_part, full_true, full_pred)
        metrics["Interval"] = interval_name
        results.append(metrics)

    df = pd.DataFrame(results)
    print(df)

    # Reorder columns for readability
    cols = [
        "Interval",
        "Participation F1",
        "Precision",
        "Recall",
        "Side F1",
        "Price RMSE",
        "Qty RMSE",
        "Trades",
    ]
    df = df[cols]

    csv_path = os.path.join("Outputs", f"intervals_{stock}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Metrics saved to: {csv_path}")
    print(df.to_string(index=False))

    print(f"\n[Generating] Creating visualization graphs...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Plot 1: F1 Scores
    df_f1 = df.melt(
        id_vars="Interval",
        value_vars=["Participation F1", "Side F1"],
        var_name="Metric",
        value_name="Score",
    )

    sns.lineplot(
        data=df_f1,
        x="Interval",
        y="Score",
        hue="Metric",
        style="Metric",
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Model Accuracy (F1 Score) by Time of Day", fontsize=14)
    axes[0].set_ylim(0, 1.0)
    for x, y in zip(range(len(df)), df["Participation F1"]):
        axes[0].text(x, y + 0.02, f"{y:.3f}", ha="center", color="black", fontsize=9)

    # Plot 2: RMSE (Dual Axis)
    ax2 = axes[1]
    sns.lineplot(
        data=df,
        x="Interval",
        y="Price RMSE",
        marker="s",
        color="tab:red",
        label="Price RMSE",
        ax=ax2,
        linewidth=2,
    )
    ax2.set_ylabel("Price RMSE", color="tab:red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.grid(False)  # Turn off grid for primary axis to avoid clutter

    ax2_twin = ax2.twinx()
    sns.lineplot(
        data=df,
        x="Interval",
        y="Qty RMSE",
        marker="^",
        color="tab:orange",
        label="Qty RMSE",
        ax=ax2_twin,
        linewidth=2,
    )
    ax2_twin.set_ylabel("Quantity RMSE (log)", color="tab:orange", fontsize=12)
    ax2_twin.tick_params(axis="y", labelcolor="tab:orange")

    axes[1].set_title("Prediction Error (RMSE) by Time of Day", fontsize=14)

    # Legend for Plot 2
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center")

    # Plot 3: Trade Counts
    sns.barplot(
        data=df,
        x="Interval",
        y="Trades",
        ax=axes[2],
        color="cornflowerblue",
        alpha=0.8,
    )
    axes[2].set_title("Number of Correctly Predicted Trades (Sample Size)", fontsize=14)
    for i, v in enumerate(df["Trades"]):
        axes[2].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    img_path = os.path.join("Outputs", f"metrics_{stock}.png")
    plt.savefig(img_path)
    print(f"[Saved] Visualization saved to: {img_path}")

    return df


true_files = [f"True/{stock}/True_{date}.csv" for date in DATES]
pred_files = [f"Pred/{stock}/Pred_{date}.csv" for date in DATES]
eval_intervals(true_files, pred_files)
