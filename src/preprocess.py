import numpy as np

from .constants import WINDOW_SIZE


def preprocess(df):
    g = df.groupby("symbol")

    df["next_participated"] = g["prop_participated"].shift(-1).astype(float)
    df["next_side"] = g["is_prop_buy"].shift(-1).astype(float)
    df["next_price_delta"] = (g["prop_price"].shift(-1) / g["mid_price"].shift(-1)) - 1
    df["next_qty_log"] = np.log1p(g["prop_qty"].shift(-1))

    df = df.dropna(subset=["next_participated"]).copy()

    df["spread_pct"] = df["spread"] / df["mid_price"]
    df["depth_ratio"] = df["best_bid_depth"] / (df["best_ask_depth"])
    df["deep_depth_ratio"] = df["deep_bid_depth"] / (df["deep_ask_depth"])
    df["mid_return"] = g["mid_price"].transform(lambda x: np.log(x).diff()).fillna(0)

    df["prev_bid"] = g["best_bid"].shift(1)
    df["prev_ask"] = g["best_ask"].shift(1)
    df["prev_bid_depth"] = g["best_bid_depth"].shift(1)
    df["prev_ask_depth"] = g["best_ask_depth"].shift(1)

    g = df.groupby("symbol")
    df["ofi_bid"] = np.where(
        df["best_bid"] > df["prev_bid"],
        df["best_bid_depth"],
        np.where(
            df["best_bid"] < df["prev_bid"],
            -df["prev_bid_depth"],
            df["best_bid_depth"] - df["prev_bid_depth"],
        ),
    )
    df["ofi_ask"] = np.where(
        df["best_ask"] < df["prev_ask"],
        -df["best_ask_depth"],
        np.where(
            df["best_ask"] > df["prev_ask"],
            df["prev_ask_depth"],
            -(df["best_ask_depth"] - df["prev_ask_depth"]),
        ),
    )
    df["ofi"] = df["ofi_bid"] + df["ofi_ask"]

    df["mid_price_velocity"] = g["mid_price"].transform(
        lambda x: (x - x.shift(WINDOW_SIZE - 1)) / WINDOW_SIZE
    )

    df["mid_price_volatility"] = (
        g["mid_return"].rolling(WINDOW_SIZE).std().reset_index(level=0, drop=True)
    )

    for col in ["imbalance", "mid_return", "spread_pct", "volatility"]:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag2"] = g[col].shift(2)

    return df
