import pandas as pd
import torch

from .constants import WINDOW_SIZE


class Predictor:
    def __init__(
        self,
        output_file,
        model,
        features,
        scaler,
    ):
        self.output_file = output_file
        self.model = model
        self.features = features
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, df):
        print("Loading LOB file Data...")

        X = self.scaler.transform(df[self.features])
        results = [df.iloc[i].to_dict() for i in range(WINDOW_SIZE)]

        # Prediction loop
        for i in range(WINDOW_SIZE, len(df)):
            x_window = X[i - WINDOW_SIZE : i]
            x = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                p_part, p_side, p_price, p_qty = self.model(x)

            row = df.iloc[i].copy()
            active = p_part.item() > 0.7
            row["next_participated"] = float(active)

            if active:
                row["next_side"] = float(p_side.item() > 0.5)
                # Assuming price = mid_price * (1 + delta)
                row["next_price_delta"] = p_price.item()
                row["next_qty_log"] = p_qty.item()
            else:
                row["next_side"] = 0.0
                row["next_price_delta"] = 0.0
                row["next_qty_log"] = 0.0

            results.append(row.to_dict())

        # Save result
        pd.DataFrame(results).to_csv(self.output_file, index=False)
        print(f"Simulation Finished. Saved to {self.output_file}.")
