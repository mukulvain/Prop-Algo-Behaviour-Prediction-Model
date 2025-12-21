import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .constants import WINDOW_SIZE


class Predictor:
    def __init__(self, output_file, model, features, scaler):
        self.output_file = output_file
        self.model = model
        self.features = features
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1024

    def run(self, df):
        print("Preparing Batch Data...")
        self.model.eval()

        # 1. Scale features and create sliding windows efficiently
        X = self.scaler.transform(df[self.features]).astype(np.float32)

        X_for_windows = X[:-1]
        window_shape = (WINDOW_SIZE, X.shape[1])
        windows = np.lib.stride_tricks.sliding_window_view(X_for_windows, window_shape)
        windows = windows.squeeze(1).copy()

        print(f"Window Shape: {windows.shape}")

        # 2. Setup DataLoader for batching
        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_part, all_side, all_price, all_qty = [], [], [], []

        print(f"Running Inference on {len(windows)} windows...")
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                l_part, l_side, l_price, l_qty = self.model(x)

                # Collect raw outputs
                all_part.append(torch.sigmoid(l_part).cpu().numpy())
                all_side.append(torch.sigmoid(l_side).cpu().numpy())
                all_price.append(l_price.cpu().numpy())
                all_qty.append(l_qty.cpu().numpy())

        # Flatten results
        p_part = np.concatenate(all_part).flatten()
        p_side = np.concatenate(all_side).flatten()
        l_price = np.concatenate(all_price).flatten()
        l_qty = np.concatenate(all_qty).flatten()

        # 3. Apply Logic Vectorially using Numpy
        # Initialize columns with 0
        df["next_participated"] = 0.0
        df["next_side"] = 0.0
        df["next_price_delta"] = 0.0
        df["next_qty_log"] = 0.0

        active_mask = p_part > 0.65
        idx = slice(WINDOW_SIZE, len(df))
        df.loc[df.index[idx], "next_participated"] = active_mask.astype(float)

        # Apply conditional logic: Only set side/price/qty if active_mask is True
        # If False, they remain 0.0 as initialized
        df.loc[df.index[idx], "next_side"] = np.where(
            active_mask, (p_side > 0.5).astype(float), 0.0
        )
        df.loc[df.index[idx], "next_price_delta"] = np.where(active_mask, l_price, 0.0)
        df.loc[df.index[idx], "next_qty_log"] = np.where(active_mask, l_qty, 0.0)

        # 4. Save result
        df.to_csv(self.output_file, index=False)
        print(f"Simulation Finished. Saved to {self.output_file}.")
