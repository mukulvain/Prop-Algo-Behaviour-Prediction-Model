import torch
from torch.utils.data import Dataset


class State(Dataset):
    def __init__(self, features, part, side, price, qty, window_size=10):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.part = torch.tensor(part.values, dtype=torch.float32)
        self.side = torch.tensor(side.values, dtype=torch.float32)
        self.price = torch.tensor(price.values, dtype=torch.float32)
        self.qty = torch.tensor(qty.values, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        target_idx = idx + self.window_size
        return (
            x,
            self.part[target_idx],
            self.side[target_idx],
            self.price[target_idx],
            self.qty[target_idx],
        )
