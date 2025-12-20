import torch
import torch.nn as nn


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )

        # Shared representation layer
        self.fc_shared = nn.Linear(hidden_size, 64)

        # Task-specific Heads
        self.head_part = nn.Linear(64, 1)  # Participation (Binary)
        self.head_side = nn.Linear(64, 1)  # Side (Binary Buy/Sell)
        self.head_price = nn.Linear(64, 1)  # Price Delta (Regression)
        self.head_qty = nn.Linear(64, 1)  # Quantity (Regression)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # Use the last output of the sequence
        shared = torch.relu(self.fc_shared(last_step))

        return (
            self.head_part(shared),
            self.head_side(shared),
            self.head_price(shared),
            self.head_qty(shared),
        )
