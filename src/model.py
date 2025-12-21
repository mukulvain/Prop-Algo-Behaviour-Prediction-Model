import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        seq_len = lstm_output.shape[1]
        
        # Calculate Attention Scores
        # Q: Last hidden state (batch, 1, hidden) - "What is the context now?"
        # K: All hidden states (batch, seq, hidden) - "What happened before?"
        last_hidden = lstm_output[:, -1, :].unsqueeze(1)
        
        q = self.query(last_hidden)
        k = self.key(lstm_output)
        v = self.value(lstm_output)
        
        # scores: (batch, 1, seq_len)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale.to(lstm_output.device)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # context: (batch, 1, hidden) -> (batch, hidden)
        context = torch.bmm(attn_weights, v).squeeze(1)
        return context, attn_weights

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.attention = TemporalAttention(hidden_size)

        # Split Shared Layers
        # 1. Detection Stream (Classification)
        self.shared_cls = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Regression Stream (Price/Qty)
        self.shared_reg = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific Heads
        self.head_part = nn.Linear(64, 1)   # Participation
        self.head_side = nn.Linear(64, 1)   # Side
        self.head_price = nn.Linear(64, 1)  # Price Delta
        self.head_qty = nn.Linear(64, 1)    # Log Quantity

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Apply Attention
        context, _ = self.attention(lstm_out)
        
        # Split Flows
        feat_cls = self.shared_cls(context)
        feat_reg = self.shared_reg(context)

        return (
            self.head_part(feat_cls),
            self.head_side(feat_cls),
            self.head_price(feat_reg),
            self.head_qty(feat_reg),
        )
