import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import MultiTaskLSTM
from .losses import MultiTaskLoss
from .constants import HIDDEN_SIZE


def train_model(
    dataset, features, model=None, optimizer=None, criterion=None, epochs=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if criterion is None:
        from .losses import MultiTaskLoss
        criterion = MultiTaskLoss(num_tasks=4).to(device)

    if model is None:
        model = MultiTaskLSTM(input_size=len(features), hidden_size=HIDDEN_SIZE).to(
            device
        )
        # Optimizer must include both model params and loss params (log_vars)
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=0.0005
        )

    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # --- TRAINING LOOP ---
    print("Starting Multi-Task Training with Uncertainty Weighting...")
    model.train()
    criterion.train()

    for epoch in range(epochs):
        total_loss = 0
        part_loss_sum, side_loss_sum, price_loss_sum, qty_loss_sum = 0, 0, 0, 0
        count_valid_batches = 0

        for x_batch, y_part, y_side, y_price, y_qty in loader:
            x_batch = x_batch.to(device)
            targets = (
                y_part.to(device),
                y_side.to(device),
                y_price.to(device),
                y_qty.to(device),
            )

            optimizer.zero_grad()
            preds = model(x_batch)

            # Loss forward pass
            loss, l1, l2, l3, l4 = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            # Tracking
            total_loss += loss.item()
            part_loss_sum += l1
            side_loss_sum += l2
            price_loss_sum += l3
            qty_loss_sum += l4
            count_valid_batches += 1

        # Print Epoch Stats
        # Get learned weights (sigmas)
        with torch.no_grad():
            weights = torch.exp(-criterion.log_vars).cpu().numpy()

        print(
            f"Epoch {epoch+1} | "
            f"Total: {total_loss/len(loader):.4f} | "
            f"Part: {part_loss_sum/count_valid_batches:.4f} | "
            f"Side: {side_loss_sum/count_valid_batches:.4f} | "
            f"Price: {price_loss_sum/count_valid_batches:.4f} | "
            f"Qty: {qty_loss_sum/count_valid_batches:.4f} | "
            f"W: {weights}"
        )

    print("\nModel Training Complete.")
    return model, optimizer, criterion
