import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import MultiTaskLSTM
from .constants import HIDDEN_SIZE


def train_model(dataset, features, model=None, optimizer=None, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = MultiTaskLSTM(input_size=len(features), hidden_size=HIDDEN_SIZE).to(
            device
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion_cls = nn.BCEWithLogitsLoss()  # For Part and Side
    criterion_reg = nn.MSELoss()  # For Price and Qty

    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # --- TRAINING LOOP ---
    print("Starting Multi-Task Training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        part_loss_sum, side_loss_sum, price_loss_sum, qty_loss_sum = 0, 0, 0, 0
        count_mask = 0

        for x_batch, y_part, y_side, y_price, y_qty in loader:
            x_batch = x_batch.to(device)
            y_part, y_side, y_price, y_qty = (
                y_part.to(device),
                y_side.to(device),
                y_price.to(device),
                y_qty.to(device),
            )

            optimizer.zero_grad()
            l_part, l_side, l_price, l_qty = model(x_batch)

            loss_part = criterion_cls(l_part.squeeze(), y_part)
            mask = y_part == 1

            if mask.sum() > 0:
                loss_side = criterion_cls(l_side.squeeze()[mask], y_side[mask])
                loss_price = criterion_reg(l_price.squeeze()[mask], y_price[mask])
                loss_qty = criterion_reg(l_qty.squeeze()[mask], y_qty[mask])

                loss = 2.0 * loss_part + 1.5 * loss_side + 10 * loss_price + loss_qty

                side_loss_sum += loss_side.item()
                price_loss_sum += loss_price.item()
                qty_loss_sum += loss_qty.item()
                count_mask += 1
            else:
                loss = loss_part

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            part_loss_sum += loss_part.item()

        print(
            f"Epoch {epoch+1} | "
            f"Total: {total_loss/len(loader):.4f} | "
            f"Part: {part_loss_sum/len(loader):.4f} | "
            f"Side: {side_loss_sum/max(1,count_mask):.4f} | "
            f"Price: {price_loss_sum/max(1,count_mask):.4f} | "
            f"Qty: {qty_loss_sum/max(1,count_mask):.4f}"
        )

    print("\nModel Training Complete.")
    return model, optimizer
