import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MultiTaskLSTM


def train_model(dataset, features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLSTM(input_size=len(features), hidden_size=128).to(device)

    criterion_cls = nn.BCELoss()  # For Part and Side
    criterion_reg = nn.MSELoss()  # For Price and Qty
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # --- 4. TRAINING LOOP ---
    print("Starting Multi-Task Training...")
    model.train()
    for epoch in range(10):
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
            p_part, p_side, p_price, p_qty = model(x_batch)

            loss_part = criterion_cls(p_part, y_part)
            mask = y_part == 1

            if mask.sum() > 0:
                loss_side = criterion_cls(p_side[mask], y_side[mask])
                loss_price = criterion_reg(p_price[mask], y_price[mask])
                loss_qty = criterion_reg(p_qty[mask], y_qty[mask])

                loss = loss_part + loss_side + (10 * loss_price) + loss_qty

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


def save_model(model, optimizer, features, scaler):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_size": len(features),
            "window_size": 10,
        },
        "models/multitask_lstm.pt",
    )
    joblib.dump(scaler, "models/feature_scaler.pkl")
