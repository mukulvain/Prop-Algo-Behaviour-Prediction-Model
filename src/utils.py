import joblib
import torch
import torch.optim as optim

from .constants import HIDDEN_SIZE, WINDOW_SIZE
from .losses import MultiTaskLoss
from .model import MultiTaskLSTM


def save_model(model, optimizer, criterion, features, scaler, model_path, scaler_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "input_size": len(features),
            "window_size": WINDOW_SIZE,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)


def load_model(model_path, scaler_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = MultiTaskLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    criterion = MultiTaskLoss(num_tasks=4).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.0005
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()

    return model, optimizer, criterion, scaler
