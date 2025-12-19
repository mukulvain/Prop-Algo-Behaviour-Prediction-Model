import joblib
import torch

from .constants import WINDOW_SIZE, HIDDEN_SIZE
from .model import MultiTaskLSTM


def save_model(model, optimizer, features, scaler, model_path, scaler_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler
