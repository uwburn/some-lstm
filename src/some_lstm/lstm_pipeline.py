from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"CUDA non available ({exc}). Fallback to CPU")
    return torch.device("cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pipeline_config(overrides=None):
    config = {
        "seq_length": 256,
        "rollout_horizon": 30,
        "future_steps": 1000,
        "batch_size": 256,
        "epochs": 120,
        "learning_rate": 0.001,
    }
    if overrides:
        config.update(overrides)
    return config


def validate_config(config, train_size):
    if config["seq_length"] <= 1:
        raise ValueError("seq_length deve essere > 1")
    if config["rollout_horizon"] <= 0:
        raise ValueError("rollout_horizon deve essere > 0")
    if config["future_steps"] <= 0:
        raise ValueError("future_steps deve essere > 0")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size deve essere > 0")
    if config["epochs"] <= 0:
        raise ValueError("epochs deve essere > 0")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate deve essere > 0")

    min_points = config["seq_length"] + config["rollout_horizon"] + 1
    if train_size < min_points:
        raise ValueError(
            f"Segnale di train troppo corto ({train_size}). Servono almeno {min_points} punti."
        )


def create_supervised_sequences(series, seq_length, horizon=1):
    values = series.to_numpy()
    X, y = [], []
    for i in range(len(values) - seq_length - horizon + 1):
        X.append(values[i : i + seq_length])
        if horizon == 1:
            y.append(values[i + seq_length])
        else:
            y.append(values[i + seq_length : i + seq_length + horizon])
    return np.array(X), np.array(y)


def build_training_loader(train_df, seq_length, rollout_horizon, batch_size):
    X_np, y_np = create_supervised_sequences(
        train_df["signal"], seq_length, rollout_horizon
    )
    X = torch.tensor(X_np, dtype=torch.float32).view(-1, seq_length, 1)
    y = torch.tensor(y_np, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def build_one_step_tensors(train_df, seq_length, device):
    X_np, y_np = create_supervised_sequences(train_df["signal"], seq_length, horizon=1)
    X = torch.tensor(X_np, dtype=torch.float32).view(-1, seq_length, 1).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)
    return X, y


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=64, num_layers=2, batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_forecaster(
    model, train_loader, rollout_horizon, device, epochs, learning_rate
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_samples = 0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )

        for xb, yb in batch_bar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            current_seq = xb
            rollout_preds = []

            for _ in range(rollout_horizon):
                pred = model(current_seq)
                rollout_preds.append(pred)
                current_seq = torch.cat(
                    (current_seq[:, 1:, :], pred.unsqueeze(-1).unsqueeze(-1)),
                    dim=1,
                )

            preds = torch.stack(rollout_preds, dim=1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_loss_sum += float(loss.item()) * bs
            n_samples += bs
            avg_loss = epoch_loss_sum / n_samples
            batch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")

        epoch_loss = epoch_loss_sum / n_samples
        history.append({"epoch": epoch, "rollout_loss": epoch_loss})
        print(f"Epoch {epoch + 1}/{epochs}, Rollout Loss: {epoch_loss:.6f}")

    return pd.DataFrame(history)


def forecast_autoregressive(model, last_sequence, n_steps, device):
    model.eval()
    current_seq = (
        torch.tensor(last_sequence, dtype=torch.float32).view(1, -1, 1).to(device)
    )
    preds = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current_seq)
            preds.append(float(pred.item()))
            current_seq = torch.cat((current_seq[:, 1:, :], pred.view(1, 1, 1)), dim=1)
    return np.array(preds)


def build_report(future_df):
    err = future_df["signal_pred"] - future_df["signal_true"]
    pred_std = float(future_df["signal_pred"].std())
    true_std = float(future_df["signal_true"].std())
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": float(
            np.corrcoef(future_df["signal_pred"], future_df["signal_true"])[0, 1]
        ),
        "pred_std": pred_std,
        "true_std": true_std,
        "std_ratio": float(pred_std / true_std) if true_std > 0 else np.nan,
    }


def save_outputs(out_dir, future_df, history_df, report, experiment_name=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_name}_"

    export_df = future_df[["time", "signal_pred", "signal_true"]].copy()
    export_df["abs_err"] = np.abs(export_df["signal_pred"] - export_df["signal_true"])
    export_df.to_csv(out_dir / f"{prefix}future_values.csv", index=False)
    history_df.to_csv(out_dir / f"{prefix}training_history.csv", index=False)

    with open(out_dir / f"{prefix}report.txt", "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v:.6f}\n")


def plot_results(
    out_dir, train_df, one_step_true, one_step_pred, future_df, experiment_name
):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_name}_"

    plt.figure(figsize=(12, 4))
    plt.plot(one_step_true, label="True")
    plt.plot(one_step_pred, label="Predicted")
    plt.title("One-step prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}one_step_prediction.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    train_x = np.arange(len(train_df))
    future_x = np.arange(len(train_df), len(train_df) + len(future_df))
    plt.plot(train_x, train_df["signal"], label="Train signal")
    plt.plot(future_x, future_df["signal_true"], label="True future", linestyle="--")
    plt.plot(future_x, future_df["signal_pred"], label="Forecast")
    plt.title("Autoregressive continuation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}autoregressive_continuation.png")
    plt.show()


def run_pipeline_from_train_df(train_df, future_df, config, experiment_name=None):
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    if experiment_name:
        print(f"Experiment: {experiment_name}")

    validate_config(config, train_size=len(train_df))

    train_loader = build_training_loader(
        train_df=train_df,
        seq_length=config["seq_length"],
        rollout_horizon=config["rollout_horizon"],
        batch_size=config["batch_size"],
    )
    one_X, one_y = build_one_step_tensors(
        train_df=train_df,
        seq_length=config["seq_length"],
        device=device,
    )

    model = LSTMForecaster().to(device)
    history_df = train_forecaster(
        model=model,
        train_loader=train_loader,
        rollout_horizon=config["rollout_horizon"],
        device=device,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
    )

    with torch.no_grad():
        one_step_pred = model(one_X).detach().cpu().numpy()
    one_step_true = one_y.detach().cpu().numpy()

    last_sequence = train_df["signal"].to_numpy()[-config["seq_length"] :]
    future_pred = forecast_autoregressive(
        model=model,
        last_sequence=last_sequence,
        n_steps=len(future_df),
        device=device,
    )
    out_future_df = future_df.copy()
    out_future_df["signal_pred"] = future_pred

    report = build_report(out_future_df)
    print("Forecast report:")
    for k, v in report.items():
        print(f"- {k}: {v:.6f}")

    save_outputs(
        out_dir=Path("outputs"),
        future_df=out_future_df,
        history_df=history_df,
        report=report,
        experiment_name=experiment_name,
    )
    plot_results(
        Path("outputs"),
        train_df,
        one_step_true,
        one_step_pred,
        out_future_df,
        experiment_name,
    )
    return report


def split_and_normalize_df(df, signal_col, time_col, future_steps):
    if signal_col not in df.columns:
        raise ValueError(f"Colonna segnale non trovata: {signal_col}")
    if time_col not in df.columns:
        raise ValueError(f"Colonna tempo non trovata: {time_col}")
    if future_steps <= 0:
        raise ValueError("future_steps deve essere > 0")
    if len(df) <= future_steps:
        raise ValueError(
            "future_steps deve essere minore del numero di righe del dataset"
        )

    ordered = (
        df[[time_col, signal_col]].copy().sort_values(time_col).reset_index(drop=True)
    )
    ordered.columns = ["time", "signal_raw"]

    train_raw = ordered.iloc[:-future_steps].copy()
    future_raw = ordered.iloc[-future_steps:].copy()

    mean = float(train_raw["signal_raw"].mean())
    std = float(train_raw["signal_raw"].std())
    if std == 0:
        raise ValueError("Deviazione standard nulla sul train set.")

    train_raw["signal"] = (train_raw["signal_raw"] - mean) / std
    future_raw["signal_true"] = (future_raw["signal_raw"] - mean) / std
    return train_raw, future_raw


def run_signal_experiment(signal, time=None, config=None, experiment_name="signal"):
    signal = np.asarray(signal, dtype=float)
    if time is None:
        time = np.arange(len(signal), dtype=float)
    else:
        time = np.asarray(time, dtype=float)
    if len(signal) != len(time):
        raise ValueError("signal e time devono avere la stessa lunghezza")

    pipeline_config = build_pipeline_config(config)
    df = pd.DataFrame({"time": time, "signal_raw": signal})
    train_df, future_df = split_and_normalize_df(
        df=df,
        signal_col="signal_raw",
        time_col="time",
        future_steps=pipeline_config["future_steps"],
    )
    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=pipeline_config,
        experiment_name=experiment_name,
    )


def run_csv_experiment(
    csv_path, signal_col, time_col="time", config=None, experiment_name="signal"
):
    df = pd.read_csv(csv_path)
    pipeline_config = build_pipeline_config(config)
    train_df, future_df = split_and_normalize_df(
        df=df,
        signal_col=signal_col,
        time_col=time_col,
        future_steps=pipeline_config["future_steps"],
    )
    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=pipeline_config,
        experiment_name=experiment_name,
    )
