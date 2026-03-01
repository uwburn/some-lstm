import numpy as np
import pandas as pd

from some_lstm.lstm_pipeline import build_pipeline_config, run_pipeline_from_train_df


def build_sine_train_df(train_end_t, dt, omega):
    time = np.arange(0, train_end_t, dt)
    df = pd.DataFrame({"time": time})
    df["signal_raw"] = np.sin(omega * df["time"])

    mean = float(df["signal_raw"].mean())
    std = float(df["signal_raw"].std())
    df["signal"] = (df["signal_raw"] - mean) / std
    return df, mean, std


def build_sine_future_df(last_t, future_steps, dt, omega, mean, std):
    future_t = last_t + dt * np.arange(1, future_steps + 1)
    df = pd.DataFrame({"time": future_t})
    df["signal_raw_true"] = np.sin(omega * df["time"])
    df["signal_true"] = (df["signal_raw_true"] - mean) / std
    return df


def sine(config_overrides=None):
    # Parametri del segnale seno sintetico.
    train_end_t = 700.0
    dt = 0.1
    omega = 0.1

    config = build_pipeline_config(
        {
            "seq_length": 256,
            "rollout_horizon": 30,
            "future_steps": 1000,
            "batch_size": 256,
            "epochs": 120,
            "learning_rate": 0.001,
            **(config_overrides or {}),
        }
    )

    train_df, mean, std = build_sine_train_df(
        train_end_t=train_end_t,
        dt=dt,
        omega=omega,
    )
    future_df = build_sine_future_df(
        last_t=float(train_df["time"].iloc[-1]),
        future_steps=config["future_steps"],
        dt=dt,
        omega=omega,
        mean=mean,
        std=std,
    )

    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=config,
        experiment_name="sine",
    )
