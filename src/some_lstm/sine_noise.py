import numpy as np
import pandas as pd

from some_lstm.lstm_pipeline import build_pipeline_config, run_pipeline_from_train_df


def build_noisy_sine_train_future_df(
    train_end_t, future_steps, dt, omega, noise_std, noise_seed
):
    n_train = int(np.round(train_end_t / dt))
    n_total = n_train + future_steps
    time = np.arange(0, n_total * dt, dt)

    clean_signal = np.sin(omega * time)
    rng = np.random.default_rng(noise_seed)
    noisy_signal = clean_signal + rng.normal(loc=0.0, scale=noise_std, size=n_total)

    train_raw = noisy_signal[:n_train]
    future_raw = noisy_signal[n_train:]

    train_df = pd.DataFrame({"time": time[:n_train], "signal_raw": train_raw})
    mean = float(train_df["signal_raw"].mean())
    std = float(train_df["signal_raw"].std())
    train_df["signal"] = (train_df["signal_raw"] - mean) / std

    future_df = pd.DataFrame({"time": time[n_train:], "signal_raw_true": future_raw})
    future_df["signal_true"] = (future_df["signal_raw_true"] - mean) / std
    return train_df, future_df


def sine_noise(config_overrides=None, noise_std=0.20, noise_seed=123):
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

    train_df, future_df = build_noisy_sine_train_future_df(
        train_end_t=train_end_t,
        future_steps=config["future_steps"],
        dt=dt,
        omega=omega,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=config,
        experiment_name="sine_noise",
    )
