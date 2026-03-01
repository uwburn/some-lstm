import numpy as np

from some_lstm.utils import base_lengths
from some_lstm.lstm_pipeline import build_pipeline_config, run_experiment


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

    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])
    time = np.arange(0, n_total * dt, dt)
    clean_signal = np.sin(omega * time)
    rng = np.random.default_rng(noise_seed)
    signal = clean_signal + rng.normal(loc=0.0, scale=noise_std, size=n_total)

    return run_experiment(
        signal=signal,
        time=time,
        config=config,
        tag="sine_noise",
    )
