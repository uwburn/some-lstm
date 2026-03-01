import numpy as np

from some_lstm.utils import base_lengths
from some_lstm.lstm_pipeline import build_pipeline_config, run_experiment


def sine_regime_shift(config_overrides=None):
    train_end_t = 700.0
    dt = 0.1

    config = build_pipeline_config(config_overrides)
    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])
    t = np.arange(0, n_total * dt, dt)

    split_idx = int(0.6 * n_total)
    signal = np.empty(n_total, dtype=float)
    signal[:split_idx] = 1.0 * np.sin(0.10 * t[:split_idx])
    signal[split_idx:] = 0.65 * np.sin(0.18 * t[split_idx:] + 0.7)
    return run_experiment(
        signal=signal,
        time=t,
        config=config,
        tag="sine_regime_shift",
    )
