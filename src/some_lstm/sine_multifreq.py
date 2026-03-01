import numpy as np

from some_lstm.utils import base_lengths
from some_lstm.lstm_pipeline import build_pipeline_config, run_experiment


def sine_multifreq(config_overrides=None):
    train_end_t = 700.0
    dt = 0.1

    config = build_pipeline_config(config_overrides)
    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])
    t = np.arange(0, n_total * dt, dt)
    signal = (
        1.00 * np.sin(0.10 * t)
        + 0.55 * np.sin(0.23 * t + 0.40)
        + 0.30 * np.sin(0.05 * t - 0.20)
    )
    return run_experiment(
        signal=signal,
        time=t,
        config=config,
        tag="sine_multifreq",
    )
