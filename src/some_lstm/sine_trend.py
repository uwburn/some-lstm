import numpy as np

from some_lstm.generated_case_utils import base_lengths, run_generated_case
from some_lstm.lstm_pipeline import build_pipeline_config


def sine_trend(config_overrides=None):
    train_end_t = 700.0
    dt = 0.1
    omega = 0.1
    trend_slope = 0.001

    config = build_pipeline_config(config_overrides)
    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])
    t = np.arange(0, n_total * dt, dt)
    signal = np.sin(omega * t) + trend_slope * t
    return run_generated_case("sine_trend", signal, dt, config)
