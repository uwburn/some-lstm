import numpy as np

from some_lstm.generated_case_utils import base_lengths, run_generated_case
from some_lstm.lstm_pipeline import build_pipeline_config


def piecewise_constant(config_overrides=None):
    train_end_t = 700.0
    dt = 0.1
    step_len = 350
    noise_std = 0.06

    config = build_pipeline_config(config_overrides)
    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])

    rng = np.random.default_rng(77)
    n_steps = int(np.ceil(n_total / step_len))
    levels = rng.uniform(low=-1.2, high=1.2, size=n_steps)
    base = np.repeat(levels, step_len)[:n_total]
    noise = rng.normal(0.0, noise_std, size=n_total)
    signal = base + noise
    return run_generated_case("piecewise_constant", signal, dt, config)
