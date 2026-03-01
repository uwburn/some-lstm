import numpy as np

from some_lstm.generated_case_utils import base_lengths, run_generated_case
from some_lstm.lstm_pipeline import build_pipeline_config


def ar_signal(config_overrides=None):
    train_end_t = 700.0
    dt = 0.1
    noise_std = 0.25

    config = build_pipeline_config(config_overrides)
    _, n_total = base_lengths(train_end_t, dt, config["future_steps"])

    phi1, phi2 = 0.75, -0.20
    rng = np.random.default_rng(202)
    eps = rng.normal(0.0, noise_std, size=n_total)
    signal = np.zeros(n_total, dtype=float)
    signal[0] = eps[0]
    signal[1] = phi1 * signal[0] + eps[1]
    for i in range(2, n_total):
        signal[i] = phi1 * signal[i - 1] + phi2 * signal[i - 2] + eps[i]
    return run_generated_case("ar_signal", signal, dt, config)
