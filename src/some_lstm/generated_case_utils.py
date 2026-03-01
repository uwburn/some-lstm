import numpy as np
import pandas as pd

from some_lstm.lstm_pipeline import run_pipeline_from_train_df, split_and_normalize_df


def run_generated_case(case_name, signal, dt, config):
    time = np.arange(0, len(signal) * dt, dt)
    df = pd.DataFrame({"time": time, "signal_raw": signal})
    train_df, future_df = split_and_normalize_df(
        df=df,
        signal_col="signal_raw",
        time_col="time",
        future_steps=config["future_steps"],
    )
    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=config,
        experiment_name=case_name,
    )


def base_lengths(train_end_t, dt, future_steps):
    n_train = int(np.round(train_end_t / dt))
    n_total = n_train + future_steps
    return n_train, n_total
