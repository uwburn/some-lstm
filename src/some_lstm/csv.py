import pandas as pd

from some_lstm.lstm_pipeline import run_experiment


def csv_case(
    csv_path,
    signal_col,
    time_col="time",
    tag="csv",
    config_overrides=None,
):
    df = pd.read_csv(csv_path)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    if time_col not in df.columns:
        raise ValueError(f"Time column not found: {time_col}")

    return run_experiment(
        signal=df[signal_col].to_numpy(),
        time=df[time_col].to_numpy(),
        config=config_overrides,
        tag=tag,
    )
