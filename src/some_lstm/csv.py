import pandas as pd

from some_lstm.lstm_pipeline import (
    build_pipeline_config,
    run_pipeline_from_train_df,
    split_and_normalize_df,
)


def csv_case(
    csv_path,
    signal_col,
    time_col="time",
    tag="csv",
    config_overrides=None,
):
    df = pd.read_csv(csv_path)
    config = build_pipeline_config(config_overrides)
    train_df, future_df = split_and_normalize_df(
        df=df,
        signal_col=signal_col,
        time_col=time_col,
        future_steps=config["future_steps"],
    )
    return run_pipeline_from_train_df(
        train_df=train_df,
        future_df=future_df[["time", "signal_true"]],
        config=config,
        experiment_name=tag,
    )
