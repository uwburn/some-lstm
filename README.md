# some-lstm

Small practical examples of LSTM-based forecasting on generated time-series signals.

This project is intended as a lightweight playground to train an LSTM forecaster and evaluate how well it predicts the future part of different signals (synthetic and CSV-based).

## What This Project Does

- Builds and trains an LSTM model for univariate sequence forecasting.
- Uses autoregressive multi-step continuation for future prediction.
- Runs multiple signal scenarios (sine, noisy sine, trend, regime shift, AR process, piecewise constant, etc.).
- Exports metrics, predictions, and training history.

## Project Structure

- `src/some_lstm/lstm_pipeline.py`: core training and forecasting pipeline.
- `src/some_lstm/__main__.py`: CLI entrypoint and experiment commands.
- `src/some_lstm/*.py`: signal generators and experiment wrappers.
- `outputs/`: generated charts, reports, and CSV files.

## Requirements

- Python `>=3.14,<3.15` (as declared in `pyproject.toml`)
- Poetry

## Installation

```bash
poetry install
```

## Usage

The project exposes a CLI script named `start`.

```bash
poetry run start --help
```

### Run Synthetic Experiments

```bash
poetry run start sine
poetry run start sine_noise --noise-std 0.20 --noise-seed 123
poetry run start sine_trend
poetry run start sine_multifreq
poetry run start sine_regime_shift
poetry run start ar_signal
poetry run start piecewise_constant
```

### Run on a CSV Dataset

```bash
poetry run start csv \
  --csv path/to/data.csv \
  --signal-col signal_column_name \
  --time-col time \
  --tag my_experiment
```

## Common Training Parameters

Most commands support:

- `--seq-length` (default: `256`)
- `--rollout-horizon` (default: `30`)
- `--future-steps` (default: `1000`, `300` for CSV)
- `--batch-size` (default: `256`)
- `--epochs` (default: `120`)
- `--learning-rate` (default: `0.001`)

Example:

```bash
poetry run start sine_multifreq --epochs 200 --seq-length 128 --future-steps 800
```

## Outputs

Each run writes files in `outputs/`:

- `<tag>_future_values.csv`
- `<tag>_training_history.csv`
- `<tag>_report.txt`
- `<tag>_one_step_prediction.png`
- `<tag>_autoregressive_continuation.png`

The report includes common forecast metrics such as MAE, RMSE, correlation, and standard deviation ratio.

## Notes

- The pipeline normalizes the signal using training-set statistics.
- It automatically uses CUDA when available, otherwise CPU.
