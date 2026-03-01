import argparse
from some_lstm.ar_signal import ar_signal
from some_lstm.lstm_pipeline import run_csv_experiment
from some_lstm.piecewise_constant import piecewise_constant
from some_lstm.sine_multifreq import sine_multifreq
from some_lstm.sine_regime_shift import sine_regime_shift
from some_lstm.sine import sine
from some_lstm.sine_noise import sine_noise
from some_lstm.sine_trend import sine_trend


def add_training_args(p, default_future_steps=1000):
    p.add_argument("--seq-length", type=int, default=256)
    p.add_argument("--rollout-horizon", type=int, default=30)
    p.add_argument("--future-steps", type=int, default=default_future_steps)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=0.001)


def training_config_from_args(args):
    return {
        "seq_length": args.seq_length,
        "rollout_horizon": args.rollout_horizon,
        "future_steps": args.future_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="LSTM signal experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sine")
    p_sine_noise = subparsers.add_parser("sine_noise")
    p_sine_noise.add_argument("--noise-std", type=float, default=0.20)
    p_sine_noise.add_argument("--noise-seed", type=int, default=123)
    add_training_args(p_sine_noise, default_future_steps=1000)

    p_sine_trend = subparsers.add_parser("sine_trend")
    add_training_args(p_sine_trend, default_future_steps=1000)

    p_sine_multifreq = subparsers.add_parser("sine_multifreq")
    add_training_args(p_sine_multifreq, default_future_steps=1000)

    p_sine_regime_shift = subparsers.add_parser("sine_regime_shift")
    add_training_args(p_sine_regime_shift, default_future_steps=1000)

    p_ar_signal = subparsers.add_parser("ar_signal")
    add_training_args(p_ar_signal, default_future_steps=1000)

    p_piecewise_constant = subparsers.add_parser("piecewise_constant")
    add_training_args(p_piecewise_constant, default_future_steps=1000)

    p_csv = subparsers.add_parser("csv")
    p_csv.add_argument("--csv", required=True, help="Path CSV input")
    p_csv.add_argument("--signal-col", required=True, help="Signal column name")
    p_csv.add_argument("--time-col", default="time", help="Time column name")
    p_csv.add_argument("--tag", default="csv", help="Output prefix")
    add_training_args(p_csv, default_future_steps=300)

    args = parser.parse_args()

    if args.command == "sine":
        sine()
    elif args.command == "sine_noise":
        sine_noise(
            noise_std=args.noise_std,
            noise_seed=args.noise_seed,
            config_overrides=training_config_from_args(args),
        )
    elif args.command == "sine_trend":
        sine_trend(config_overrides=training_config_from_args(args))
    elif args.command == "sine_multifreq":
        sine_multifreq(config_overrides=training_config_from_args(args))
    elif args.command == "sine_regime_shift":
        sine_regime_shift(config_overrides=training_config_from_args(args))
    elif args.command == "ar_signal":
        ar_signal(config_overrides=training_config_from_args(args))
    elif args.command == "piecewise_constant":
        piecewise_constant(config_overrides=training_config_from_args(args))
    elif args.command == "csv":
        run_csv_experiment(
            csv_path=args.csv,
            signal_col=args.signal_col,
            time_col=args.time_col,
            experiment_name=args.tag,
            config=training_config_from_args(args),
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
