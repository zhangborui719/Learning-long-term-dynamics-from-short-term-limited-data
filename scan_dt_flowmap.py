"""Scan the sampling interval to locate the source of the flow-map error.

The main experiment fixes ``dt = 0.1`` because that is the requested grid.  This
script asks the follow-up question: is the residual error a property of the
network, or of the sampling interval?

For ``F = 64`` the largest Lyapunov exponent is about ``11.8`` per time unit, so
``tau_Lyap ~ 0.085``.  A step of ``0.1`` therefore spans **1.2 Lyapunov times**:
the true one-step map is locally expanding by a factor ``exp(0.1 * 11.8) ~ 3.3``,
and two states that are close on the attractor map to states that are far apart.
That is a property of the system, not of the model, so the one-step error should
scale with ``dt / tau_Lyap`` regardless of how well the network is fitted.

Every configuration here is trained with the *same* number of gradient steps and
the same architecture, so the sweep compares sampling scales at fixed compute.

Usage::

    python scan_dt_flowmap.py --steps 80000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flowmap_data import build_codec, build_split, encode_dataset
from flowmap_model import FlowMapMLP, predict, train_flow_map
from train_flowmap import largest_lyapunov, shift_augment

LOGGER = logging.getLogger("flowmap.scan")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "flowmap"


def run_one(
    stride: int, args: argparse.Namespace, lyapunov_time: float
) -> dict[str, float]:
    """Train one network at a given stride and report the single-step accuracy.

    Args:
        stride: resampling stride applied to the raw ``dt = 0.01`` record.
        args: parsed command-line arguments.
        lyapunov_time: ``1 / lambda_max``, used only for labelling.

    Returns:
        A row of metrics.
    """
    split = build_split(
        Path(args.data), stride=stride, train_fraction=args.train_fraction,
        burn_in=args.burn_in,
    )
    codec = build_codec("direct", split)
    data = encode_dataset(split, codec)
    train_x, train_y = shift_augment(data["train_x"], data["train_y"], args.shift_augment)

    # hold the number of gradient steps fixed instead of the number of epochs
    batches_per_epoch = max(1, int(np.ceil(train_x.shape[0] / args.batch_size)))
    epochs = max(20, int(round(args.steps / batches_per_epoch)))

    torch.manual_seed(args.seed)
    model = FlowMapMLP(split.n_dim, args.hidden_width, args.hidden_layers, args.activation)
    history = train_flow_map(
        model, train_x, train_y, data["test_x"], data["test_y"],
        epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr, log_every=0,
    )

    prediction = codec.decode(data["test_x"], predict(model, data["test_x"]))
    truth = split.test_target
    initial = split.test_input
    state_rms = float(np.sqrt(np.mean(truth**2)))
    increment_rms = float(np.sqrt(np.mean((truth - initial) ** 2)))

    row = {
        "stride": float(stride),
        "dt": split.dt,
        "dt_over_lyapunov_time": split.dt / lyapunov_time,
        "n_train_pairs": float(split.train_input.shape[0]),
        "n_train_augmented": float(train_x.shape[0]),
        "epochs": float(epochs),
        "gradient_steps": float(epochs * batches_per_epoch),
        "nrmse": float(np.sqrt(np.mean((prediction - truth) ** 2)) / state_rms),
        "increment_nrmse": float(
            np.sqrt(np.mean(((prediction - initial) - (truth - initial)) ** 2)) / increment_rms
        ),
        "increment_correlation": float(
            np.corrcoef((prediction - initial).ravel(), (truth - initial).ravel())[0, 1]
        ),
        "train_mse": float(history.train_loss[-1]),
        "test_mse": float(history.test_loss[-1]),
    }
    LOGGER.info(
        "stride %2d  dt %.3f  dt/tau %.2f  nrmse %.4f  inc_corr %.4f  (%d steps)",
        stride,
        split.dt,
        row["dt_over_lyapunov_time"],
        row["nrmse"],
        row["increment_correlation"],
        int(row["gradient_steps"]),
    )
    return row


def plot_scan(path: Path, rows: list[dict[str, float]], forcing: int) -> None:
    """Plot the one-step error against the sampling interval in Lyapunov units."""
    ratio = np.array([row["dt_over_lyapunov_time"] for row in rows])
    nrmse = np.array([row["nrmse"] for row in rows])
    increment = np.array([row["increment_nrmse"] for row in rows])
    correlation = np.array([row["increment_correlation"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    axes[0].semilogx(ratio, nrmse, "o-", lw=1.8, ms=6, color="#2166ac",
                     label="next-state NRMSE")
    axes[0].semilogx(ratio, increment, "s--", lw=1.6, ms=5, color="#b2182b",
                     label="increment NRMSE")
    axes[0].axvline(1.0, color="0.5", lw=1.0, ls=":")
    axes[0].annotate("one Lyapunov time", xy=(1.0, axes[0].get_ylim()[0]),
                     xytext=(4, 10), textcoords="offset points", fontsize=8, color="0.35")
    axes[0].set(xlabel=r"Sampling interval $dt / \tau_{Lyap}$",
                ylabel="Normalised error",
                title="Error grows with the sampling interval")
    axes[0].legend()
    axes[0].grid(alpha=0.25, which="both")

    axes[1].semilogx(ratio, correlation, "o-", lw=1.8, ms=6, color="#1b7837")
    axes[1].axhline(0.0, color="0.4", lw=0.9)
    axes[1].axvline(1.0, color="0.5", lw=1.0, ls=":")
    axes[1].set(xlabel=r"Sampling interval $dt / \tau_{Lyap}$",
                ylabel="corr(predicted increment, true increment)",
                title="Directional skill decays with the sampling interval")
    axes[1].grid(alpha=0.25, which="both")

    fig.suptitle(rf"Flow-map learnability versus sampling interval, $F={forcing:g}$", y=1.04)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--steps", type=int, default=80000,
                        help="gradient steps per configuration (fixed compute budget)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=0, help="unused, kept for symmetry")
    parser.add_argument("--burn-in", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--hidden-width", type=int, default=80)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--shift-augment", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forcing", type=float, default=64.0)
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    reference = build_split(Path(args.data), stride=1, burn_in=args.burn_in)
    estimate = largest_lyapunov(
        reference.test_input[0], forcing=args.forcing, dt=0.01, t_total=40.0
    )
    LOGGER.info(
        "lambda_max = %.3f per time unit, tau_Lyap = %.4f",
        estimate["lambda_max"],
        estimate["lyapunov_time"],
    )

    rows = [run_one(stride, args, estimate["lyapunov_time"]) for stride in args.strides]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    np.savetxt(
        output / "flowmap_dt_scan_F64.csv",
        np.array([[row[key] for key in header] for row in rows]),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.6e",
    )
    plot_scan(output / "flowmap_dt_scan_F64.png", rows, int(args.forcing))

    print("\nSampling-interval scan")
    print("-" * 78)
    print(f"{'dt':>6} {'dt/tau':>7} {'n_train':>8} {'nrmse':>7} {'inc_nrmse':>10} {'inc_corr':>9}")
    for row in rows:
        print(
            f"{row['dt']:6.3f} {row['dt_over_lyapunov_time']:7.2f} "
            f"{int(row['n_train_augmented']):8d} {row['nrmse']:7.4f} "
            f"{row['increment_nrmse']:10.4f} {row['increment_correlation']:9.4f}"
        )
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
