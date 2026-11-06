"""Scan every run of a sweep and measure the *dynamics*, not just the one-step error.

The point of this scan is the pair of numbers that the training loop never
reports: the leading Lyapunov exponent of the learned map and whether its
attractor is periodic.  A one-step NRMSE of 0.80 says "the map is roughly as good
as a conditional-mean predictor"; it says nothing about whether iterating it
produces a chaotic attractor at all.  With ``tau_Lyap = 0.084 < dt = 0.1`` those
two questions come apart completely, and this script shows by how much.

Usage::

    python scan_dynamics.py --root results/flowmap_v3_lr
    python scan_dynamics.py --root results/flowmap_v3_lr --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_codecs import (
    COLOURS,
    Run,
    one_step_budget,
    true_flow_map,
    ensemble_rollout,
    invariant_statistics,
    load_run,
    recurrence,
    tangent_spectrum,
)
from flowmap_data import build_split

LOGGER = logging.getLogger("flowmap.scan")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"


def learned_map(run: Run) -> Callable[[np.ndarray], np.ndarray]:
    """The composition of codec, network and decoder, as one batchable map."""

    def map_fn(states: np.ndarray) -> np.ndarray:
        standardized = run.codec.standardize(states)
        from diagnose_codecs import predict64

        return run.codec.decode(standardized, predict64(run.model, standardized))

    return map_fn


def scan_one(
    directory: Path,
    split: Any,
    reference: np.ndarray,
    initial: np.ndarray,
    n_steps: int,
    warm_up: int = 2000,
) -> dict[str, Any] | None:
    """Measure accuracy and dynamics for a single trained run.

    The Lyapunov exponent is measured twice.  ``lambda_transient`` starts at the
    test state with no warm-up, which is what a short-horizon evaluation sees;
    ``lambda_max`` attaches the tangent frame only after ``warm_up`` iterations.
    The two differ whenever the learned map has a chaotic transient that ends on
    a limit cycle, and that difference is itself the finding.
    """
    try:
        run = load_run("run", directory, split)
    except FileNotFoundError:
        return None

    budget = one_step_budget(run)

    map_fn = learned_map(run)
    transient = tangent_spectrum(map_fn, split.test_input[0], n_steps=120)
    spectrum_result = tangent_spectrum(map_fn, split.test_input[0], n_steps=200,
                                       warm_up=warm_up)
    cycle = recurrence(map_fn, split.test_input[0], n_steps=4000 + warm_up,
                       burn_in=warm_up, max_period=3000)
    sampled = ensemble_rollout(map_fn, initial, n_steps, burn_in=n_steps // 4)
    stats = invariant_statistics(sampled, reference)

    return {
        "run": directory.name,
        "mode": run.mode,
        "lr": float(run.metrics.get("learning_rate", float("nan"))),
        "seed": int(run.metrics.get("seed", -1)),
        "epochs": int(run.metrics.get("epochs", 0)),
        "state_nrmse": float(budget["state_nrmse"]),
        "beta": float(budget["beta"]),
        "increment_r2": float(budget["increment_r2"]),
        "lambda_transient": float(transient["lambda_max"]),
        "lambda_max": float(spectrum_result["lambda_max"]),
        "n_unstable": int(spectrum_result["n_unstable"]),
        "periodic": int(cycle["periodic"]),
        "period": int(cycle["period"]),
        "rms_ratio": float(stats["rms_ratio"]),
        "mean_ratio": float(stats["mean_ratio"]),
        "spectrum_correlation": float(stats["spectrum_correlation"]),
        "acf_time": float(stats["acf_time"]),
    }


def figure_scan(rows: list[dict[str, Any]], truth: dict[str, float], path: Path) -> None:
    """Three views of the same experiment: accuracy vs dynamics vs statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    for mode in ("direct", "delta"):
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        if "lambda_transient" in subset[0]:
            for row in subset:
                axes[0].plot([row["state_nrmse"], row["state_nrmse"]],
                             [row["lambda_transient"], row["lambda_max"]],
                             color=COLOURS[mode], lw=0.8, alpha=0.35, zorder=1)
            axes[0].scatter([r["state_nrmse"] for r in subset],
                            [r["lambda_transient"] for r in subset],
                            s=34, facecolors="none", edgecolors=COLOURS[mode],
                            linewidths=1.0, alpha=0.9, zorder=2,
                            label=f"{mode}, first 120 steps")
        axes[0].scatter([r["state_nrmse"] for r in subset], [r["lambda_max"] for r in subset],
                        s=46, color=COLOURS[mode], alpha=0.85, zorder=3,
                        label=f"{mode}, after 2000-step transient",
                        edgecolors="white", linewidths=0.6)
    axes[0].axhline(truth["lambda_max"], color="0.15", lw=1.2, ls="--",
                    label=f"true map ({truth['lambda_max']:.1f})")
    axes[0].axhline(0.0, color="0.45", lw=1.0)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set(xlabel="One-step NRMSE (state space)",
                ylabel=r"Learned $\lambda_{max}$ per time unit (symlog)",
                title="Short-horizon vs asymptotic stability")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=6.2, loc="lower left", framealpha=0.92)

    for mode in ("direct", "delta"):
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        axes[1].scatter([r["beta"] for r in subset], [r["rms_ratio"] for r in subset],
                        s=46, color=COLOURS[mode], alpha=0.85, label=mode,
                        edgecolors="white", linewidths=0.6)
    axes[1].axhline(1.0, color="0.15", lw=1.2, ls="--", label="truth")
    axes[1].set(xlabel=r"Attenuation $\beta$ of the one-step increment",
                ylabel="Attractor rms ratio", title="Amplitude follows attenuation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    for mode in ("direct", "delta"):
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        axes[2].scatter([r["state_nrmse"] for r in subset],
                        [r["spectrum_correlation"] for r in subset],
                        s=46, color=COLOURS[mode], alpha=0.85, label=mode,
                        edgecolors="white", linewidths=0.6)
    axes[2].set(xlabel="One-step NRMSE (state space)", ylabel="Spectrum correlation",
                title="The spectrum metric ranks the worse codec higher")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("Learning-rate sweep, F=64, dt=0.1: one-step accuracy is not the metric that matters",
                 fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "results" / "flowmap_v3_lr"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "flowmap_v3_codec"))
    parser.add_argument("--steps", type=int, default=1200, help="ensemble length per member")
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--warm-up", type=int, default=2000,
                        help="iterations discarded before attaching the tangent frame")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / f"dynamics_scan_{root.name}.csv"

    split = build_split(DATA, stride=10, train_fraction=2.0 / 3.0, burn_in=5.0,
                        standardization="global")
    reference_path = output / "truth_reference.npz"
    if not reference_path.exists():
        from diagnose_codecs import long_truth_reference

        long_truth_reference(cache=reference_path)
    reference = np.load(reference_path)["states"]
    index = np.linspace(0, split.test_input.shape[0] - 1, num=args.ensemble).astype(int)
    initial = split.test_input[index]

    if args.report_only and cache.exists():
        with cache.open(encoding="utf-8") as handle:
            rows = [{k: (float(v) if k not in ("run", "mode") else v)
                     for k, v in row.items()} for row in csv.DictReader(handle)]
    else:
        rows = []
        directories = sorted(path.parent for path in root.glob("*/metrics.json"))
        LOGGER.info("scanning %d runs under %s", len(directories), root)
        for position, directory in enumerate(directories, start=1):
            row = scan_one(directory, split, reference, initial, args.steps, args.warm_up)
            if row is None:
                continue
            rows.append(row)
            LOGGER.info(
                "[%2d/%2d] %-28s NRMSE %.4f  beta %.3f  lambda %+6.2f (transient %+6.2f)  %s  rms %.3f",
                position, len(directories), row["run"], row["state_nrmse"], row["beta"],
                row["lambda_max"], row["lambda_transient"],
                f"cycle p={row['period']}" if row["periodic"] else "chaotic",
                row["rms_ratio"],
            )
        with cache.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    truth = {"lambda_max": float(tangent_spectrum(true_flow_map, split.test_input[0],
                                                  n_steps=500)["lambda_max"])}
    figure_scan(rows, truth, output / "fig07_accuracy_vs_dynamics.png")

    n_periodic = sum(row["periodic"] for row in rows)
    LOGGER.info("%d / %d learned maps are limit cycles", n_periodic, len(rows))
    summary = {
        "n_runs": len(rows),
        "n_periodic": n_periodic,
        "runs": rows,
    }
    (output / "dynamics_scan.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
