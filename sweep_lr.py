"""Learning-rate annealing sweep for the Lorenz-96 flow map.

The training loop already anneals (cosine by default), so the *starting* rate is
the only free knob left in the optimiser.  This driver sweeps it on a grid of
``(target codec, peak lr, seed)`` and aggregates the resulting ``metrics.json``
files into one table and one figure.

Two things are deliberately kept separate:

* **codec-independent accuracy** — ``metrics["test_nrmse"]`` is the one-step
  error in *physical state* units, so it is comparable between ``direct``
  (regress ``z(t+dt)``) and ``delta`` (regress ``dz``).  The per-codec training
  loss is not.
* **attractor statistics** — because ``tau_Lyap = 0.084 < dt = 0.1`` the
  trajectory itself is destroyed by a single step, so the only metrics that
  discriminate between codecs are the invariant ones: the energy spectrum, the
  variance ratio and the mean offset of the free-running rollout.

Usage::

    python sweep_lr.py --tag lr --lrs 3e-4,1e-3,2e-3,3e-3,5e-3,8e-3,1.2e-2,2e-2
    python sweep_lr.py --tag lr --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger("flowmap.sweep")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "results"
PYTHON = Path(sys.executable)

#: Metrics copied from ``metrics.json`` into the aggregate table.
METRIC_KEYS: tuple[str, ...] = (
    "test_nrmse",
    "test_nrmse_persistence",
    "test_nrmse_affine",
    "delta_nrmse",
    "delta_correlation",
    "rollout_nrmse",
    "mean_ratio",
    "rms_ratio",
    "spectrum_correlation",
    "spectrum_peak_mode",
    "reference_peak_mode",
    "train_seconds",
    "n_train_augmented",
)

COLOURS = {"direct": "#2166ac", "delta": "#b2182b"}
MARKERS = {"direct": "o", "delta": "s"}


def run_name(tag: str, mode: str, lr: float, warmup: float, epochs: int, seed: int) -> str:
    """Deterministic, filesystem-safe directory name for one run."""
    return f"{tag}_{mode}_lr{lr:.0e}_w{warmup:.2f}_e{epochs}_s{seed}"


def build_command(args: argparse.Namespace, output: Path) -> list[str]:
    """Assemble the ``train_flowmap.py`` invocation for one grid point."""
    return [
        str(PYTHON),
        str(PROJECT_ROOT / "train_flowmap.py"),
        "--data", str(PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"),
        "--mode", args.mode,
        "--standardization", "global",
        "--epochs", str(args.epochs),
        "--lr", repr(args.lr),
        "--schedule", args.schedule,
        "--lr-final-frac", repr(args.lr_final_frac),
        "--warmup-frac", repr(args.warmup),
        "--seed", str(args.seed),
        "--output", str(output),
    ]


def launch(job: dict[str, Any]) -> tuple[str, int, float]:
    """Run one training job in a subprocess.

    Args:
        job: keys ``name``, ``command``, ``output``, ``log``.

    Returns:
        ``(name, returncode, wall_seconds)``.
    """
    import time

    output = Path(job["output"])
    log = Path(job["log"])
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            job["command"], stdout=handle, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), env=environment, check=False,
        )
    return job["name"], completed.returncode, time.perf_counter() - started


def collect(root: Path) -> list[dict[str, Any]]:
    """Read every ``metrics.json`` below ``root`` into flat rows."""
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("unreadable metrics file: %s", path)
            continue
        row: dict[str, Any] = {
            "run": path.parent.name,
            "mode": metrics.get("mode"),
            "lr": metrics.get("learning_rate"),
            "schedule": metrics.get("lr_schedule", "cosine"),
            "warmup": metrics.get("lr_warmup_frac", 0.0),
            "final_frac": metrics.get("lr_final_frac", 1e-3),
            "epochs": metrics.get("epochs"),
            "seed": metrics.get("seed"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """Write the aggregate sweep as a fixed-width text table."""
    header = (
        f"{'run':<34}{'lr':>8}{'warm':>6}{'ep':>6}{'1-step':>9}"
        f"{'incr r':>8}{'spec r':>8}{'rms r':>7}{'mean r':>7}{'rollout':>8}"
    )
    lines = [title, "=" * len(header), header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: (str(r["mode"]), float(r["lr"]), r["warmup"], r["seed"])):
        lines.append(
            f"{row['run']:<34}{row['lr']:>8.1e}{row['warmup']:>6.2f}{row['epochs']:>6d}"
            f"{row['test_nrmse']:>9.5f}{row['delta_correlation']:>8.4f}"
            f"{row['spectrum_correlation']:>8.4f}{row['rms_ratio']:>7.3f}"
            f"{row['mean_ratio']:>7.3f}{row['rollout_nrmse']:>8.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the aggregate sweep as CSV (one row per run)."""
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r["mode"]), float(r["lr"]), r["warmup"], r["seed"])):
            writer.writerow(row)


def _aggregate(rows: Iterable[dict[str, Any]], mode: str, lr: float, key: str) -> tuple[float, float, float]:
    """Mean / min / max of ``key`` over seeds for one ``(mode, lr)`` cell."""
    values = [r[key] for r in rows if r["mode"] == mode and abs(float(r["lr"]) - lr) < 1e-12
              and float(r[key]) == float(r[key])]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def plot_sweep(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """Four-panel view of the sweep: accuracy, spectrum, amplitude, mean.

    Each curve is the seed mean; the shaded band is the seed min/max, which is
    the honest uncertainty given 1500 epochs of stochastic mini-batches on a
    966-pair training set.
    """
    modes = [m for m in ("direct", "delta") if any(r["mode"] == m for r in rows)]
    lrs = sorted({float(r["lr"]) for r in rows})

    panels = [
        ("test_nrmse", "One-step NRMSE (state space)", False),
        ("spectrum_correlation", "Energy-spectrum correlation\n(log $E_k$, $k\\geq1$)", False),
        ("rms_ratio", "Free-rollout rms / truth rms", False),
        ("mean_ratio", "Free-rollout mean / truth mean", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    for ax, (key, label, _) in zip(axes.ravel(), panels):
        for mode in modes:
            mean = np.array([_aggregate(rows, mode, lr, key)[0] for lr in lrs])
            low = np.array([_aggregate(rows, mode, lr, key)[1] for lr in lrs])
            high = np.array([_aggregate(rows, mode, lr, key)[2] for lr in lrs])
            ax.plot(lrs, mean, marker=MARKERS[mode], ms=5, lw=1.7,
                    color=COLOURS[mode], label=mode)
            ax.fill_between(lrs, low, high, color=COLOURS[mode], alpha=0.16, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("Peak (starting) learning rate of the anneal")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, which="both")
        ax.legend(title="target codec", fontsize=8)
    axes[0, 0].set_title("one-step accuracy", loc="left", fontsize=10)
    axes[0, 1].set_title("attractor spectrum", loc="left", fontsize=10)
    axes[1, 0].set_title("amplitude retention", loc="left", fontsize=10)
    axes[1, 1].set_title("mean offset", loc="left", fontsize=10)
    axes[1, 1].axhline(1.0, color="0.4", lw=0.9, ls=":")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def write_markdown(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """Aggregate per ``(mode, lr)`` with seed spread, as a Markdown table."""
    keys = [
        ("test_nrmse", "1-step NRMSE"),
        ("delta_correlation", "increment r"),
        ("spectrum_correlation", "spectrum r"),
        ("rms_ratio", "rms ratio"),
        ("mean_ratio", "mean ratio"),
    ]
    lines = [f"**{title}**", ""]
    header = "| mode | peak lr | n | " + " | ".join(label for _, label in keys) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(keys)))
    present = {str(row["mode"]) for row in rows}
    for mode in [m for m in ("direct", "delta") if m in present] + sorted(present - {"direct", "delta"}):
        for lr in sorted({float(row["lr"]) for row in rows}):
            subset = [row for row in rows
                      if row["mode"] == mode and abs(float(row["lr"]) - lr) < 1e-12]
            if not subset:
                continue
            cells = []
            for key, _ in keys:
                values = [float(row[key]) for row in subset]
                mean = float(np.mean(values))
                if len(values) > 1:
                    cells.append(f"{mean:.4f} [{min(values):.4f}, {max(values):.4f}]")
                else:
                    cells.append(f"{mean:.4f}")
            lines.append(f"| {mode} | {lr:.0e} | {len(subset)} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", path)


def figure_convergence(
    root: Path, rows: list[dict[str, Any]], path: Path, lr: float | None = None
) -> None:
    """Per-epoch one-step NRMSE in *state* space, averaged over seeds.

    This is the only curve that can be compared across codecs: the training loss
    lives in each codec's own target space, but ``test_state_nrmse`` is computed
    after decoding, so both curves measure the same quantity.  It answers whether
    ``delta`` is merely slower (curves converging to the same floor) or worse
    (curves plateauing at different floors).
    """
    if lr is None:
        candidates = [row for row in rows if row["mode"] == "direct"]
        if not candidates:
            return
        grid = sorted({float(row["lr"]) for row in candidates})
        means = [_aggregate(candidates, "direct", point, "test_nrmse")[0] for point in grid]
        lr = grid[int(np.nanargmin(means))]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for mode in ("direct", "delta"):
        chosen = [row for row in rows
                  if row["mode"] == mode and abs(float(row["lr"]) - float(lr)) < 1e-12]
        if not chosen:
            continue
        series = []
        for row in chosen:
            file = root / row["run"] / "flowmap_training_F64.csv"
            if not file.exists():
                continue
            data = np.genfromtxt(file, delimiter=",", names=True)
            if "test_state_nrmse" in data.dtype.names:
                series.append(data["test_state_nrmse"])
        if not series:
            continue
        length = min(len(item) for item in series)
        stacked = np.vstack([item[:length] for item in series])
        epochs = np.arange(length)
        mean = stacked.mean(axis=0)
        ax.plot(epochs, mean, lw=1.9, color=COLOURS[mode],
                label=f"{mode} (final {mean[-1]:.4f})")
        if stacked.shape[0] > 1:
            ax.fill_between(epochs, stacked.min(axis=0), stacked.max(axis=0),
                            color=COLOURS[mode], alpha=0.18, lw=0)

    ax.axhline(1.2508, color="0.45", lw=1.0, ls=":", label="persistence baseline")
    ax.axhline(0.9653, color="0.60", lw=1.0, ls="--", label="affine ridge baseline")
    ax.set(xlabel="Epoch", ylabel="One-step NRMSE on the held-out segment",
           title=f"Convergence in a codec-independent metric (peak lr = {lr:.1e})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="lr", help="results/flowmap_v3_<tag>/ subdirectory")
    parser.add_argument("--modes", default="direct,delta")
    parser.add_argument("--lrs", default="3e-4,1e-3,2e-3,3e-3,5e-3,8e-3,1.2e-2,2e-2")
    parser.add_argument("--warmup-fracs", default="0.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--lr-final-frac", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report-only", action="store_true",
                        help="skip training and only re-aggregate existing runs")
    parser.add_argument("--no-curves", action="store_true",
                        help="skip the per-epoch convergence figure")
    parser.add_argument("--title", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: launch the grid, then aggregate and plot it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)

    root = DEFAULT_ROOT / f"flowmap_v3_{args.tag}"
    root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    if not args.report_only:
        for mode in args.modes.split(","):
            for warmup in (float(w) for w in args.warmup_fracs.split(",")):
                for lr in (float(v) for v in args.lrs.split(",")):
                    for seed in (int(s) for s in args.seeds.split(",")):
                        point = argparse.Namespace(**vars(args))
                        point.mode, point.lr, point.warmup, point.seed = mode, lr, warmup, seed
                        name = run_name(args.tag, mode, lr, warmup, args.epochs, seed)
                        output = root / name
                        if (output / "metrics.json").exists():
                            LOGGER.info("skip (cached): %s", name)
                            continue
                        jobs.append({
                            "name": name,
                            "command": build_command(point, output),
                            "output": output,
                            "log": root / f"{name}.log",
                        })

    if jobs:
        LOGGER.info("launching %d runs on %d workers", len(jobs), args.workers)
        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, code, seconds in pool.map(launch, jobs):
                if code != 0:
                    failures += 1
                    LOGGER.error("FAILED %s (rc=%d, %.1fs)", name, code, seconds)
                else:
                    LOGGER.info("done %-40s %6.1fs", name, seconds)
        if failures:
            LOGGER.error("%d runs failed; aggregate covers the successful ones", failures)

    rows = collect(root)
    if not rows:
        LOGGER.error("no metrics found under %s", root)
        return 1
    write_csv(rows, root / "sweep_summary.csv")
    write_table(rows, root / "sweep_summary.txt",
                args.title or f"Learning-rate annealing sweep ({args.tag}), F=64, dt=0.1")
    write_markdown(rows, root / "sweep_summary.md",
                   args.title or f"Learning-rate annealing sweep ({args.tag})")
    plot_sweep(rows, root / "sweep_lr_curve.png",
               args.title or f"Starting learning rate vs attractor statistics ({args.tag})")
    if not args.no_curves:
        figure_convergence(root, rows, root / "sweep_convergence.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
