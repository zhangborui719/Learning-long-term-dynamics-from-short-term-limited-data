"""Test temporal decorrelation and estimate effective sample sizes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


def read_solution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1:]


def acf_fft(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased, normalized ACF computed with an FFT."""
    centered = np.asarray(series, dtype=float) - np.mean(series)
    n = centered.size
    if n < 2 or np.allclose(centered, 0.0):
        return np.ones(max_lag + 1)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1]
    return covariance / covariance[0]


def integrated_autocorrelation_time(acf: np.ndarray) -> tuple[float, int]:
    """Initial-positive-sequence estimate of tau_int and its cutoff lag."""
    if acf.size <= 1:
        return 1.0, 0
    cutoff = acf.size - 1
    for lag in range(1, acf.size):
        if acf[lag] <= 0.0:
            cutoff = lag - 1
            break
    tau = 1.0 + 2.0 * float(np.sum(acf[1 : cutoff + 1]))
    return max(tau, 1.0), cutoff


def ljung_box(acf: np.ndarray, n: int, lag: int) -> tuple[float, float]:
    """Ljung-Box Q statistic and chi-square p-value at one lag order."""
    lag = min(lag, acf.size - 1, n - 1)
    lags = np.arange(1, lag + 1, dtype=float)
    q = n * (n + 2.0) * float(np.sum(acf[1 : lag + 1] ** 2 / (n - lags)))
    return q, float(chi2.sf(q, lag))


def analyze(states: np.ndarray, dt: float, burn_time: float, max_lag: int) -> tuple[list[dict], np.ndarray]:
    burn = int(round(burn_time / dt))
    series = states[burn:]
    max_lag = min(max_lag, series.shape[0] - 1)
    acfs = np.empty((series.shape[1], max_lag + 1))
    rows: list[dict] = []
    test_lags = [10, 25, 50, 100]
    for component in range(series.shape[1]):
        acf = acf_fft(series[:, component], max_lag)
        acfs[component] = acf
        tau, cutoff = integrated_autocorrelation_time(acf)
        row = {
            "component": component,
            "n_samples": series.shape[0],
            "tau_int_steps": tau,
            "tau_int_time": tau * dt,
            "ess": series.shape[0] / tau,
            "acf_cutoff_lag": cutoff,
        }
        for lag in test_lags:
            q, p = ljung_box(acf, series.shape[0], lag)
            row[f"ljung_box_q_{lag}"] = q
            row[f"ljung_box_p_{lag}"] = p
        rows.append(row)
    return rows, acfs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_analysis(path: Path, results: dict[str, tuple[list[dict], np.ndarray]], max_lag: int, dt: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"F=8": "#2166ac", "F=64": "#b2182b"}
    for label, (rows, acfs) in results.items():
        mean_acf = np.mean(acfs, axis=0)
        axes[0, 0].plot(np.arange(max_lag + 1) * dt, mean_acf, label=label, color=colors[label])
        ess = np.array([row["ess"] for row in rows])
        axes[0, 1].plot(np.arange(1, len(ess) + 1), ess, label=label, color=colors[label])
        p_values = np.array([row["ljung_box_p_50"] for row in rows])
        axes[1, 0].plot(np.arange(1, len(p_values) + 1), p_values, ".-", label=label, color=colors[label])
        axes[1, 1].hist(ess, bins=12, alpha=0.55, label=label, color=colors[label])

    axes[0, 0].axhline(0.0, color="0.35", lw=0.8)
    axes[0, 0].set(xlabel="Lag time", ylabel="Mean ACF", title="Temporal autocorrelation")
    axes[0, 0].set_xlim(0, max_lag * dt)
    axes[0, 1].set(xlabel="Spatial index", ylabel="ESS", title="Effective samples by component")
    axes[1, 0].axhline(0.05, color="0.35", ls="--", lw=0.8, label="p = 0.05")
    axes[1, 0].set(xlabel="Spatial index", ylabel="Ljung–Box p-value", title="Decorrelated? Q(50) test")
    axes[1, 1].set(xlabel="ESS", ylabel="Components", title="ESS distribution")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/correlation"))
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-time", type=float, default=5.0)
    parser.add_argument("--max-lag", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[list[dict], np.ndarray]] = {}
    for forcing in (8, 64):
        times, states = read_solution(args.data_dir / f"lorenz96_rk4_F{forcing}.csv")
        rows, acfs = analyze(states, times[1] - times[0], args.burn_time, args.max_lag)
        write_csv(args.output_dir / f"correlation_F{forcing}.csv", rows)
        np.savetxt(args.output_dir / f"acf_F{forcing}.csv", acfs, delimiter=",")
        results[f"F={forcing}"] = (rows, acfs)

    plot_analysis(args.output_dir / "decorrelation_comparison.png", results, args.max_lag, args.dt)
    summary_path = args.output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Burn-in: t < {args.burn_time:g}\n")
        handle.write("Ljung-Box null hypothesis: no autocorrelation through lag 50.\n")
        for label, (rows, _) in results.items():
            ess = np.array([row["ess"] for row in rows])
            tau = np.array([row["tau_int_time"] for row in rows])
            p = np.array([row["ljung_box_p_50"] for row in rows])
            handle.write(f"\n{label}\n")
            handle.write(f"n_samples per component: {rows[0]['n_samples']}\n")
            handle.write(f"ESS median/min/max: {np.median(ess):.2f} / {ess.min():.2f} / {ess.max():.2f}\n")
            handle.write(f"tau_int time median/min/max: {np.median(tau):.4f} / {tau.min():.4f} / {tau.max():.4f}\n")
            handle.write(f"Ljung-Box Q(50) p > 0.05: {np.sum(p > 0.05)}/{p.size} components\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
