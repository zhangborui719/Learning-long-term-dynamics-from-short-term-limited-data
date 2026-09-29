"""Energy fields and temporal/spatial space-time autocorrelation for F=64."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_solution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1:]


def acf_1d(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased normalized ACF of a one-dimensional series using an FFT."""
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = x.size
    max_lag = min(max_lag, n - 1)
    if n < 2 or np.allclose(x, 0.0):
        return np.ones(max_lag + 1)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    cov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1]
    return cov / cov[0]


def temporal_acf(energy: np.ndarray, max_lag: int) -> np.ndarray:
    """Return ACF for each spatial component and their mean."""
    acfs = np.vstack([acf_1d(energy[:, j], max_lag) for j in range(energy.shape[1])])
    return acfs


def spatial_acfs(energy: np.ndarray) -> dict[str, np.ndarray]:
    """Compute three common periodic spatial ACF definitions.

    ``snapshot_centered`` centers energy separately at each time, then averages
    the normalized spatial correlations. ``global_centered`` uses one global
    mean and variance. ``component_anomaly`` removes each component's temporal
    mean before calculating the spatial covariance and normalizes by its mean
    variance. All definitions use periodic spatial shifts.
    """
    n_time, n_space = energy.shape
    lags = np.arange(n_space // 2 + 1)

    snapshot = np.empty(lags.size)
    for lag in lags:
        a = energy - energy.mean(axis=1, keepdims=True)
        b = np.roll(a, -lag, axis=1)
        denom = np.mean(a * a, axis=1)
        snapshot[lag] = np.mean(np.divide(np.mean(a * b, axis=1), denom,
                                          out=np.zeros_like(denom), where=denom > 0))

    centered = energy - energy.mean()
    variance = np.mean(centered * centered)
    global_centered = np.array([
        np.mean(centered * np.roll(centered, -lag, axis=1)) / variance
        for lag in lags
    ])

    anomaly = energy - energy.mean(axis=0, keepdims=True)
    component_variance = np.mean(anomaly * anomaly)
    component_anomaly = np.array([
        np.mean(anomaly * np.roll(anomaly, -lag, axis=1)) / component_variance
        for lag in lags
    ])
    return {
        "snapshot_centered": snapshot,
        "global_centered": global_centered,
        "component_anomaly": component_anomaly,
    }


def spacetime_acf(energy: np.ndarray, max_time_lag: int, max_space_lag: int) -> np.ndarray:
    """C(dt, dj), centered globally and averaged over periodic space shifts."""
    x = energy - energy.mean()
    max_time_lag = min(max_time_lag, x.shape[0] - 1)
    max_space_lag = min(max_space_lag, x.shape[1] // 2)
    variance = np.mean(x * x)
    padded_time = 1 << (2 * x.shape[0] - 1).bit_length()
    spectrum = np.fft.rfftn(x, s=(padded_time, x.shape[1]), axes=(0, 1))
    covariance = np.fft.irfftn(spectrum * np.conjugate(spectrum),
                               s=(padded_time, x.shape[1]), axes=(0, 1)).real
    # Normalize by the number of available time pairs and spatial sites.
    denominator = (x.shape[0] - np.arange(max_time_lag + 1))[:, None] * x.shape[1] * variance
    return covariance[: max_time_lag + 1, : max_space_lag + 1] / denominator


def save_energy_csv(path: Path, times: np.ndarray, energy: np.ndarray) -> None:
    header = "time," + ",".join(f"e_{j}" for j in range(energy.shape[1]))
    np.savetxt(path, np.column_stack((times, energy)), delimiter=",", header=header, comments="")


def make_plots(output: Path, times: np.ndarray, energy: np.ndarray, dt: float,
               max_time_lag: int, max_space_lag: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    energy_min, energy_max = float(energy.min()), float(energy.max())
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    image = ax.imshow(energy.T, origin="lower", aspect="auto",
                      extent=(times[0], times[-1], 0, energy.shape[1] - 1),
                      cmap="magma", vmin=energy_min, vmax=energy_max,
                      interpolation="nearest")
    ax.set(xlabel="Time", ylabel="Spatial index", title=r"Lorenz-96 energy $E_j=x_j^2/2$, $F=64$")
    ax.set_yticks([0, 10, 20, 30, energy.shape[1] - 1])
    fig.colorbar(image, ax=ax, label=r"$E_j(t)$")
    fig.savefig(output / "energy_space_time_F64.png", dpi=220)
    plt.close(fig)

    acfs = temporal_acf(energy, max_time_lag)
    lag_time = np.arange(acfs.shape[1]) * dt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(lag_time, acfs.mean(axis=0), color="#b2182b", lw=2, label="mean over 40 components")
    ax.fill_between(lag_time, acfs.min(axis=0), acfs.max(axis=0), color="#b2182b", alpha=0.18,
                    label="component range")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set(xlabel="Time lag", ylabel="Energy ACF", title=r"Temporal ACF of energy, $F=64$")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "energy_temporal_acf_F64.png", dpi=220)
    plt.close(fig)
    np.savetxt(output / "energy_temporal_acf_F64.csv", np.column_stack((lag_time, acfs.mean(axis=0))),
               delimiter=",", header="lag_time,mean_acf", comments="")

    spatial = spatial_acfs(energy)
    lags = np.arange(len(next(iter(spatial.values()))))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True, sharey=True, constrained_layout=True)
    titles = {
        "snapshot_centered": "Snapshot-centered",
        "global_centered": "Global-centered",
        "component_anomaly": "Component temporal anomaly",
    }
    for ax, (name, values) in zip(axes, spatial.items()):
        ax.plot(lags, values, "o-", ms=3, color="#2166ac")
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_title(titles[name])
        ax.set_xlabel("Spatial lag")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Energy spatial ACF")
    fig.suptitle(r"Three definitions of spatial energy ACF, $F=64$", y=1.06)
    fig.savefig(output / "energy_spatial_acf_definitions_F64.png", dpi=220)
    plt.close(fig)
    np.savetxt(output / "energy_spatial_acf_definitions_F64.csv",
               np.column_stack((lags, *spatial.values())), delimiter=",",
               header="spatial_lag," + ",".join(spatial), comments="")

    surface = spacetime_acf(energy, max_time_lag, max_space_lag)
    time_lags = np.arange(surface.shape[0]) * dt
    space_lags = np.arange(surface.shape[1])
    np.savetxt(output / "energy_spacetime_acf_F64.csv", surface, delimiter=",")
    t_grid, s_grid = np.meshgrid(time_lags, space_lags, indexing="ij")
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface_plot = ax.plot_surface(t_grid, s_grid, surface, cmap="coolwarm", linewidth=0,
                                   antialiased=True, rstride=max(1, surface.shape[0] // 180),
                                   cstride=1)
    ax.set(xlabel="Time lag", ylabel="Spatial lag", zlabel="Energy ACF",
           title=r"Space-time ACF of energy, $F=64$")
    fig.colorbar(surface_plot, ax=ax, shrink=0.62, pad=0.1, label="ACF")
    fig.savefig(output / "energy_spacetime_acf_3d_F64.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/lorenz96_rk4_F64.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/energy_correlation"))
    parser.add_argument("--burn-time", type=float, default=5.0)
    parser.add_argument("--max-time-lag", type=int, default=500)
    parser.add_argument("--max-space-lag", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    times, states = read_solution(args.input)
    burn = int(round(args.burn_time / (times[1] - times[0])))
    times, states = times[burn:], states[burn:]
    energy = 0.5 * states ** 2
    save_energy_csv(args.output / "energy_F64.csv", times, energy)
    make_plots(args.output, times, energy, times[1] - times[0], args.max_time_lag, args.max_space_lag)
    print(f"Saved energy and correlation figures to {args.output}")


if __name__ == "__main__":
    main()
