"""Solve the Lorenz-96 system with the classical fourth-order RK method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def lorenz96_rhs(x: np.ndarray, forcing: float) -> np.ndarray:
    """Return dx/dt for periodic Lorenz-96 state x."""
    return (
        (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1)
        - x
        + forcing
    )


def rk4_step(x: np.ndarray, dt: float, forcing: float) -> np.ndarray:
    """Advance one fixed RK4 step."""
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_lorenz96(
    x0: np.ndarray,
    forcing: float,
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate from t=0 through t_end and return times and states."""
    steps_float = t_end / dt
    steps = round(steps_float)
    if not np.isclose(steps, steps_float):
        raise ValueError("t_end must be an integer multiple of dt")
    if dt <= 0.0 or t_end <= 0.0:
        raise ValueError("dt and t_end must be positive")

    times = np.linspace(0.0, t_end, steps + 1)
    states = np.empty((steps + 1, x0.size), dtype=float)
    states[0] = x0
    for n in range(steps):
        states[n + 1] = rk4_step(states[n], dt, forcing)
    return times, states


def initial_condition(n: int, forcing: float, seed: int) -> np.ndarray:
    """Return a reproducible perturbation of the equilibrium x_j=F."""
    rng = np.random.default_rng(seed)
    return forcing + rng.normal(loc=0.0, scale=0.01, size=n)


def save_csv(path: Path, times: np.ndarray, states: np.ndarray) -> None:
    header = "time," + ",".join(f"x_{j}" for j in range(states.shape[1]))
    np.savetxt(
        path,
        np.column_stack((times, states)),
        delimiter=",",
        header=header,
        comments="",
    )


def plot_solution(
    path: Path,
    times: np.ndarray,
    states: np.ndarray,
    forcing: float,
) -> None:
    limit = float(np.max(np.abs(states)))
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    image = ax.imshow(
        states.T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], 0, states.shape[1] - 1),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.text(
        0.02,
        0.94,
        f"F = {forcing:g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylabel("Spatial index", fontsize=15)
    ax.set_yticks([0, 10, 20, 30, states.shape[1] - 1])
    ax.tick_params(labelsize=12)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label(r"$x_j(t)$", fontsize=14)
    colorbar.ax.tick_params(labelsize=11)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forcing", type=float, default=8.0)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x0 = initial_condition(args.n, args.forcing, args.seed)
    times, states = integrate_lorenz96(x0, args.forcing, args.t_end, args.dt)

    figure_path = args.output_dir / f"lorenz96_rk4_F{args.forcing:g}.png"
    csv_path = args.output_dir / f"lorenz96_rk4_F{args.forcing:g}.csv"
    initial_path = args.output_dir / f"initial_condition_F{args.forcing:g}.csv"
    plot_solution(figure_path, times, states, args.forcing)
    save_csv(csv_path, times, states)
    np.savetxt(initial_path, x0, delimiter=",", header="x_j(0)", comments="")

    print(f"Saved figure: {figure_path}")
    print(f"Saved solution: {csv_path}")
    print(f"State range: [{states.min():.6f}, {states.max():.6f}]")


if __name__ == "__main__":
    main()
