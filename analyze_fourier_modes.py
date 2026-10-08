"""Discrete Fourier decomposition of the Lorenz-96 state: mode-wise ACF and energy budget.

The state ``x(t) in R^N`` is expanded in the discrete Fourier basis of the
periodic spatial index,

.. math::

    \\hat{x}_k(t) = \\frac{1}{\\sqrt{N}} \\sum_{j=1}^{N} x_j(t)
        \\exp\\!\\left(-2\\pi i \\frac{(j-1)k}{N}\\right),
    \\qquad k = 0, \\dots, N-1 .

Because ``x_j`` is real, the coefficients obey the conjugate symmetry
``\\hat{x}_{N-k} = \\overline{\\hat{x}_k}``, so every piece of information is
already contained in the independent modes ``k = 0, 1, \\dots, N/2``.  Each mode
therefore carries a *complex-valued time series* ``\\{\\hat{x}_k(t)\\}``, which is
what the temporal ACF below is applied to.

Energy convention
-----------------
For a single mode the energy is the norm of its amplitude.  With the unitary
normalisation used above, Parseval reads ``\\sum_k |\\hat{x}_k|^2 = \\|x\\|^2``,
so the total energy ``\\|x\\|^2 / 2`` splits exactly as

.. math::

    \\frac{1}{2}\\|x\\|^2 = \\frac{1}{2}\\Big[
        |\\hat{x}_0|^2 + |\\hat{x}_{N/2}|^2
        + 2 \\sum_{k=1}^{N/2-1} |\\hat{x}_k|^2 \\Big] .

Mode ``k`` of a real signal owns ``c_k`` real degrees of freedom with
``c_k = 1`` for the self-conjugate modes ``k = 0`` and ``k = N/2`` and
``c_k = 2`` for ``1 <= k <= N/2 - 1`` (a conjugate pair ``+k``/``-k``).  Hence the
energy per degree of freedom is ``|\\hat{x}_k|^2 / 2`` and the total energy owned
by mode ``k`` is ``c_k |\\hat{x}_k|^2 / 2``.  The reported *share* is the
time-averaged total energy of the mode divided by the time-averaged total energy
of the state, which sums to one by construction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_solution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``time,x_0,...,x_{N-1}`` CSV produced by ``solve_lorenz96.py``."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1:]


def save_table(path: Path, header: list[str], columns: list[np.ndarray],
               fmt: str = "%.6e") -> None:
    """Write a numeric table with an explicit header line.

    Six significant digits keep the committed tables small while remaining well
    inside the accuracy of the integrator and the estimators.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header),
               comments="", fmt=fmt)


# --------------------------------------------------------------------------- #
# Fourier transform of the state
# --------------------------------------------------------------------------- #
def spatial_dft(states: np.ndarray) -> np.ndarray:
    """Unitary DFT along the spatial axis: ``(n_time, N) -> (n_time, N)`` complex."""
    n_space = states.shape[1]
    return np.fft.fft(states, axis=1) / np.sqrt(n_space)


def independent_mode_indices(n_space: int) -> np.ndarray:
    """Mode indices that are not implied by conjugate symmetry: ``0 .. N/2``."""
    return np.arange(n_space // 2 + 1)


def degrees_of_freedom(n_space: int, modes: np.ndarray) -> np.ndarray:
    """Real degrees of freedom owned by each independent mode.

    ``c_k = 1`` for the self-conjugate modes ``k = 0`` and ``k = N/2`` (these are
    real), and ``c_k = 2`` for every interior mode (the ``+k``/``-k`` pair).
    """
    weights = np.full(modes.shape, 2, dtype=int)
    weights[modes == 0] = 1
    if n_space % 2 == 0:
        weights[modes == n_space // 2] = 1
    return weights


def mode_energies(modes: np.ndarray, dof: np.ndarray) -> np.ndarray:
    """Total energy owned by each mode as a function of time, shape ``(n_time, n_modes)``."""
    return 0.5 * dof[np.newaxis, :] * np.abs(modes) ** 2


# --------------------------------------------------------------------------- #
# Complex autocorrelation
# --------------------------------------------------------------------------- #
def complex_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased, normalised autocorrelation of a complex series.

    Uses the Hermitian convention ``rho(tau) = E[conj(z_t) z_{t+tau}] / E[|z|^2]``,
    so ``rho(tau)`` is complex with ``rho(-tau) = conj(rho(tau))`` and
    ``rho(0) = 1``.  For a real input this reduces to the usual ACF.  The mean is
    removed first, as in the real-valued analysis.
    """
    z = np.asarray(series, dtype=complex)
    z = z - z.mean()
    n = z.size
    max_lag = min(max_lag, n - 1)
    if n < 2 or np.allclose(z, 0.0):
        return np.ones(max_lag + 1, dtype=complex)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.fft(z, size)
    covariance = np.fft.ifft(np.abs(spectrum) ** 2)[: max_lag + 1]
    return covariance / covariance[0]


def geyer_initial_positive_sequence(acf: np.ndarray) -> tuple[float, int]:
    """Estimate ``tau_int`` from a real ACF with Geyer's initial-positive sequence.

    ``Gamma_k = rho_(2k) + rho_(2k+1)``; the sum stops before the first
    non-positive ``Gamma_k``.  The returned cutoff is the last included lag.
    """
    if acf.size <= 1:
        return 1.0, 0
    pair_count = (acf.size - 1) // 2
    if pair_count == 0:
        return 1.0, 0
    pair_sums = acf[0 : 2 * pair_count : 2] + acf[1 : 2 * pair_count : 2]
    nonpositive = np.flatnonzero(pair_sums <= 0.0)
    included = int(nonpositive[0]) if nonpositive.size else pair_count
    if included == 0:
        return 1.0, 0
    tau = -1.0 + 2.0 * float(np.sum(pair_sums[:included]))
    return max(tau, 1.0e-12), 2 * included - 1


def ljung_box(acf: np.ndarray, n: int, lag: int) -> tuple[float, float]:
    """Ljung-Box Q statistic and chi-square p-value computed from a real ACF."""
    lag = min(lag, acf.size - 1, n - 1)
    lags = np.arange(1, lag + 1, dtype=float)
    q = n * (n + 2.0) * float(np.sum(acf[1 : lag + 1] ** 2 / (n - lags)))
    return q, float(chi2.sf(q, lag))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
ACF_LAGS = (1, 5, 10, 25, 50, 100)


def analyse_modes(modes: np.ndarray, dof: np.ndarray, dt: float, max_lag: int) -> tuple[list[dict], np.ndarray]:
    """Per-mode temporal ACF statistics plus the full real/imag curve stack."""
    n_time, n_modes = modes.shape
    max_lag = min(max_lag, n_time - 1)
    curves = np.empty((n_modes, max_lag + 1), dtype=complex)
    rows: list[dict] = []
    for index in range(n_modes):
        rho = complex_acf(modes[:, index], max_lag)
        curves[index] = rho
        # Re rho is itself a valid real ACF: it equals the variance-weighted
        # average of the ACFs of Re z and Im z.  Geyer's IPS is applied to it.
        real_part = rho.real
        tau, cutoff = geyer_initial_positive_sequence(real_part)
        mean_abs2 = float(np.mean(np.abs(modes[:, index]) ** 2))
        row = {
            "mode_k": index,
            "dof": int(dof[index]),
            "mean_abs2": mean_abs2,
            "energy_per_dof": 0.5 * mean_abs2,
            "mean_energy": 0.5 * float(dof[index]) * mean_abs2,
            "n_samples": n_time,
            "acf_abs_min": float(np.min(np.abs(real_part[1:]))),
            "tau_int_steps": tau,
            "tau_int_time": tau * dt,
            "ess": n_time / tau,
            "geyer_ips_cutoff_lag": cutoff,
            "acf_real_peak_abs": float(np.max(np.abs(real_part[1:]))),
            "acf_imag_peak_abs": float(np.max(np.abs(rho.imag[1:]))),
        }
        for lag in ACF_LAGS:
            if lag <= max_lag:
                row[f"acf_real_lag_{lag}"] = float(real_part[lag])
                row[f"acf_abs_lag_{lag}"] = float(abs(rho[lag]))
        q, p = ljung_box(real_part, n_time, 50)
        row["ljung_box_q_50"] = q
        row["ljung_box_p_50"] = p
        rows.append(row)

    total = sum(row["mean_energy"] for row in rows)
    for row in rows:
        row["energy_share"] = row["mean_energy"] / total
    return rows, curves


def verify_identities(states: np.ndarray, modes_all: np.ndarray, keep: np.ndarray,
                      energy: np.ndarray) -> dict[str, float]:
    """Numerically confirm Parseval, conjugate symmetry and the energy partition.

    ``modes_all`` must hold all ``N`` coefficients, while ``energy`` holds the
    independent modes only (``k = 0, ..., N/2``) as a ``(n_time, n_modes)`` array.
    The partition is checked against the *per-snapshot* state energy
    ``||x(t)||^2 / 2``, which is what the mode energies sum to.
    """
    n_space = states.shape[1]
    per_snapshot_energy = 0.5 * np.sum(states**2, axis=1)
    total_state = float(np.sum(per_snapshot_energy))
    total_modes = float(np.sum(0.5 * np.abs(modes_all) ** 2))
    reflected = modes_all[:, (n_space - np.arange(n_space)) % n_space]
    symmetry_error = float(np.max(np.abs(reflected - np.conjugate(modes_all))))
    error = float(np.max(np.abs(np.sum(energy, axis=1) - per_snapshot_energy)))
    return {
        "max_abs_parseval_error": abs(total_modes - total_state),
        "max_abs_conjugate_symmetry_error": symmetry_error,
        "max_abs_energy_partition_error": error,
        "mean_total_energy": float(np.mean(per_snapshot_energy)),
        "total_energy_half_norm": total_state,
        "independent_modes": float(keep.size),
    }


def brute_force_complex_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Direct O(n * lag) evaluation of :func:`complex_acf`, used as a self-check.

    Uses the same biased normalisation as :func:`complex_acf`, i.e. every lag is
    divided by the full length ``n`` rather than by ``n - lag``.
    """
    z = np.asarray(series, dtype=complex)
    z = z - z.mean()
    n = z.size
    out = np.empty(max_lag + 1, dtype=complex)
    for lag in range(max_lag + 1):
        out[lag] = np.sum(np.conjugate(z[: n - lag]) * z[lag:]) / n
    return out / out[0]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_energy_spectrum(path: Path, rows: list[dict], forcing: int) -> None:
    modes = np.array([row["mode_k"] for row in rows])
    shares = np.array([row["energy_share"] for row in rows])
    energy = np.array([row["mean_energy"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    axes[0].bar(modes, 100.0 * shares, color="#2166ac")
    axes[0].set(xlabel="Mode $k$", ylabel="Energy share [%]",
                title="Time-averaged energy share")
    axes[0].set_xticks(modes)
    axes[0].grid(alpha=0.2, axis="y")

    finite = energy[energy > 0]
    if finite.size:
        axes[1].semilogy(modes, np.maximum(energy, finite.min() * 1e-3), "o-", color="#b2182b")
    axes[1].set(xlabel="Mode $k$", ylabel=r"$\langle E_k \rangle$  (log scale)",
                title="Time-averaged mode energy")
    axes[1].set_xticks(modes)
    axes[1].grid(alpha=0.2, which="both")

    axes[2].plot(modes, 100.0 * np.cumsum(shares), "o-", color="#1b7837")
    axes[2].axhline(90.0, color="0.35", ls="--", lw=0.8)
    axes[2].set(xlabel="Mode $k$", ylabel="Cumulative share [%]",
                title="Cumulative energy share", ylim=(0, 105))
    axes[2].set_xticks(modes)
    axes[2].grid(alpha=0.2)

    fig.suptitle(rf"Fourier energy budget, Lorenz-96, $F={forcing}$", y=1.06)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_wavenumber_time(path: Path, times: np.ndarray, energy: np.ndarray, forcing: int) -> None:
    """Wavenumber-time energy density in decibels, clipped to the bulk of the data."""
    reference = float(np.mean(energy))
    with np.errstate(divide="ignore"):
        power_db = 10.0 * np.log10(np.maximum(energy, reference * 1e-8) / reference)
    low, high = np.percentile(power_db, [2.0, 98.0])
    low, high = min(low, -6.0), max(high, 6.0)
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    image = ax.imshow(power_db.T, origin="lower", aspect="auto",
                      extent=(times[0], times[-1], 0, energy.shape[1] - 1),
                      cmap="inferno", vmin=low, vmax=high, interpolation="nearest")
    ax.set(xlabel="Time", ylabel="Mode $k$",
           title=rf"Mode energy density $10\log_{{10}}(E_k/\bar{{E}})$, $F={forcing}$")
    ax.set_yticks(list(range(energy.shape[1])))
    fig.colorbar(image, ax=ax, label=f"dB relative to the mean (clipped to [{low:.0f}, {high:.0f}])")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_acf_curves(path: Path, rows: list[dict], curves: np.ndarray, dt: float,
                    forcing: int, window: float) -> None:
    """Magnitude curves, a zoomed heat map and the integrated decorrelation time."""
    lag_time = np.arange(curves.shape[1]) * dt
    inside = lag_time <= window
    modes = np.array([row["mode_k"] for row in rows])
    tau = np.array([row["tau_int_time"] for row in rows])
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    for index, mode in enumerate(modes):
        axes[0].semilogy(lag_time[inside], np.maximum(np.abs(curves[index][inside]), 1e-6),
                         lw=1.1, color=cmap(index / max(1, len(modes) - 1)))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=modes.min(), vmax=modes.max()))
    fig.colorbar(sm, ax=axes[0], label="Mode $k$")
    axes[0].axhline(0.0, color="0.35", lw=0.8)
    axes[0].set(xlabel="Lag time", ylabel=r"$|\rho_k(\tau)|$ (log scale)", ylim=(1e-3, 1.4),
                title="Decay of the mode autocorrelation")
    axes[0].grid(alpha=0.2, which="both")

    image = axes[1].imshow(np.abs(curves[:, inside]), origin="lower", aspect="auto",
                           extent=(lag_time[inside][0], lag_time[inside][-1],
                                   -0.5, len(modes) - 0.5),
                           cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].set(xlabel="Lag time", ylabel="Mode $k$", title=r"$|\rho_k(\tau)|$ heat map")
    axes[1].set_yticks(modes)
    fig.colorbar(image, ax=axes[1], label=r"$|\rho_k(\tau)|$")

    axes[2].bar(modes, tau, color="#b2182b")
    axes[2].set(xlabel="Mode $k$", ylabel=r"$\tau_{\rm int}$  (time)",
                title="Integrated decorrelation time")
    axes[2].set_xticks(modes)
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].grid(alpha=0.2, axis="y")

    fig.suptitle(rf"Temporal ACF of the Fourier mode time series, $F={forcing}$", y=1.06)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_acf_real_imag(path: Path, rows: list[dict], curves: np.ndarray, dt: float,
                       forcing: int, selected: tuple[int, ...], window: float) -> None:
    lag_time = np.arange(curves.shape[1]) * dt
    inside = lag_time <= window
    modes = [row["mode_k"] for row in rows]
    present = [mode for mode in selected if mode in modes]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.6),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, present):
        index = modes.index(mode)
        ax.plot(lag_time[inside], curves[index][inside].real, color="#2166ac", lw=1.4,
                label=r"$\Re\,\rho$")
        if np.max(np.abs(curves[index].imag)) > 1e-12:
            ax.plot(lag_time[inside], curves[index][inside].imag, color="#b2182b", lw=1.4,
                    ls="--", label=r"$\Im\,\rho$")
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set(xlabel="Lag time", title=rf"$k={mode}$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\rho_k(\tau)$")
    fig.suptitle(rf"Real and imaginary parts of the mode ACF, $F={forcing}$", y=1.08)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def write_mode_summary(path: Path, identifier: dict[str, float], rows: list[dict],
                       args: argparse.Namespace) -> str:
    modes = [row["mode_k"] for row in rows]
    shares = np.array([row["energy_share"] for row in rows])
    ess = np.array([row["ess"] for row in rows])
    tau = np.array([row["tau_int_time"] for row in rows])
    lines = [
        f"Input: {args.input}",
        f"Forcing: F={args.forcing}",
        f"Output directory: {args.output}",
        f"Burn-in discarded: t < {args.burn_time:g}",
        f"Samples per mode after burn-in: {rows[0]['n_samples']}",
        "",
        "Transform: unitary spatial DFT, xhat_k = N^-1/2 * sum_j x_j exp(-2*pi*i*(j-1)*k/N).",
        "Independent modes kept: k = 0 .. N/2 (all others follow from conjugate symmetry).",
        "Mode ACF: rho_k(tau) = E[conj(z_t) z_{t+tau}] / E[|z|^2] with the temporal mean removed.",
        "Integrated time: Geyer initial-positive sequence applied to Re rho_k(tau).",
        "Energy: E_k = dof_k * 0.5 * |xhat_k|^2, dof_k = 1 for k = 0 and k = N/2 else 2.",
        "Share: time-averaged E_k divided by the time-averaged total energy.",
        "",
        "Numerical self-checks",
        f"  independent modes ................. {identifier['independent_modes']:.0f}",
        f"  max |sum_k |xhat_k|^2 - ||x||^2| .. {identifier['max_abs_parseval_error']:.3e}",
        f"  max |xhat_(N-k) - conj(xhat_k)| ... {identifier['max_abs_conjugate_symmetry_error']:.3e}",
        f"  max |sum_k E_k - ||x||^2 / 2| ..... {identifier['max_abs_energy_partition_error']:.3e}",
        f"  time-averaged ||x||^2 / 2 ......... {identifier['mean_total_energy']:.6g}",
        "",
        "Mode summary",
        f"{'k':>3} {'dof':>4} {'<|xhat_k|^2>':>14} {'<E_k>':>15} {'share %':>9} "
        f"{'tau_int':>9} {'ESS':>10} {'LB p(50)':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['mode_k']:>3} {row['dof']:>4} {row['mean_abs2']:>14.6g} "
            f"{row['mean_energy']:>15.6g} {100.0 * row['energy_share']:>9.4f} "
            f"{row['tau_int_time']:>9.4f} {row['ess']:>10.2f} {row['ljung_box_p_50']:>10.3e}"
        )
    order = np.argsort(shares)[::-1]
    lines += ["", "Ranking by energy share"]
    for rank, index in enumerate(order, start=1):
        lines.append(f"  {rank:>2}. k = {modes[index]:>2}  share = {100.0 * shares[index]:.4f} %")
    lines += [
        "",
        f"tau_int (time) over modes: median {np.median(tau):.4f}, min {tau.min():.4f}, max {tau.max():.4f}",
        f"ESS over modes: median {np.median(ess):.2f}, min {ess.min():.2f}, max {ess.max():.2f}",
        f"cumulative share of k = 0..5 ........ {100.0 * np.sum(shares[:6]):.2f} %",
        "Ljung-Box Q(50) rejects at 95% confidence for "
        f"{int(np.sum(np.array([row['ljung_box_p_50'] for row in rows]) < 0.05))}/{len(rows)} modes",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("results/lorenz96_rk4_F64.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/fourier_modes"))
    parser.add_argument("--forcing", type=int, default=64)
    parser.add_argument("--burn-time", type=float, default=5.0)
    parser.add_argument("--max-lag", type=int, default=2000)
    parser.add_argument("--acf-plot-window", type=float, default=1.0,
                        help="Lag-time window used for the ACF figures (the mode series "
                             "decorrelate in well under one time unit at dt=0.01).")
    parser.add_argument("--self-check", action="store_true",
                        help="Compare the FFT autocorrelation against a direct O(n*lag) loop.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    times, states = read_solution(args.input)
    dt = float(times[1] - times[0])
    burn = int(round(args.burn_time / dt))
    times, states = times[burn:], states[burn:]

    modes_all = spatial_dft(states)
    keep = independent_mode_indices(states.shape[1])
    modes = modes_all[:, keep]
    dof = degrees_of_freedom(states.shape[1], keep)

    identifier = verify_identities(states, modes_all, keep, mode_energies(modes, dof))
    rows, curves = analyse_modes(modes, dof, dt, args.max_lag)

    if args.self_check:
        reference = brute_force_complex_acf(modes[:, 3], min(50, args.max_lag))
        error = float(np.max(np.abs(reference - curves[3, : reference.size])))
        print(f"ACF self-check (mode k=3, first {reference.size} lags) max abs error: {error:.3e}")

    # Numbers
    save_table(
        args.output / "mode_amplitudes_F64.csv",
        ["time"] + [f"abs_k{k:02d}" for k in keep],
        [times, np.abs(modes)],
    )
    save_table(
        args.output / "mode_real_imag_F64.csv",
        ["time"] + [f"re_k{k:02d}" for k in keep] + [f"im_k{k:02d}" for k in keep],
        [times, modes.real, modes.imag],
    )
    energy = mode_energies(modes, dof)
    save_table(
        args.output / "mode_energy_F64.csv",
        ["time"] + [f"E_k{k:02d}" for k in keep],
        [times, energy],
    )
    lag_time = np.arange(curves.shape[1]) * dt
    save_table(
        args.output / f"fourier_acf_curves_F{args.forcing}.csv",
        ["mode_k", "lag_time", "acf_real", "acf_imag", "acf_abs"],
        [np.repeat(keep, curves.shape[1]),
         np.tile(lag_time, len(keep)),
         curves.real.reshape(-1),
         curves.imag.reshape(-1),
         np.abs(curves).reshape(-1)],
    )
    with (args.output / f"fourier_mode_summary_F{args.forcing}.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Figures
    plot_energy_spectrum(args.output / f"fourier_energy_spectrum_F{args.forcing}.png", rows, args.forcing)
    plot_wavenumber_time(args.output / f"fourier_wavenumber_time_F{args.forcing}.png",
                         times, energy, args.forcing)
    plot_acf_curves(args.output / f"fourier_acf_F{args.forcing}.png", rows, curves, dt,
                    args.forcing, args.acf_plot_window)
    plot_acf_real_imag(args.output / f"fourier_acf_real_imag_F{args.forcing}.png",
                       rows, curves, dt, args.forcing, (0, 1, 5, 10, 20), args.acf_plot_window)

    text = write_mode_summary(args.output / "summary.txt", identifier, rows, args)
    print(text)


if __name__ == "__main__":
    main()
