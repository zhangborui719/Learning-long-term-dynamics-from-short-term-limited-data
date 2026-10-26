"""Train and evaluate a neural flow map for the F=64 Lorenz-96 system.

This is *method 1* of the four regression strategies described in
:mod:`flowmap_data`: the network learns the one-step propagator directly,

    x(t + dt) = Phi_dt(x(t)),   dt = 0.1,

on the first two thirds of the record and is then asked to predict the final
third.

Evaluation deliberately goes beyond the training objective.  A one-step mean
squared error of a few times ``1e-3`` says nothing about whether the map is
*usable*: a network can shade the truth at short lag while drifting into a
non-physical region as soon as it is iterated.  We therefore report

* the one-step error against two baselines (persistence and a ridge-fit affine map),
* the teacher-forced ``k``-step error growth for ``k = 1 .. 50``,
* a free-running rollout over the whole test segment,
* the reproduced attractor statistics (mean, variance, energy spectrum).

Usage::

    python train_flowmap.py --mode direct --epochs 2000
    python train_flowmap.py --self-check
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flowmap_data import (
    SUPPORTED_MODES,
    DirectCodec,
    TargetCodec,
    build_codec,
    build_split,
    encode_dataset,
    fit_ridge_flow_map,
)
from flowmap_model import FlowMapMLP, predict, train_flow_map
from solve_lorenz96 import rk4_step

LOGGER = logging.getLogger("flowmap.train")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "flowmap"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def normalized_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error normalised by the RMS amplitude of ``truth``.

    Args:
        prediction: predicted states.
        truth: reference states with the same shape.

    Returns:
        ``sqrt(<(pred - truth)^2>) / sqrt(<truth^2>)``; dimensionless.
    """
    error = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    reference = float(np.sqrt(np.mean(truth**2)))
    return error / reference if reference > 0.0 else float("inf")


def one_step_report(
    model: FlowMapMLP,
    codec: TargetCodec,
    split: Any,
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    """One-step accuracy of the network and of two reference predictors.

    Args:
        model: trained network.
        codec: target codec used during training.
        split: chronological split (for physical inputs and targets).
        data: encoded arrays from :func:`flowmap_data.encode_dataset`.

    Returns:
        A flat dict of metrics, all dimensionless unless stated otherwise.
    """
    train_pred = codec.decode(data["train_x"], predict(model, data["train_x"]))
    test_pred = codec.decode(data["test_x"], predict(model, data["test_x"]))

    # persistence: the trivial map Phi(x) = x
    persistence = split.test_input
    # affine baseline in standardised coordinates, fitted on training pairs only
    z_train_in = codec.standardize(split.train_input)
    z_train_out = codec.standardize(split.train_target)
    weights, bias = fit_ridge_flow_map(z_train_in, z_train_out)
    z_test_in = codec.standardize(split.test_input)
    affine = codec.destandardize(z_test_in @ weights.T + bias)

    metrics = {
        "train_nrmse": normalized_rmse(train_pred, split.train_target),
        "test_nrmse": normalized_rmse(test_pred, split.test_target),
        "test_nrmse_persistence": normalized_rmse(persistence, split.test_target),
        "test_nrmse_affine": normalized_rmse(affine, split.test_target),
    }
    metrics["skill_vs_persistence"] = (
        1.0 - metrics["test_nrmse"] / metrics["test_nrmse_persistence"]
    )
    metrics["skill_vs_affine"] = 1.0 - metrics["test_nrmse"] / metrics["test_nrmse_affine"]
    return metrics


def teacher_forced_error_curve(
    model: FlowMapMLP,
    codec: TargetCodec,
    split: Any,
    max_horizon: int = 50,
    max_starts: int = 500,
) -> np.ndarray:
    """Relative error of a ``k``-step rollout restarted from true states.

    For every start index ``i`` the network is iterated ``k`` times from the true
    ``x_i`` and compared with the true ``x_{i+k}``.  The starts are re-anchored at
    truth, so the curve isolates *how fast the learned map accumulates error* from
    the separate question of trajectory stability.

    Args:
        model: trained network.
        codec: target codec.
        split: chronological split.
        max_horizon: largest ``k`` to evaluate.
        max_starts: cap on the number of start indices (evenly spaced).

    Returns:
        Array of shape ``(max_horizon,)`` holding the normalised error at each
        horizon ``k = 1 .. max_horizon``.
    """
    n_available = split.test_input.shape[0]
    max_horizon = int(min(max_horizon, n_available))
    starts = np.linspace(0, n_available - max_horizon, num=min(max_starts, n_available))
    starts = np.unique(starts.astype(int))
    reference_rms = float(np.sqrt(np.mean(split.test_target**2)))

    errors = np.zeros(max_horizon)
    # states is (n_starts, d): the current iterate for each start, always reset to truth
    states = split.test_input[starts].copy()
    for step in range(max_horizon):
        z_in = codec.standardize(states)
        states = codec.decode(z_in, predict(model, z_in))
        truth = split.test_target[starts + step]
        errors[step] = np.sqrt(np.mean((states - truth) ** 2)) / reference_rms
    return errors


def free_rollout(
    model: FlowMapMLP,
    codec: TargetCodec,
    initial_state: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Iterate the learned map for ``n_steps`` without any truth anchoring.

    Args:
        model: trained network.
        codec: target codec.
        initial_state: physical state to start from, shape ``(d,)``.
        n_steps: number of compositions of the map.

    Returns:
        Array of shape ``(n_steps, d)`` with the predicted states.
    """
    state = np.asarray(initial_state, dtype=np.float64)[None, :]
    trajectory = np.empty((n_steps, state.shape[1]), dtype=np.float64)
    for step in range(n_steps):
        z_in = codec.standardize(state)
        state = codec.decode(z_in, predict(model, z_in))
        trajectory[step] = state[0]
    return trajectory


def crossing_time(curve: np.ndarray, dt: float, threshold: float) -> float:
    """First time at which ``curve`` exceeds ``threshold``.

    Args:
        curve: error samples at uniform spacing ``dt``.
        dt: spacing between samples.
        threshold: dimensionless error level.

    Returns:
        The crossing time in physical units, or ``nan`` if never reached.
    """
    above = np.nonzero(curve > threshold)[0]
    if above.size == 0:
        return float("nan")
    return float((above[0] + 1) * dt)


def largest_lyapunov(
    state: np.ndarray,
    forcing: float,
    dt: float,
    t_total: float = 200.0,
    renormalize_every: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Benettin renormalisation estimate of the largest Lyapunov exponent.

    A nearest-neighbour (Rosenstein) estimate is unreliable here: with 500 states
    in 40 dimensions the nearest neighbour is farther away than the attractor
    diameter, so there is no exponential regime to fit.  We therefore integrate
    the true right-hand side, evolving a reference state and an infinitesimally
    perturbed companion, and renormalise the separation every few steps.

    Args:
        state: physical state used as the reference initial condition.
        forcing: the ``F`` parameter.
        dt: integration step (must be the solver's ``0.01`` - RK4 is unstable at
            ``0.1`` for ``F=64``).
        t_total: total integration time.
        renormalize_every: steps between renormalisations.
        seed: seed for the initial perturbation direction.

    Returns:
        Dict with ``lambda_max`` (per unit time), ``lyapunov_time`` and
        ``predictability_time`` (time for a ``1e-8`` error to reach 30 % of the
        attractor rms).
    """
    rng = np.random.default_rng(seed)
    reference = np.asarray(state, dtype=np.float64).copy()
    direction = rng.normal(size=reference.size)
    direction /= np.linalg.norm(direction)
    initial_separation = 1e-8
    companion = reference + initial_separation * direction

    n_steps = int(round(t_total / dt))
    log_growth = 0.0
    n_renorm = 0
    for step in range(1, n_steps + 1):
        reference = rk4_step(reference, dt, forcing)
        companion = rk4_step(companion, dt, forcing)
        if step % renormalize_every == 0:
            separation = companion - reference
            norm = float(np.linalg.norm(separation))
            if not np.isfinite(norm) or norm <= 0.0:
                break
            log_growth += np.log(norm / initial_separation)
            companion = reference + separation * (initial_separation / norm)
            n_renorm += 1

    if n_renorm == 0:
        return {"lambda_max": float("nan"), "lyapunov_time": float("nan"),
                "predictability_time": float("nan")}
    exponent = log_growth / (n_renorm * renormalize_every * dt)
    amplitude = float(np.sqrt(np.mean(reference**2)))
    if exponent <= 0.0:
        predictability = float("inf")
    else:
        predictability = np.log(0.3 * amplitude / initial_separation) / exponent
    return {
        "lambda_max": float(exponent),
        "lyapunov_time": float(1.0 / exponent) if exponent > 0 else float("nan"),
        "predictability_time": float(predictability),
    }


def attractor_statistics(trajectory: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare low-order statistics of a rollout with the true test segment.

    Args:
        trajectory: predicted states, shape ``(n, d)``.
        reference: true states, same shape.

    Returns:
        Dict of ratios and correlations summarising the reproduced attractor.
    """
    def spectrum(field: np.ndarray) -> np.ndarray:
        coeff = np.fft.fft(field, axis=1) / np.sqrt(field.shape[1])
        power = np.abs(coeff) ** 2
        independent = field.shape[1] // 2 + 1
        dof = np.ones(independent)
        dof[1 : field.shape[1] // 2] = 2.0
        return (power[:, :independent] * dof / 2.0).mean(axis=0)

    predicted_spectrum = spectrum(trajectory)
    reference_spectrum = spectrum(reference)
    return {
        "mean_ratio": float(trajectory.mean() / reference.mean()),
        "rms_ratio": float(
            np.sqrt((trajectory**2).mean()) / np.sqrt((reference**2).mean())
        ),
        "spectrum_correlation": float(
            np.corrcoef(np.log(predicted_spectrum[1:]), np.log(reference_spectrum[1:]))[0, 1]
        ),
        "spectrum_peak_mode": float(np.argmax(predicted_spectrum[1:]) + 1),
        "reference_peak_mode": float(np.argmax(reference_spectrum[1:]) + 1),
    }


def shift_augment(
    inputs: np.ndarray, targets: np.ndarray, n_shift: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expand the training set using the cyclic symmetry of Lorenz-96.

    The equations are invariant under a rotation of the index ``j``, so
    ``Phi(roll(x, m)) = roll(Phi(x), m)`` for every offset ``m``.  Exploiting that
    is not data leakage: it is a symmetry of the dynamical system, and it is
    applied to the *training* pairs only.

    This matters more than it looks.  With 966 training pairs in a 40-dimensional
    space the nearest neighbour of a test state is farther away than the attractor
    itself, so a plain MLP can only memorise: the binding constraint is the number
    of distinct training points, not the network.  Rotation lifts 966 samples to
    38 640 at zero extra measurement cost.

    Args:
        inputs: standardised training inputs, shape ``(n, d)``.
        targets: encoded training targets, shape ``(n, d)``.
        n_shift: number of cyclic offsets to include (``d`` covers the full group).

    Returns:
        Augmented ``(inputs, targets)`` stacked along the sample axis.
    """
    n_shift = int(n_shift)
    if n_shift <= 1:
        return inputs, targets
    n_shift = min(n_shift, inputs.shape[1])
    return (
        np.vstack([np.roll(inputs, m, axis=1) for m in range(n_shift)]),
        np.vstack([np.roll(targets, m, axis=1) for m in range(n_shift)]),
    )


def delta_metrics(
    prediction: np.ndarray, truth: np.ndarray, initial: np.ndarray
) -> dict[str, float]:
    """How much of the *increment* the model captures.

    The states themselves move by more than one standard deviation per 0.1 time
    unit, so an error in ``x(t+dt)`` can look large even when the predicted
    direction is right.  These metrics isolate the increment.

    Args:
        prediction: predicted next states.
        truth: true next states.
        initial: the states the prediction started from.

    Returns:
        Dict with the increment NRMSE and the correlation of the increment
        components.
    """
    predicted_increment = prediction - initial
    true_increment = truth - initial
    scale = float(np.sqrt(np.mean(true_increment**2)))
    return {
        "delta_nrmse": float(
            np.sqrt(np.mean((predicted_increment - true_increment) ** 2)) / scale
        ),
        "delta_correlation": float(
            np.corrcoef(predicted_increment.ravel(), true_increment.ravel())[0, 1]
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_training(path: Path, history: dict[str, np.ndarray], forcing: int) -> None:
    """Loss and gradient-norm diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), constrained_layout=True)
    axes[0].semilogy(history["epoch"], history["train_loss"], lw=1.6, label="train (fit)")
    axes[0].semilogy(history["epoch"], history["test_loss"], lw=1.6, label="test (held out)")
    axes[0].set(xlabel="Epoch", ylabel="MSE in standardised space",
                title="Regression loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25, which="both")

    axes[1].semilogy(history["epoch"], history["grad_norm"], lw=1.2, color="#7b3294")
    axes[1].set(xlabel="Epoch", ylabel="Mean batch gradient norm",
                title="Gradient magnitude (before clipping)")
    axes[1].grid(alpha=0.25, which="both")

    twin = axes[1].twinx()
    twin.plot(history["epoch"], history["learning_rate"], lw=1.0, color="#e08214", ls="--")
    twin.set_ylabel("Learning rate", color="#e08214")
    twin.tick_params(axis="y", labelcolor="#e08214")
    twin.set_yscale("log")

    fig.suptitle(rf"Flow-map training, $F={forcing:g}$", y=1.04)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_one_step(path: Path, prediction: np.ndarray, truth: np.ndarray, forcing: int) -> None:
    """Scatter of predicted versus true next state, plus the residual histogram."""
    flat_pred = prediction.ravel()
    flat_true = truth.ravel()
    rng = np.random.default_rng(0)
    sample = rng.choice(flat_pred.size, size=min(40000, flat_pred.size), replace=False)

    residual = (prediction - truth) / truth.std()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    axes[0].scatter(flat_true[sample], flat_pred[sample], s=4, alpha=0.25,
                    color="#2166ac", edgecolors="none")
    low = float(np.min(flat_true[sample]))
    high = float(np.max(flat_true[sample]))
    axes[0].plot([low, high], [low, high], color="0.25", lw=1.0, ls="--", label="$y=x$")
    axes[0].set(xlabel="True $x_j(t+0.1)$", ylabel="Predicted $x_j(t+0.1)$",
                title="One-step prediction, held-out segment")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].hist(residual.ravel(), bins=120, color="#b2182b", alpha=0.75, density=True)
    axes[1].axvline(0.0, color="0.25", lw=1.0)
    axes[1].set(xlabel="Residual / std of the state", ylabel="Density",
                title=f"Residual distribution (std = {residual.std():.4f})")
    axes[1].grid(alpha=0.2)

    fig.suptitle(rf"Single-step accuracy of the learned flow map, $F={forcing:g}$", y=1.04)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_error_growth(
    path: Path,
    horizons: np.ndarray,
    curve: np.ndarray,
    baselines: dict[str, np.ndarray],
    crossing: dict[str, float],
    lyapunov: dict[str, float],
    dt: float,
    forcing: int,
) -> None:
    """Teacher-forced error growth against the two baselines.

    The interesting region is the first few steps: with ``tau_Lyap ~ 0.085`` a
    0.1 step error saturates almost immediately, and everything past horizon ~5
    just says "the prediction is uncorrelated with the truth".
    """
    time_axis = horizons * dt
    fig, ax = plt.subplots(figsize=(9.6, 5.0), constrained_layout=True)
    ax.semilogy(time_axis, curve, "o-", ms=4, lw=1.8, color="#2166ac",
                label="learned flow map")
    colours = {"persistence": "#b2182b", "affine_ridge": "#e08214"}
    labels = {"persistence": "persistence, $\\Phi(x)=x$", "affine_ridge": "affine ridge map"}
    for name, values in baselines.items():
        ax.semilogy(time_axis, values, "--", lw=1.4, color=colours.get(name, "0.4"),
                    label=labels.get(name, name))

    # A prediction that keeps the right amplitude but loses all correlation with
    # the truth has relative error sqrt(2); that is the plateau all curves reach.
    ax.axhline(np.sqrt(2.0), color="0.45", lw=1.0, ls="-.")
    ax.annotate("uncorrelated prediction ($\\sqrt{2}$)", xy=(time_axis[0], np.sqrt(2.0)),
                xytext=(4, 5), textcoords="offset points", fontsize=8, color="0.35")

    ax.axhline(0.3, color="0.6", lw=0.9, ls=":")
    ax.annotate("30 % relative error", xy=(time_axis[0], 0.3), xytext=(4, 5),
                textcoords="offset points", fontsize=8, color="0.35")

    summary = "\n".join(
        f"{name}: {value:.2f}" for name, value in crossing.items() if np.isfinite(value)
    )
    ax.text(0.985, 0.96, "time to 30 % error\n" + summary, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="0.2",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})

    subtitle = (
        rf"$\tau_{{Lyap}} \approx {lyapunov['lyapunov_time']:.3f}$ time units"
        if np.isfinite(lyapunov["lyapunov_time"])
        else ""
    )
    ax.set(xlabel="Rollout horizon (time units, truth-anchored restart)",
           ylabel=r"Relative error $\|pred - true\| / \|true\|_{rms}$",
           title=f"Error growth per horizon, $F={forcing:g}$   {subtitle}")
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_rollout(
    path: Path,
    times: np.ndarray,
    truth: np.ndarray,
    trajectory: np.ndarray,
    components: tuple[int, ...],
    forcing: int,
    envelope_window: int = 25,
) -> None:
    """Free-running prediction against truth, plus the amplitude envelope.

    The bottom panel is the point of the figure.  A regression network trained on
    one-step targets contracts towards the conditional mean whenever it is
    uncertain, and iterating that contraction produces a *smaller* attractor
    rather than a divergence: the prediction keeps oscillating, at
    ``rms_ratio < 1``.  Plotting the running rms makes that collapse visible.
    """
    present = [c for c in components if 0 <= c < truth.shape[1]]
    length = min(len(trajectory), len(times))
    n_panels = len(present)

    fig = plt.figure(figsize=(11.2, 1.9 * n_panels + 2.9))
    grid = fig.add_gridspec(n_panels + 1, 1, height_ratios=[1.0] * n_panels + [1.45],
                            hspace=0.30, top=0.935, bottom=0.075,
                            left=0.085, right=0.985)
    axes = [fig.add_subplot(grid[index]) for index in range(n_panels)]
    ax_envelope = fig.add_subplot(grid[n_panels])

    for index, (ax, component) in enumerate(zip(axes, present)):
        ax.plot(times[:length], truth[:length, component], lw=1.6, color="0.15",
                label="truth" if index == 0 else None)
        ax.plot(times[:length], trajectory[:length, component], lw=1.2, color="#2166ac",
                alpha=0.9, label="free-running prediction" if index == 0 else None)
        ax.set_ylabel(f"$x_{{{component}}}$")
        ax.grid(alpha=0.2)
        if index == 0:
            ax.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.set_xticklabels([])

    def running_rms(field: np.ndarray) -> np.ndarray:
        kernel = np.ones(envelope_window) / envelope_window
        power = np.convolve((field**2).mean(axis=1), kernel, mode="valid")
        return np.sqrt(np.maximum(power, 0.0))

    reference = running_rms(truth[:length])
    predicted = running_rms(trajectory[:length])
    centre = times[:length][envelope_window // 2 : envelope_window // 2 + reference.size]
    ax_envelope.plot(centre, reference, lw=1.8, color="0.15", label="truth")
    ax_envelope.plot(centre, predicted, lw=1.6, color="#b2182b", label="prediction")
    ratio = float(np.mean(predicted / np.maximum(reference, 1e-12)))
    ax_envelope.axhline(0.0, color="0.6", lw=0.8)
    ax_envelope.set(xlabel="Time", ylabel="Running rms of $x$",
                    title=f"Amplitude envelope (mean ratio {ratio:.2f})")
    ax_envelope.grid(alpha=0.2)
    ax_envelope.legend(loc="lower right", fontsize=8)

    fig.suptitle(rf"Free-running rollout over the held-out segment, $F={forcing:g}$", y=0.985)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spectrum(
    path: Path,
    truth: np.ndarray,
    trajectory: np.ndarray,
    forcing: int,
) -> None:
    """Time-averaged energy spectrum of the rollout against truth."""
    def spectrum(field: np.ndarray) -> np.ndarray:
        coeff = np.fft.fft(field, axis=1) / np.sqrt(field.shape[1])
        power = np.abs(coeff) ** 2
        independent = field.shape[1] // 2 + 1
        dof = np.ones(independent)
        dof[1 : field.shape[1] // 2] = 2.0
        return (power[:, :independent] * dof / 2.0).mean(axis=0)

    inside = spectrum(truth[: len(trajectory)])
    outside = spectrum(trajectory)
    modes = np.arange(inside.size)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.semilogy(modes, inside, "o-", ms=4, lw=1.6, color="0.15", label="truth")
    ax.semilogy(modes, np.maximum(outside, 1e-12), "s--", ms=4, lw=1.4,
                color="#b2182b", label="free-running prediction")
    ax.set(xlabel="Wavenumber $k$", ylabel=r"Time-averaged mode energy $E_k$",
           title=rf"Energy spectrum reproduced by the rollout, $F={forcing:g}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def write_summary(path: Path, rows: list[tuple[str, str]], forcing: int) -> None:
    """Write a two-column key/value text report."""
    width = max(len(key) for key, _ in rows) + 2
    lines = [rf"Flow-map learning on Lorenz-96, F = {forcing}", "=" * 62]
    for key, value in rows:
        lines.append(f"{key:<{width}} {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full pipeline for one configuration.

    Args:
        args: parsed command-line arguments.

    Returns:
        A dict of all scalar results, also written to ``metrics.json``.
    """
    started = time.perf_counter()
    forcing = float(args.forcing)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = build_split(
        Path(args.data), stride=args.stride, train_fraction=args.train_fraction,
        burn_in=args.burn_in,
    )
    codec = build_codec(args.mode, split)
    data = encode_dataset(split, codec)
    LOGGER.info("mode: %s", codec.description)

    model = FlowMapMLP(
        n_dim=split.n_dim,
        hidden_width=args.hidden_width,
        n_hidden_layers=args.hidden_layers,
        activation=args.activation,
    )
    LOGGER.info("model: %s", model.describe())

    train_x, train_y = shift_augment(
        data["train_x"], data["train_y"], args.shift_augment
    )
    if args.shift_augment > 1:
        LOGGER.info(
            "shift augmentation: %d -> %d training pairs (cyclic symmetry)",
            data["train_x"].shape[0],
            train_x.shape[0],
        )

    history = train_flow_map(
        model,
        train_x,
        train_y,
        data["test_x"],
        data["test_y"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=max(1, args.epochs // 10),
    )

    metrics: dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "dt": split.dt,
        "shift_augment": args.shift_augment,
        "burn_in": args.burn_in,
        "n_train_augmented": int(train_x.shape[0]),
    }
    metrics.update(one_step_report(model, codec, split, data))
    test_prediction = codec.decode(data["test_x"], predict(model, data["test_x"]))
    metrics.update(delta_metrics(test_prediction, split.test_target, split.test_input))

    # error growth, with the two baselines evaluated on the same grid
    horizons = np.arange(1, args.max_horizon + 1)
    curve = teacher_forced_error_curve(model, codec, split, max_horizon=args.max_horizon)
    persistence_curve = teacher_forced_error_curve_persistence(split, args.max_horizon)
    z_train_in = codec.standardize(split.train_input)
    z_train_out = codec.standardize(split.train_target)
    weights, bias = fit_ridge_flow_map(z_train_in, z_train_out)
    affine_curve = teacher_forced_error_curve_affine(
        codec, split, weights, bias, args.max_horizon
    )
    crossing = {
        "map": crossing_time(curve, split.dt, args.error_threshold),
        "persistence": crossing_time(persistence_curve, split.dt, args.error_threshold),
        "affine": crossing_time(affine_curve, split.dt, args.error_threshold),
    }
    metrics["error_threshold"] = args.error_threshold
    metrics["valid_time_map"] = crossing["map"]
    metrics["valid_time_persistence"] = crossing["persistence"]
    metrics["valid_time_affine"] = crossing["affine"]
    metrics["nrmse_at_horizon_1"] = float(curve[0])
    metrics[f"nrmse_at_horizon_{args.max_horizon}"] = float(curve[-1])

    # free-running rollout over the whole test segment
    trajectory = free_rollout(model, codec, split.test_input[0], split.test_input.shape[0])
    metrics["rollout_nrmse"] = normalized_rmse(trajectory, split.test_target)
    metrics.update(attractor_statistics(trajectory[: len(split.test_target)], split.test_target))

    lyapunov = largest_lyapunov(split.test_input[0], forcing=forcing, dt=0.01)
    metrics.update({f"lyapunov_{k}": v for k, v in lyapunov.items()})
    # Effective initial error implied by the one-step accuracy: the network's
    # single-step NRMSE times the state rms.  Comparing it with the 1e-8 used in
    # the Lyapunov integration isolates how much of the shortness of the useful
    # horizon is chaos and how much is approximation error.
    metrics["effective_initial_error"] = metrics["test_nrmse"] * float(
        np.sqrt(np.mean(split.test_target**2))
    )
    if np.isfinite(lyapunov["lambda_max"]) and lyapunov["lambda_max"] > 0:
        metrics["implied_horizon_from_error"] = float(
            np.log(0.3 * np.sqrt(np.mean(split.test_target**2))
                    / metrics["effective_initial_error"]) / lyapunov["lambda_max"]
        )
    else:
        metrics["implied_horizon_from_error"] = float("nan")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    plot_training(output / f"flowmap_training_F{forcing:g}.png", history.as_arrays(), forcing)
    one_step_pred = codec.decode(data["test_x"], predict(model, data["test_x"]))
    plot_one_step(output / f"flowmap_one_step_F{forcing:g}.png", one_step_pred,
                  split.test_target, forcing)
    plot_error_growth(
        output / f"flowmap_error_growth_F{forcing:g}.png",
        horizons,
        curve,
        {"persistence": persistence_curve, "affine_ridge": affine_curve},
        crossing,
        lyapunov,
        split.dt,
        forcing,
    )
    plot_rollout(
        output / f"flowmap_rollout_F{forcing:g}.png",
        split.test_time[0] + np.arange(len(trajectory)) * split.dt,
        split.test_target,
        trajectory,
        (0, 4, 10, 20),
        forcing,
    )
    plot_spectrum(output / f"flowmap_spectrum_F{forcing:g}.png", split.test_target,
                  trajectory[: len(split.test_target)], forcing)

    # numeric artefacts
    np.savetxt(
        output / f"flowmap_error_growth_F{forcing:g}.csv",
        np.column_stack([horizons, horizons * split.dt, curve, persistence_curve, affine_curve]),
        delimiter=",",
        header="horizon_steps,horizon_time,nrmse_map,nrmse_persistence,nrmse_affine",
        comments="",
        fmt="%.6e",
    )
    np.savetxt(
        output / f"flowmap_rollout_F{forcing:g}.csv",
        np.column_stack([split.test_time, trajectory[: len(split.test_target)],
                         split.test_target]),
        delimiter=",",
        header="time," + ",".join(f"pred_{j}" for j in range(split.n_dim))
        + "," + ",".join(f"true_{j}" for j in range(split.n_dim)),
        comments="",
        fmt="%.6e",
    )
    np.savetxt(
        output / f"flowmap_training_F{forcing:g}.csv",
        np.column_stack(list(history.as_arrays().values())),
        delimiter=",",
        header=",".join(history.as_arrays().keys()),
        comments="",
        fmt="%.6e",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": split.mean,
            "scale": split.scale,
            "dt": split.dt,
            "mode": args.mode,
            "architecture": model.describe(),
        },
        output / f"flowmap_model_F{forcing:g}.pt",
    )

    metrics["train_seconds"] = time.perf_counter() - started
    metrics["n_parameters"] = model.n_parameters
    metrics["n_train_pairs"] = int(split.train_input.shape[0])
    metrics["n_test_pairs"] = int(split.test_input.shape[0])
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = [("method", codec.description), ("architecture", model.describe())]
    rows += split.summary_rows()
    rows += [("shift augmentation", f"{args.shift_augment}x cyclic -> "
                                   f"{metrics['n_train_augmented']} pairs")]
    rows += [("", "")]
    rows += [("one-step NRMSE (train)", f"{metrics['train_nrmse']:.5f}"),
             ("one-step NRMSE (test)", f"{metrics['test_nrmse']:.5f}"),
             ("one-step NRMSE persistence", f"{metrics['test_nrmse_persistence']:.5f}"),
             ("one-step NRMSE affine ridge", f"{metrics['test_nrmse_affine']:.5f}"),
             ("skill vs persistence", f"{100 * metrics['skill_vs_persistence']:.2f} %"),
             ("skill vs affine", f"{100 * metrics['skill_vs_affine']:.2f} %"),
             ("increment NRMSE (test)", f"{metrics['delta_nrmse']:.5f}"),
             ("increment correlation", f"{metrics['delta_correlation']:.5f}"),
             ("", "")]
    rows += [
        (f"NRMSE at horizon 1 step", f"{curve[0]:.5f}"),
        (f"NRMSE at horizon {args.max_horizon} steps", f"{curve[-1]:.5f}"),
        (f"valid time (map, {100 * args.error_threshold:.0f}% err)",
         f"{crossing['map']:.3f} time units"),
        ("valid time (persistence)", f"{crossing['persistence']:.3f} time units"),
        ("valid time (affine ridge)", f"{crossing['affine']:.3f} time units"),
        ("", ""),
    ]
    rows += [
        ("free-running NRMSE", f"{metrics['rollout_nrmse']:.5f}"),
        ("rollout mean / truth mean", f"{metrics['mean_ratio']:.4f}"),
        ("rollout rms / truth rms", f"{metrics['rms_ratio']:.4f}"),
        ("spectrum correlation (log E, k>=1)", f"{metrics['spectrum_correlation']:.4f}"),
        ("spectrum peak mode", f"{metrics['spectrum_peak_mode']:.0f} "
                              f"(truth {metrics['reference_peak_mode']:.0f})"),
        ("", ""),
    ]
    rows += [
        ("largest Lyapunov exponent", f"{lyapunov['lambda_max']:.4f} per time unit"),
        ("Lyapunov time", f"{lyapunov['lyapunov_time']:.3f} time units"),
        ("perfect-model predictability (1e-8 -> 30%)",
         f"{lyapunov['predictability_time']:.2f} time units"),
        ("effective initial error of the map",
         f"{metrics['effective_initial_error']:.3f} (state rms "
         f"{np.sqrt(np.mean(split.test_target**2)):.2f})"),
        ("horizon implied by that error",
         f"{metrics['implied_horizon_from_error']:.3f} time units"),
        ("", ""),
    ]
    rows += [("wall clock", f"{metrics['train_seconds']:.1f} s"),
             ("trainable parameters", f"{model.n_parameters}")]
    write_summary(output / "summary.txt", rows, forcing)

    return metrics


def teacher_forced_error_curve_persistence(split: Any, max_horizon: int) -> np.ndarray:
    """Error growth of the trivial map ``Phi(x) = x`` for reference.

    Args:
        split: chronological split.
        max_horizon: largest ``k``.

    Returns:
        Normalised error at each horizon ``k = 1 .. max_horizon``.
    """
    n_available = split.test_input.shape[0]
    max_horizon = int(min(max_horizon, n_available))
    starts = np.linspace(0, n_available - max_horizon, num=min(500, n_available))
    starts = np.unique(starts.astype(int))
    reference_rms = float(np.sqrt(np.mean(split.test_target**2)))
    errors = np.zeros(max_horizon)
    for step in range(max_horizon):
        truth = split.test_target[starts + step]
        errors[step] = np.sqrt(np.mean((split.test_input[starts] - truth) ** 2)) / reference_rms
    return errors


def teacher_forced_error_curve_affine(
    codec: TargetCodec,
    split: Any,
    weights: np.ndarray,
    bias: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    """Error growth of the affine ridge map ``A z + b`` for reference.

    Args:
        codec: target codec supplying the standardisation.
        split: chronological split.
        weights: affine map matrix, shape ``(d, d)``.
        bias: affine map intercept, shape ``(d,)``.
        max_horizon: largest ``k``.

    Returns:
        Normalised error at each horizon ``k = 1 .. max_horizon``.
    """
    n_available = split.test_input.shape[0]
    max_horizon = int(min(max_horizon, n_available))
    starts = np.linspace(0, n_available - max_horizon, num=min(500, n_available))
    starts = np.unique(starts.astype(int))
    reference_rms = float(np.sqrt(np.mean(split.test_target**2)))
    errors = np.zeros(max_horizon)
    states = split.test_input[starts].copy()
    for step in range(max_horizon):
        z_in = codec.standardize(states)
        states = codec.destandardize(z_in @ weights.T + bias)
        truth = split.test_target[starts + step]
        errors[step] = np.sqrt(np.mean((states - truth) ** 2)) / reference_rms
    return errors


def self_check(args: argparse.Namespace) -> int:
    """Fast sanity checks on the data pipeline and the network.

    Args:
        args: parsed command-line arguments.

    Returns:
        Process exit code (0 on success).
    """
    split = build_split(Path(args.data), stride=args.stride,
                        train_fraction=args.train_fraction, burn_in=args.burn_in)
    codec = build_codec(args.mode, split)
    model = FlowMapMLP(split.n_dim, args.hidden_width, args.hidden_layers, args.activation)

    checks: list[tuple[str, str, bool]] = []
    expected_dt = 0.01 * args.stride
    checks.append(("sampling interval", f"dt = {split.dt:.6g}",
                   abs(split.dt - expected_dt) < 1e-9))
    # The last training target must sit at or before the first test input, i.e.
    # the boundary-straddling transition must have been dropped.
    checks.append(("no boundary leak", "last train target <= first test input",
                   split.train_time[-1] + split.dt <= split.test_time[0] + 1e-9))
    # Exactly one transition -- the one straddling the split instant -- is dropped.
    dropped = (split.test_time[0] - split.train_time[-1]) / split.dt - 1.0
    checks.append(("dropped transitions", f"{dropped:.0f} straddling pair dropped",
                   abs(dropped - 1.0) < 1e-9))
    checks.append(("chronological split",
                   f"{split.train_input.shape[0]} / {split.test_input.shape[0]} pairs",
                   split.train_time[-1] < split.test_time[0]))
    checks.append(("parameter count", f"{model.n_parameters}", model.n_parameters == 13000))

    physical = split.test_input[:4]
    round_trip = codec.destandardize(codec.standardize(physical))
    checks.append(("standardisation round trip", f"max err {np.abs(round_trip - physical).max():.2e}",
                   np.allclose(round_trip, physical, atol=1e-10)))

    z_in = codec.standardize(physical)
    encoded = codec.encode(z_in, split.test_target[:4])
    recovered = codec.decode(z_in, encoded)
    checks.append(("codec round trip", f"max err {np.abs(recovered - split.test_target[:4]).max():.2e}",
                   np.allclose(recovered, split.test_target[:4], atol=1e-8)))

    width = max(len(name) for name, _, _ in checks) + 2
    print("Self-check")
    print("-" * 62)
    for name, detail, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
    print("-" * 62)
    return 0 if all(ok for _, _, ok in checks) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="solver CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="artefact directory")
    parser.add_argument("--mode", default="direct", choices=SUPPORTED_MODES,
                        help="regression strategy")
    parser.add_argument("--stride", type=int, default=10,
                        help="resampling stride applied to the raw dt=0.01 record")
    parser.add_argument("--train-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--burn-in", type=float, default=5.0,
                        help="physical time discarded from the start of the record "
                             "(the spin-up transient is not on the attractor)")
    parser.add_argument("--forcing", type=float, default=64.0, help="the F parameter")
    parser.add_argument("--shift-augment", type=int, default=40,
                        help="number of cyclic offsets used to augment the training "
                             "pairs; 1 disables augmentation, 40 uses the full group")
    parser.add_argument("--hidden-width", type=int, default=80)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", default="tanh",
                        choices=["tanh", "silu", "gelu", "relu"])
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--error-threshold", type=float, default=0.3)
    parser.add_argument("--self-check", action="store_true",
                        help="run the fast consistency checks and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.self_check:
        return self_check(args)
    metrics = run(args)
    print("\nKey results")
    print("-" * 66)
    print(f"  one-step NRMSE (test)      {metrics['test_nrmse']:.5f}")
    print(f"  persistence NRMSE          {metrics['test_nrmse_persistence']:.5f}")
    print(f"  affine ridge NRMSE         {metrics['test_nrmse_affine']:.5f}")
    print(f"  skill vs persistence       {100 * metrics['skill_vs_persistence']:.2f} %")
    print(f"  increment NRMSE            {metrics['delta_nrmse']:.5f}")
    print(f"  increment correlation      {metrics['delta_correlation']:.5f}")
    print(f"  valid time                 {metrics['valid_time_map']:.2f} time units")
    print(f"  free-running NRMSE         {metrics['rollout_nrmse']:.5f}")
    print(f"  rollout rms / truth rms    {metrics['rms_ratio']:.4f}")
    print(f"  Lyapunov time              {metrics['lyapunov_lyapunov_time']:.3f}")
    print(f"  perfect-model predictability "
          f"{metrics['lyapunov_predictability_time']:.2f} time units")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
