"""Why does regressing the increment lose to regressing the next state?

The two codecs in :mod:`flowmap_data` have the *same estimand*: since
``z(t+dt) = z(t) + dz`` with ``z(t)`` known at prediction time, a perfect
``delta`` model and a perfect ``direct`` model define the same map.  Any measured
difference is therefore a property of the estimator, not of the target.  This
module measures where that difference appears and how it propagates.

Because ``tau_Lyap = 0.084 < dt = 0.1`` a single step already multiplies an error
by ``exp(1.19) ~ 3.3``, so the one-step NRMSE is really "how much of the
increment is captured" and every trajectory-based metric saturates immediately.
The discriminating evidence is therefore *statistical*: the invariant measure of
the learned map (energy spectrum, variance, mean, autocorrelation), estimated
from a long free-running ensemble with the transient discarded.

Battery
-------
1. one-step error budget, split into bias and dispersion, plus the *attenuation
   slope* ``beta = <pred_dz, true_dz> / <true_dz, true_dz>`` -- the fraction of
   the true increment the map actually produces;
2. the same budget per wavenumber, to see which scales are lost;
3. the Lyapunov exponent of the *learned* map, via Benettin on the map at the
   sampling interval ``dt`` (exactly comparable with the RK4 flow map);
4. the invariant measure of the learned map: transient-corrected ensemble
   rollout -> ``E_k``, pdf, autocorrelation, collapse diagnostics;
5. a controlled probe on the *true* map: ``x -> x + beta (Phi(x) - x)`` for a
   grid of ``beta``.  This reproduces the effect of increment attenuation
   without any network, and separates "attenuation" from "everything else".

Usage::

    python diagnose_codecs.py --runs direct=results/flowmap_v3_lr/lr_direct_... \\
                                     delta=results/flowmap_v3_lr/lr_delta_...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flowmap_data import FlowMapSplit, TargetCodec, build_codec, build_split, fit_ridge_flow_map
from flowmap_model import FlowMapMLP, predict
from solve_lorenz96 import rk4_step

LOGGER = logging.getLogger("flowmap.diagnose")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"
COLOURS = {"direct": "#2166ac", "delta": "#b2182b", "truth": "0.15"}
FORCING = 64.0
RAW_DT = 0.01
#: Size of the tangent-space perturbation for Benettin.  Large enough to survive
#: float64 rounding on states of amplitude ~2e1, small enough to stay linear.
PERTURBATION = 1e-6


# ---------------------------------------------------------------------------
# loading and spectra
# ---------------------------------------------------------------------------
@dataclass
class Run:
    """One trained model together with the codec that decodes it."""

    label: str
    directory: Path
    model: FlowMapMLP
    codec: TargetCodec
    mode: str
    split: FlowMapSplit
    metrics: dict[str, Any]


def load_run(label: str, directory: Path, split: FlowMapSplit) -> Run:
    """Rebuild a trained run from its checkpoint.

    Args:
        label: short name used in figures and tables.
        directory: output directory containing ``flowmap_model_F64.pt``.
        split: the (already built) chronological split.

    Returns:
        A populated :class:`Run`.

    Raises:
        FileNotFoundError: if the checkpoint is missing.
    """
    checkpoint_path = directory / "flowmap_model_F64.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint in {directory}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = str(checkpoint["mode"])
    state_dict = checkpoint["state_dict"]
    hidden_width = int(state_dict["head.weight"].shape[1])
    n_hidden_layers = sum(
        1 for key in state_dict if key.startswith("body.") and key.endswith(".weight")
    )
    model = FlowMapMLP(
        n_dim=int(split.n_dim),
        hidden_width=hidden_width,
        n_hidden_layers=max(1, n_hidden_layers),
        activation="tanh",
    )
    model.load_state_dict(state_dict)
    model.eval()
    # Double precision is not optional here: the Lyapunov estimate differences two
    # trajectories 1e-6 apart, and the state amplitude is O(2e1), so float32
    # rounding (~2e-6 absolute) would erase the perturbation entirely.
    model.double()
    if not np.isclose(float(checkpoint["mean"]), float(split.mean), rtol=1e-6):
        LOGGER.warning("%s: checkpoint mean differs from rebuilt split", label)
    codec = build_codec(mode, split)
    metrics_path = directory / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    LOGGER.info("loaded %-14s mode=%-6s from %s", label, mode, directory.name)
    return Run(label=label, directory=directory, model=model, codec=codec, mode=mode,
               split=split, metrics=metrics)


@torch.no_grad()
def predict64(model: FlowMapMLP, features: np.ndarray) -> np.ndarray:
    """Evaluate the network in double precision.

    :func:`flowmap_model.predict` casts to float32, which is fine for training
    but destroys the tangent-space perturbation used by the Lyapunov estimate.

    Args:
        model: a float64 network.
        features: inputs, shape ``(n, d)``.

    Returns:
        Network outputs as float64.
    """
    model.eval()
    tensor = torch.as_tensor(np.asarray(features, dtype=np.float64))
    return model(tensor).numpy()


def spectrum(field: np.ndarray) -> np.ndarray:
    """Time-averaged energy per zonal wavenumber of a ``(time, x)`` record.

    The discrete Fourier transform is normalised so that ``sum_k E_k`` is the
    spatial variance, and the one-sided modes ``1 <= k < N/2`` are doubled to
    account for their conjugate partners.

    Args:
        field: array of shape ``(n_samples, n_dim)``.

    Returns:
        Array of shape ``(n_dim // 2 + 1,)`` with ``E_0 ... E_{N/2}``.
    """
    coefficient = np.fft.fft(field, axis=1) / np.sqrt(field.shape[1])
    power = np.abs(coefficient) ** 2
    independent = field.shape[1] // 2 + 1
    dof = np.ones(independent)
    dof[1 : field.shape[1] // 2] = 2.0
    return (power[:, :independent] * dof / 2.0).mean(axis=0)


def true_flow_map(states: np.ndarray, forcing: float = FORCING, steps: int = 10) -> np.ndarray:
    """Apply the RK4 flow map over one sampling interval ``dt = steps * 0.01``."""
    out = np.asarray(states, dtype=np.float64).copy()
    for _ in range(steps):
        out = rk4_step(out, RAW_DT, forcing)
    return out


def long_truth_reference(
    n_samples: int = 20000,
    spin_up: int = 2000,
    forcing: float = FORCING,
    cache: Path | None = None,
    seed: int = 7,
) -> np.ndarray:
    """A long independent trajectory, used as the reference invariant measure.

    A 483-sample test segment is *not* enough to locate the peak of ``E_k``: the
    spectral estimate of a single realisation scatters by a factor of order one
    per mode.  Every statistical claim in this module is therefore made against a
    long run of the same system, cached on disk.

    Args:
        n_samples: number of states to keep, sampled at ``dt = 0.1``.
        spin_up: discarded iterations (the initial condition is near ``x_j = F``).
        forcing: the F parameter.
        cache: optional ``.npz`` path; read if present, written otherwise.
        seed: seed of the initial perturbation.

    Returns:
        Array of shape ``(n_samples, 40)``.
    """
    if cache is not None and cache.exists():
        return np.load(cache)["states"]

    rng = np.random.default_rng(seed)
    state = forcing + rng.normal(loc=0.0, scale=0.5, size=40)
    for _ in range(spin_up):
        state = true_flow_map(state, forcing=forcing)
    record = np.empty((n_samples, state.size), dtype=np.float64)
    for index in range(n_samples):
        state = true_flow_map(state, forcing=forcing)
        record[index] = state
    LOGGER.info("generated %d reference states (mean %.3f, rms %.3f)",
                n_samples, record.mean(), np.sqrt((record**2).mean()))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, states=record)
    return record


# ---------------------------------------------------------------------------
# 1-2. one-step error budget
# ---------------------------------------------------------------------------
def one_step_budget(run: Run) -> dict[str, Any]:
    """Decompose the one-step error into bias, dispersion and scale content.

    Args:
        run: loaded run.

    Returns:
        A dict with scalar diagnostics and the per-wavenumber arrays.
    """
    split = run.split
    z_in = run.codec.standardize(split.test_input)
    output = predict64(run.model, z_in)
    predicted_state = run.codec.decode(z_in, output)

    true_dz = split.test_target - split.test_input
    pred_dz = predicted_state - split.test_input

    flat_true = true_dz.ravel()
    flat_pred = pred_dz.ravel()
    # Attenuation slope: how much of the true increment survives, measured in
    # the metric the decoder actually applies.  beta = 1 means calibrated.
    beta = float(flat_pred @ flat_true / (flat_true @ flat_true))
    true_variance = float(np.var(flat_true))
    residual = flat_pred - flat_true

    # Attenuation as a function of how large the increment is: this separates
    # "the map under-predicts the bulk" from "the map under-predicts the extremes".
    magnitude = np.abs(flat_true)
    edges = np.quantile(magnitude, np.linspace(0.0, 1.0, 6))
    beta_by_magnitude = np.empty(5)
    for index in range(5):
        mask = (magnitude >= edges[index]) & (magnitude <= edges[index + 1])
        if mask.sum() < 2:
            beta_by_magnitude[index] = np.nan
            continue
        selected_true = flat_true[mask]
        selected_pred = flat_pred[mask]
        beta_by_magnitude[index] = float(
            selected_pred @ selected_true / (selected_true @ selected_true)
        )

    # Split the increment into the part a *linear* map can carry (the damping
    # -x_j dt) and the nonlinear remainder (the advective transport).  Then ask how
    # much of each the network reproduced: a codec can be equally good/bad overall
    # while getting very different mixes right.
    z_train = run.codec.standardize(split.train_input)
    dz_train = (split.train_target - split.train_input) / float(split.scale)
    z_test = run.codec.standardize(split.test_input)
    dz_test = true_dz / float(split.scale)
    weights, bias = fit_ridge_flow_map(z_train, dz_train)
    truth_nonlinear = dz_test - (z_test @ weights.T + bias)
    predicted_dz_standardized = pred_dz / float(split.scale)
    weights_net, bias_net = fit_ridge_flow_map(z_test, predicted_dz_standardized)
    predicted_nonlinear = predicted_dz_standardized - (z_test @ weights_net.T + bias_net)
    linear_capture = float(
        1.0 - np.linalg.norm(weights_net - weights) / np.linalg.norm(weights)
    )
    nonlinear_r2 = float(
        1.0 - np.mean((predicted_nonlinear - truth_nonlinear) ** 2)
        / np.mean(truth_nonlinear**2)
    )

    # State-space error budget: a constant offset (bias) plus scatter.
    state_error = predicted_state - split.test_target
    bias_vector = state_error.mean(axis=0)
    return {
        "label": run.label,
        "mode": run.mode,
        "beta": beta,
        "increment_r2": float(1.0 - np.mean(residual**2) / true_variance),
        "increment_nrmse": float(
            np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(flat_true**2))
        ),
        "increment_correlation": float(np.corrcoef(flat_pred, flat_true)[0, 1]),
        "state_nrmse": float(
            np.sqrt(np.mean(state_error**2)) / np.sqrt(np.mean(split.test_target**2))
        ),
        "bias_energy_fraction": float(
            np.mean(bias_vector**2) / np.mean(state_error**2)
        ),
        "residual_skew": float(
            float(np.mean(((residual - residual.mean()) / residual.std()) ** 3))
        ),
        "residual_kurtosis": float(
            float(np.mean(((residual - residual.mean()) / residual.std()) ** 4)) - 3.0
        ),
        "true_increment_rms": float(np.sqrt(np.mean(flat_true**2))),
        "state_rms": float(np.sqrt(np.mean(split.test_target**2))),
        "spectrum_true_dz": spectrum(true_dz),
        "spectrum_pred_dz": spectrum(pred_dz),
        "spectrum_error": spectrum(residual.reshape(true_dz.shape)),
        "linear_capture": linear_capture,
        "nonlinear_r2": nonlinear_r2,
        "target_kurtosis": float(
            float(np.mean(((flat_true - flat_true.mean()) / flat_true.std()) ** 4)) - 3.0
        ),
        "beta_by_magnitude": beta_by_magnitude,
        "magnitude_edges": edges,
        "residual": residual,
        "beta_hist": (flat_pred, flat_true),
    }


# ---------------------------------------------------------------------------
# 3. Lyapunov exponent of a map at the sampling interval
# ---------------------------------------------------------------------------
def numerical_jacobian(
    map_fn: Callable[[np.ndarray], np.ndarray], state: np.ndarray, step: float = PERTURBATION
) -> np.ndarray:
    """Central-free finite-difference Jacobian, evaluated in one batched call.

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        state: ``(d,)`` point.
        step: finite-difference increment, in physical units.

    Returns:
        The ``(d, d)`` Jacobian ``d map / d x`` at ``state``.
    """
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    dimension = state.size
    batch = np.vstack([state[None, :], state[None, :] + step * np.eye(dimension)])
    moved = map_fn(batch)
    return ((moved[1:] - moved[0]) / step).T


def tangent_spectrum(
    map_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int = 300,
    dt: float = 0.1,
    step: float = PERTURBATION,
    warm_up: int = 0,
) -> dict[str, Any]:
    """Full Lyapunov spectrum of a map via QR renormalisation of the tangent.

    A single-vector (Benettin) estimate is not safe here.  The learned maps have
    strongly non-normal Jacobians -- at a typical point the singular values span
    ``2.3 ... 0.015`` -- so one vector can sit in a contracting subspace for
    thousands of steps and report a negative exponent for a chaotic map.  Carrying
    the whole orthonormal frame removes the ambiguity and yields the sign of the
    *leading* exponent as well as the number of unstable directions.

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        x0: starting state, assumed to be on the attractor.
        n_steps: number of QR steps.
        dt: time between two applications of the map.
        step: finite-difference increment.
        warm_up: iterations advanced *before* the tangent frame is attached.
            Learned maps can have long chaotic transients that end on a limit
            cycle; without a warm-up the QR average measures the transient and
            reports a positive exponent for a periodic attractor.

    Returns:
        Dict with ``lambda_max``, the full ``exponents`` (descending), the running
        finite-time estimate of the leading exponent, and the count of positive
        exponents.
    """
    state = np.asarray(x0, dtype=np.float64).reshape(-1)
    dimension = state.size
    for _ in range(int(warm_up)):
        state = map_fn(state[None, :])[0]
    frame = np.eye(dimension)
    accumulated = np.zeros(dimension)
    leading = np.empty(n_steps)

    for index in range(n_steps):
        jacobian = numerical_jacobian(map_fn, state, step)
        frame, upper = np.linalg.qr(jacobian @ frame)
        # Fix the sign ambiguity of the QR factorisation so that the diagonal of
        # R is the local stretch in each Gram-Schmidt direction.
        signs = np.sign(np.diag(upper))
        signs[signs == 0.0] = 1.0
        frame = frame * signs
        upper = (upper.T * signs).T
        diagonal = np.maximum(np.abs(np.diag(upper)), 1e-300)
        accumulated += np.log(diagonal)
        leading[index] = np.log(diagonal[0])
        state = map_fn(state[None, :])[0]

    exponents = accumulated / (n_steps * dt)
    return {
        "lambda_max": float(exponents[0]),
        "exponents": exponents,
        "running": np.cumsum(leading) / (np.arange(1, n_steps + 1) * dt),
        "n_unstable": int(np.sum(exponents > 0.0)),
        "lyapunov_time": float(1.0 / exponents[0]) if exponents[0] > 0 else float("inf"),
    }


def recurrence(
    map_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_steps: int = 5000,
    burn_in: int = 1500,
    max_period: int = 3000,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Detect an attracting periodic orbit: does the orbit ever come back exactly?

    A chaotic orbit never repeats to machine precision; a contracting one lands on
    a limit cycle.  This is the qualitative check that the one-step error cannot
    provide, and it is what separates "a slightly worse map" from "a different
    kind of system".

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        x0: starting state.
        n_steps: total iterations.
        burn_in: iterations dropped before looking for a recurrence.
        max_period: longest period searched.
        tolerance: distance below which two states count as identical.

    Returns:
        Dict with ``periodic``, ``period`` and the closest ``distance``.
    """
    state = np.asarray(x0, dtype=np.float64).reshape(1, -1)
    history = np.empty((n_steps, state.shape[1]))
    for index in range(n_steps):
        state = map_fn(state)
        history[index] = state[0]

    tail = history[burn_in:]
    window = min(max_period, tail.shape[0] - 1)
    distances = np.linalg.norm(tail[-1] - tail[-1 - window : -1], axis=1)
    best = int(np.argmin(distances))
    return {
        "periodic": bool(distances[best] < tolerance),
        "period": int(window - best),
        "distance": float(distances[best]),
    }


def jacobian_diagnostics(
    map_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_points: int = 40,
    warm_up: int = 300,
) -> dict[str, float]:
    """Non-normality of the one-step Jacobian along the learned attractor.

    The delta decoder is ``x' = x + g(x)``, so its Jacobian is ``I + dg``: the
    identity is added to whatever the network learned.  Adding a constant to a
    matrix whose eigenvalues are strongly negative produces a *sheared* operator --
    eigenvalues that look expansive while almost all directions contract.  The
    classic measure is the eigenvector condition number

        kappa = max_i ||u_i|| ||v_i|| / |<u_i, v_i>| ,

    which is 1 for a normal matrix and grows with the shear.  A large kappa means
    the eigenvalues say nothing about the actual growth of perturbations, which is
    precisely how a map with ``max Re lambda = 1.66`` can have a negative Lyapunov
    exponent.

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        x0: starting state.
        n_points: number of points sampled along the trajectory.
        warm_up: iterations discarded first.

    Returns:
        Medians of ``sigma_max``, ``rho`` (spectral radius) and ``kappa``.
    """
    state = np.asarray(x0, dtype=np.float64).reshape(1, -1)
    for _ in range(warm_up):
        state = map_fn(state)

    sigma, rho, kappa = [], [], []
    for _ in range(n_points):
        jacobian = numerical_jacobian(map_fn, state[0])
        sigma.append(float(np.linalg.svd(jacobian, compute_uv=False)[0]))
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        rho.append(float(np.max(np.abs(eigenvalues))))
        try:
            inverse = np.linalg.inv(eigenvectors)
        except np.linalg.LinAlgError:
            continue
        worst = 0.0
        for index in range(jacobian.shape[0]):
            right = eigenvectors[:, index]
            left = inverse[index]
            overlap = abs(np.dot(left / np.linalg.norm(left),
                                 right / np.linalg.norm(right)))
            worst = max(worst, 1.0 / max(overlap, 1e-12))
        kappa.append(float(worst))
        state = map_fn(state)

    return {
        "sigma_max": float(np.median(sigma)),
        "rho": float(np.median(rho)),
        "kappa": float(np.median(kappa)),
    }


def variance_decay(
    map_fn: Callable[[np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    n_steps: int = 200,
) -> np.ndarray:
    """Ensemble rms after each iteration, normalised by the initial rms.

    A contracting map shows a straight line on a log axis; a chaotic map
    saturates at its own attractor amplitude.  The *slope* is the mechanism: with
    ``x' = x + beta dz`` an attenuation ``beta < 1`` compounds multiplicatively
    over iterations, which is exactly what a "shrunken increment" predicts.

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        initial_states: ``(m, d)`` states on the true attractor.
        n_steps: iterations to record.

    Returns:
        Array of shape ``(n_steps,)`` holding rms(t) / rms(0).
    """
    states = np.asarray(initial_states, dtype=np.float64).copy()
    initial_rms = float(np.sqrt(np.mean(states**2)))
    curve = np.empty(n_steps)
    for step in range(n_steps):
        states = map_fn(states)
        curve[step] = np.sqrt(np.mean(states**2)) / initial_rms
    return curve


# ---------------------------------------------------------------------------
# 4. invariant measure of a map
# ---------------------------------------------------------------------------
def ensemble_rollout(
    map_fn: Callable[[np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    n_steps: int,
    burn_in: int,
) -> np.ndarray:
    """Iterate ``map_fn`` from several initial conditions, discarding a transient.

    Args:
        map_fn: the map, applied to an ``(m, d)`` batch.
        initial_states: shape ``(m, d)``.
        n_steps: total number of iterations.
        burn_in: iterations dropped before sampling.

    Returns:
        Array of shape ``(m, n_steps - burn_in, d)``.
    """
    states = np.asarray(initial_states, dtype=np.float64).copy()
    kept: list[np.ndarray] = []
    for step in range(n_steps):
        states = map_fn(states)
        if step >= burn_in:
            kept.append(states.copy())
    return np.stack(kept, axis=1)


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of each row of a ``(m, n)`` array, averaged."""
    centred = series - series.mean(axis=1, keepdims=True)
    variance = np.mean(centred**2, axis=1)
    out = np.empty(max_lag)
    for lag in range(max_lag):
        product = centred[:, lag:] * centred[:, : centred.shape[1] - lag]
        out[lag] = float(np.mean(product.mean(axis=1) / variance))
    return out


def invariant_statistics(
    sampled: np.ndarray, reference: np.ndarray, dt: float = 0.1
) -> dict[str, float | np.ndarray]:
    """Statistics of a sampled trajectory against a reference record.

    Args:
        sampled: ``(m, n, d)`` ensemble of states drawn from the learned measure.
        reference: ``(n_ref, d)`` true states.
        dt: sampling interval, used for the autocorrelation time.

    Returns:
        Scalar ratios plus the spectra and autocorrelations.
    """
    flattened = sampled.reshape(-1, sampled.shape[-1])
    learned_spectrum = spectrum(flattened)
    reference_spectrum = spectrum(reference)

    energy = (sampled**2).mean(axis=-1)  # (m, n)
    reference_energy = (reference**2).mean(axis=1)[None, :]
    max_lag = min(300, sampled.shape[1] - 1)
    learned_acf = autocorrelation(energy, max_lag)
    reference_acf = autocorrelation(reference_energy, min(max_lag, reference.shape[0] - 1))

    # Collapse diagnostic: spread over the last 200 steps of a long run.
    tail = sampled[:, -200:, :]
    tail_spread = float(np.sqrt(np.mean((tail - tail.mean(axis=1, keepdims=True)) ** 2)))
    reference_spread = float(np.sqrt(np.mean((reference - reference.mean(axis=0)) ** 2)))

    return {
        "mean_ratio": float(flattened.mean() / reference.mean()),
        "rms_ratio": float(
            np.sqrt((flattened**2).mean()) / np.sqrt((reference**2).mean())
        ),
        "variance_ratio": float(flattened.var() / reference.var()),
        "skew": float(
            float(np.mean(((flattened - flattened.mean()) / flattened.std()) ** 3))
        ),
        "kurtosis": float(
            float(np.mean(((flattened - flattened.mean()) / flattened.std()) ** 4)) - 3.0
        ),
        "spectrum_correlation": float(
            np.corrcoef(np.log(learned_spectrum[1:]), np.log(reference_spectrum[1:]))[0, 1]
        ),
        "spectrum_peak_mode": float(np.argmax(learned_spectrum[1:]) + 1),
        "reference_peak_mode": float(np.argmax(reference_spectrum[1:]) + 1),
        "tail_spread_ratio": float(tail_spread / reference_spread),
        "learned_spectrum": learned_spectrum,
        "reference_spectrum": reference_spectrum,
        "learned_acf": learned_acf,
        "reference_acf": reference_acf,
        "acf_time": float(np.sum(learned_acf > 1.0 / np.e) * dt),
        "reference_acf_time": float(np.sum(reference_acf > 1.0 / np.e) * dt),
    }


# ---------------------------------------------------------------------------
# 5. controlled probe: attenuate the true increment by a factor beta
# ---------------------------------------------------------------------------
def beta_probe(
    initial_states: np.ndarray,
    betas: Iterable[float],
    n_steps: int = 4000,
    burn_in: int = 800,
    reference: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Apply ``x -> x + beta (Phi_true(x) - x)`` and measure the attractor.

    This is the null model for "the network only resolves a fraction ``beta`` of
    the increment".  It contains no approximation error other than the
    attenuation itself, so whatever statistics it breaks are broken *by the
    attenuation*, not by the network.

    Args:
        initial_states: ``(m, d)`` starting states on the attractor.
        betas: attenuation factors to probe; ``1.0`` is the true map.
        n_steps: iterations per probe.
        burn_in: discarded transient.
        reference: true record used to normalise the statistics.

    Returns:
        One dict of statistics per ``beta``.
    """
    rows: list[dict[str, float]] = []
    for beta in betas:
        map_fn = lambda x, b=float(beta): x + b * (true_flow_map(x) - x)
        sampled = ensemble_rollout(map_fn, initial_states, n_steps, burn_in)
        exponents = tangent_spectrum(map_fn, initial_states[0], n_steps=150)
        stats = invariant_statistics(
            sampled, reference if reference is not None else sampled.reshape(-1, sampled.shape[-1])
        )
        rows.append({
            "beta": float(beta),
            "lambda_max": float(exponents["lambda_max"]),
            "rms_ratio": float(stats["rms_ratio"]),
            "mean_ratio": float(stats["mean_ratio"]),
            "spectrum_correlation": float(stats["spectrum_correlation"]),
            "spectrum_peak_mode": float(stats["spectrum_peak_mode"]),
            "acf_time": float(stats["acf_time"]),
        })
        LOGGER.info(
            "probe beta=%.2f  lambda=%+.3f  rms ratio %.3f  peak k=%.0f",
            beta, rows[-1]["lambda_max"], rows[-1]["rms_ratio"],
            rows[-1]["spectrum_peak_mode"],
        )
    return rows


def form_probe(
    initial_states: np.ndarray,
    values: Iterable[float],
    reference: np.ndarray,
    n_steps: int = 4000,
    burn_in: int = 800,
) -> list[dict[str, float]]:
    """Same shrinkage, two map *forms*: does the form itself cost the chaos?

    The previous probe attenuates the increment only.  To separate "attenuation"
    from "writing the correction to the increment", this probe applies the *same*
    kind of shrinkage in both forms:

    ``increment``
        ``x -> x + v (Phi(x) - x)``   (the delta codec's form)
    ``state``
        ``x -> mu + v (Phi(x) - mu)`` (the direct codec's form: the whole state is
        regressed, so shrinkage pulls the prediction towards the mean, not
        towards the previous state)

    For each ``v`` the attractor rms and the leading Lyapunov exponent are
    measured, so the two forms can be compared at *matched amplitude* rather than
    matched ``v``.

    Args:
        initial_states: ``(m, d)`` states on the true attractor.
        values: shrinkage factors to probe.
        reference: long truth record used to normalise the statistics.
        n_steps: iterations per probe.
        burn_in: discarded transient.

    Returns:
        One dict per ``(form, value)``.
    """
    mean_state = float(reference.mean())
    rows: list[dict[str, float]] = []
    for form in ("increment", "state"):
        for value in values:
            if form == "increment":
                map_fn = lambda x, v=float(value): x + v * (true_flow_map(x) - x)
            else:
                map_fn = lambda x, v=float(value): mean_state + v * (true_flow_map(x) - mean_state)
            sampled = ensemble_rollout(map_fn, initial_states, n_steps, burn_in)
            spectrum_result = tangent_spectrum(map_fn, initial_states[0], n_steps=120)
            statistics = invariant_statistics(sampled, reference)
            rows.append({
                "form": form,
                "value": float(value),
                "lambda_max": float(spectrum_result["lambda_max"]),
                "rms_ratio": float(statistics["rms_ratio"]),
                "mean_ratio": float(statistics["mean_ratio"]),
                "spectrum_correlation": float(statistics["spectrum_correlation"]),
            })
            LOGGER.info("form %-9s v=%.2f  lambda %+6.2f  rms %.3f", form, value,
                        rows[-1]["lambda_max"], rows[-1]["rms_ratio"])
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_one_step(budgets: list[dict[str, Any]], path: Path) -> None:
    """Density of predicted vs true increment, the budget table, and the tail check."""
    n_columns = len(budgets) + 1
    fig, axes = plt.subplots(2, n_columns, figsize=(5.2 * n_columns, 8.4),
                             constrained_layout=True, squeeze=False)
    for column, budget in enumerate(budgets):
        predicted, truth = budget["beta_hist"]
        rng = np.random.default_rng(0)
        index = rng.choice(predicted.size, size=min(30000, predicted.size), replace=False)
        ax = axes[0, column]
        ax.hexbin(truth[index], predicted[index], gridsize=110, bins="log", cmap="Blues",
                  mincnt=1, linewidths=0)
        limit = float(np.percentile(np.abs(truth), 99.7))
        ax.plot([-limit, limit], [-limit, limit], color="0.25", lw=1.1, ls=":",
                label=r"$\hat{dz}=dz$ (perfect)")
        ax.plot([-limit, limit], [-limit * budget["beta"], limit * budget["beta"]],
                color="#b2182b", lw=1.6, label=rf"$\beta={budget['beta']:.3f}$")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
               xlabel=r"True increment $dz_j$", ylabel=r"Predicted increment $\hat{dz}_j$",
               title=f"{budget['mode']} codec: one-step increments")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

        ax = axes[1, column]
        ax.axis("off")
        rows = [
            ("attenuation slope beta", f"{budget['beta']:.4f}"),
            ("increment $R^2$", f"{budget['increment_r2']:.4f}"),
            ("increment NRMSE", f"{budget['increment_nrmse']:.4f}"),
            ("state NRMSE", f"{budget['state_nrmse']:.4f}"),
            ("bias share of state MSE", f"{100 * budget['bias_energy_fraction']:.1f} %"),
            ("target excess kurtosis", f"{budget['target_kurtosis']:+.2f}"),
            ("linear part captured", f"{100 * budget['linear_capture']:.1f} %"),
            ("nonlinear part R2", f"{budget['nonlinear_r2']:+.4f}"),
        ]
        table = ax.table(cellText=[[k, v] for k, v in rows], loc="center", cellLoc="left",
                         colWidths=[0.66, 0.30])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.7)
        for cell in table.get_celld().values():
            cell.set_linewidth(0.4)
            cell.set_edgecolor("0.8")

    ax = axes[0, n_columns - 1]
    edges = budgets[0]["magnitude_edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    for budget in budgets:
        ax.plot(centres, budget["beta_by_magnitude"], "o-", ms=5, lw=1.7,
                color=COLOURS.get(budget["mode"], "0.4"),
                label=f"{budget['mode']} (mean {budget['beta']:.3f})")
    ax.set(xlabel="|true increment| (quintile centre)", ylabel=r"Attenuation $\beta$ in the bin",
           title="Where the fit is lost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, n_columns - 1]
    ax.axis("off")
    note = (
        "Reading guide\n\n"
        "Both codecs regress the same conditional\n"
        "mean: dz = z(t+dt) - z(t) is exactly the\n"
        "offset between the two targets, so a\n"
        "perfect direct model and a perfect delta\n"
        "model are the same map.\n\n"
        "The diagonal $\\beta$ line is the fraction of\n"
        "the increment actually produced.\n\n"
        "The right-hand curve splits that deficit by\n"
        "magnitude: a flat curve means the map\n"
        "shrinks uniformly (a scale error); a rising\n"
        "curve means the bulk is under-fitted while\n"
        "the extremes are matched (a capacity error)."
    )
    ax.text(0.0, 1.0, note, va="top", ha="left", fontsize=8.5, color="0.25",
            transform=ax.transAxes)
    fig.suptitle("One-step increments: attenuation, and where it is concentrated", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_wavenumbers(budgets: list[dict[str, Any]], path: Path) -> None:
    """Which spatial scales each codec resolves."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    modes = np.arange(budgets[0]["spectrum_true_dz"].size)
    axes[0].semilogy(modes, budgets[0]["spectrum_true_dz"], "o-", color=COLOURS["truth"],
                     ms=4, lw=1.7, label="true $dz$")
    for budget in budgets:
        axes[0].semilogy(modes, budget["spectrum_pred_dz"], "s--", ms=4, lw=1.4,
                         color=COLOURS.get(budget["mode"], "0.4"),
                         label=f"predicted $dz$ ({budget['mode']})")
    axes[0].set(xlabel="Wavenumber $k$", ylabel="Mode energy $E_k$",
                title="Increment energy per scale")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)

    for budget in budgets:
        ratio = budget["spectrum_error"][1:] / budgets[0]["spectrum_true_dz"][1:]
        axes[1].semilogy(modes[1:], ratio, "o-", ms=4, lw=1.5,
                         color=COLOURS.get(budget["mode"], "0.4"),
                         label=f"{budget['mode']} (mean {ratio.mean():.3f})")
    axes[1].axhline(1.0, color="0.4", lw=1.0, ls=":")
    axes[1].set(xlabel="Wavenumber $k$", ylabel="$E_k(\\hat{dz}-dz)\\,/\\,E_k(dz)$",
                title="Fraction of the increment energy left unexplained")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("The deficit is spread over all scales, largest where the increment is largest",
                 fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_lyapunov(
    curves: dict[str, np.ndarray],
    exponents: dict[str, float],
    decay: dict[str, np.ndarray],
    path: Path,
    dt: float,
) -> None:
    """Finite-time Lyapunov estimate plus the ensemble amplitude decay."""
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), constrained_layout=True)
    for label, curve in curves.items():
        axes[0].plot(np.arange(1, curve.size + 1) * dt, np.clip(curve, -30, 30), lw=1.8,
                     color=COLOURS.get(label, "0.4"), label=label)
    axes[0].axhline(0.0, color="0.4", lw=1.0, ls=":")
    axes[0].set(xlabel="Time $t$", ylabel=r"$\Lambda(t)$ per time unit",
                title="Running largest-Lyapunov estimate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = list(exponents)
    values = [exponents[k] for k in labels]
    bars = axes[1].bar(labels, values,
                       color=["#b2182b" if v < 0 else COLOURS.get(k, "0.4")
                              for k, v in zip(labels, values)],
                       alpha=0.85, width=0.6)
    for bar, value in zip(bars, values):
        axes[1].annotate(f"{value:+.2f}",
                         (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 6 if value >= 0 else -14),
                         ha="center", fontsize=10)
        # On a symlog axis a bar of height 0.01 is invisible; mark the top so
        # "essentially zero" reads as a datum rather than as a missing bar.
        if abs(value) < 1.0:
            axes[1].plot([bar.get_x() + bar.get_width() / 2], [value], marker="o", ms=7,
                         color=bar.get_facecolor(), zorder=5, clip_on=False)
    axes[1].axhline(0.0, color="0.25", lw=1.1)
    # The true exponent is 40x the learned ones, so a linear axis would hide the
    # two numbers that matter.  symlog keeps the sign visible.
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(-3.0, 60.0)
    axes[1].set(ylabel=r"$\lambda_{max}$ per time unit (symlog)",
                title="Leading exponent, per map")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].text(0.02, 0.06, "red: negative (contractive)", transform=axes[1].transAxes,
                 fontsize=8, color="#b2182b")

    for label, curve in decay.items():
        axes[2].plot(np.arange(1, curve.size + 1) * dt, curve, lw=1.7,
                     color=COLOURS.get(label, "0.4"), label=label)
    axes[2].axhline(1.0, color="0.4", lw=0.9, ls=":")
    axes[2].set(xlabel="Time $t$", ylabel="Ensemble rms / initial rms",
                title="Amplitude decay under iteration")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=8)
    fig.suptitle("Does the learned map still generate a chaotic attractor?", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_spectrum_convergence(
    long_reference: np.ndarray, short_reference: np.ndarray, path: Path
) -> None:
    """How many samples does it take to locate the peak of ``E_k``?

    The v2 notes compared a rollout spectrum against a 483-sample hold-out and
    concluded that the truth peaks at ``k = 11`` while the model piles up energy
    at ``k = 8``.  Both numbers move when the reference is estimated properly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), constrained_layout=True)
    modes = np.arange(long_reference.shape[1] // 2 + 1)
    sizes = [241, 483, 1451, 4000, long_reference.shape[0]]
    for size in sizes:
        block = long_reference[: min(size, long_reference.shape[0])]
        estimate = spectrum(block)
        peak = int(np.argmax(estimate[1:]) + 1)
        axes[0].semilogy(modes, estimate, lw=1.4, alpha=0.9,
                         label=f"n={block.shape[0]} (peak k={peak})")
    axes[0].set(xlabel="Wavenumber $k$", ylabel="Mode energy $E_k$",
                title="Spectrum estimate vs sample size (same long run)")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)

    full = spectrum(long_reference)
    peaks, correlations = [], []
    for size in sizes:
        estimate = spectrum(long_reference[: min(size, long_reference.shape[0])])
        peaks.append(int(np.argmax(estimate[1:]) + 1))
        correlations.append(float(np.corrcoef(np.log(estimate[1:]), np.log(full[1:]))[0, 1]))
    positions = np.arange(len(sizes))
    axes[1].plot(positions, correlations, "o-", lw=1.7, color="#2166ac")
    for position, peak, size in zip(positions, peaks, sizes):
        axes[1].annotate(f"k={peak}", (position, correlations[position]),
                         textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"n={min(s, long_reference.shape[0])}" for s in sizes], fontsize=9)
    axes[1].set(ylabel="Correlation of $\\log E_k$ with the long reference",
                title="Convergence of the spectral estimate")
    axes[1].grid(alpha=0.25)
    test_peak = int(np.argmax(spectrum(short_reference)[1:]) + 1)
    axes[1].plot([], [], " ",
                 label=f"held-out test segment (n=483) peaks at k={test_peak}")
    axes[1].legend(fontsize=8)
    fig.suptitle("The reference spectrum itself is noisy at n = 483", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_invariant(
    runs: list[Run], samples: dict[str, np.ndarray], reference: np.ndarray,
    stats: dict[str, dict[str, Any]], dt: float, path: Path,
) -> None:
    """Spectrum, pdf, autocorrelation and amplitude envelope of the learned measures."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    modes = np.arange(stats["truth"]["reference_spectrum"].size)
    axes[0, 0].semilogy(modes, stats["truth"]["reference_spectrum"], "o-",
                        color=COLOURS["truth"], ms=4, lw=1.9, label="truth")
    for label in samples:
        axes[0, 0].semilogy(modes, stats[label]["learned_spectrum"], "s--", ms=4, lw=1.5,
                            color=COLOURS.get(label, "0.4"),
                            label=f"{label} (r = {stats[label]['spectrum_correlation']:.2f})")
    axes[0, 0].set(xlabel="Wavenumber $k$", ylabel="Mode energy $E_k$",
                   title="Invariant energy spectrum")
    axes[0, 0].grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)

    bins = np.linspace(-30.0, 70.0, 130)
    axes[0, 1].hist(reference.ravel(), bins=bins, density=True, histtype="step", lw=1.9,
                    color=COLOURS["truth"], label="truth")
    for label, sampled in samples.items():
        axes[0, 1].hist(sampled.reshape(-1, sampled.shape[-1]).ravel(), bins=bins,
                        density=True, histtype="step", lw=1.5,
                        color=COLOURS.get(label, "0.4"), label=label)
    axes[0, 1].set(xlabel="$x_j$", ylabel="Density", title="Marginal distribution of $x_j$")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    truth_acf = stats["truth"]["reference_acf"]
    axes[1, 0].plot(np.arange(truth_acf.size) * dt, truth_acf, lw=1.9,
                    color=COLOURS["truth"], label="truth")
    for label in samples:
        acf = stats[label]["learned_acf"]
        axes[1, 0].plot(np.arange(acf.size) * dt, acf, lw=1.5, color=COLOURS.get(label, "0.4"),
                        label=f"{label} (1/e at {stats[label]['acf_time']:.2f})")
    axes[1, 0].axhline(1.0 / np.e, color="0.5", lw=0.9, ls=":")
    axes[1, 0].set(xlabel="Lag", ylabel="Autocorrelation of $\\langle x^2\\rangle$",
                   title="How fast the learned system forgets")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    window = 60
    for label, sampled in samples.items():
        energy = (sampled**2).mean(axis=-1)
        kernel = np.ones(window) / window
        envelope = np.sqrt(np.convolve(energy.mean(axis=0), kernel, mode="valid"))
        axes[1, 1].plot(np.arange(envelope.size) * dt, envelope, lw=0.8,
                        color=COLOURS.get(label, "0.4"), alpha=0.55,
                        label=f"{label} (rms ratio {stats[label]['rms_ratio']:.2f})")
    reference_energy = (reference**2).mean(axis=1)
    smoothed = np.sqrt(np.convolve(reference_energy, np.ones(window) / window, mode="valid"))
    axes[1, 1].plot(np.arange(smoothed.size) * dt, smoothed, lw=1.6,
                    color=COLOURS["truth"], alpha=0.95, label="truth")
    axes[1, 1].set(xlabel="Time", ylabel="Running rms of $x$",
                   title="Amplitude envelope after the transient")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)
    fig.suptitle("Invariant measure of the learned map (long free ensemble, transient dropped)",
                 fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_beta_probe(probe: list[dict[str, float]], budgets: list[dict[str, Any]],
                      path: Path) -> None:
    """The null-model experiment: attenuate the *true* increment, watch statistics die."""
    betas = np.array([row["beta"] for row in probe])
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)

    curves = [
        ("rms_ratio", "Attractor rms ratio", "Amplitude"),
        ("spectrum_correlation", "Spectrum correlation", "Spectral shape"),
        ("lambda_max", r"$\lambda_{max}$ per time unit", "Chaos"),
    ]
    for ax, (key, label, title) in zip(axes, curves):
        ax.plot(betas, [row[key] for row in probe], "o-", color="#2166ac", lw=1.9, ms=5,
                label="$x \\mapsto x + \\beta(\\Phi_{true}(x) - x)$")
        for budget in budgets:
            ax.axvline(budget["beta"], color=COLOURS.get(budget["mode"], "0.4"), lw=1.3,
                       ls="-.")
        ax.set(xlabel="Attenuation $\\beta$ applied to the true increment",
               ylabel=label, title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[2].axhline(0.0, color="0.25", lw=1.1)

    handles = [plt.Line2D([], [], color=COLOURS.get(b["mode"], "0.4"), ls="-.", lw=1.3,
                          label=f"{b['mode']}: $\\beta$={b['beta']:.3f}") for b in budgets]
    axes[0].legend(handles + axes[0].get_legend_handles_labels()[0],
                   [h.get_label() for h in handles]
                   + axes[0].get_legend_handles_labels()[1], fontsize=7, loc="lower right")
    fig.suptitle("Null model on the true map: what pure increment attenuation does to the statistics",
                 fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def figure_form_probe(
    probe: list[dict[str, float]],
    stats: dict[str, dict[str, Any]],
    exponents: dict[str, float],
    path: Path,
) -> None:
    """Matched-amplitude comparison of the two map forms.

    Horizontal axis is the *achieved* attractor amplitude, so the two curves are
    compared at the same rms rather than at the same shrinkage parameter.  If the
    increment form drops below the state form, writing the correction on top of
    the previous state is itself the problem -- independent of any network.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    styles = {"increment": ("#b2182b", "increment form  $x + v(\\Phi(x)-x)$"),
              "state": ("#2166ac", "state form  $\\mu + v(\\Phi(x)-\\mu)$")}
    for form, (colour, label) in styles.items():
        subset = sorted([row for row in probe if row["form"] == form],
                        key=lambda row: row["rms_ratio"])
        if not subset:
            continue
        axes[0].plot([row["rms_ratio"] for row in subset],
                     [row["lambda_max"] for row in subset], "-o", ms=4, lw=1.9,
                     color=colour, label=label)
        axes[1].plot([row["rms_ratio"] for row in subset],
                     [row["spectrum_correlation"] for row in subset], "-o", ms=4, lw=1.9,
                     color=colour, label=label)

    for label, table in stats.items():
        if label == "truth" or label not in exponents:
            continue
        colour = COLOURS.get(label, "0.3")
        axes[0].scatter([table["rms_ratio"]], [exponents[label]], s=90, marker="*",
                        color=colour, zorder=5, edgecolors="white", linewidths=0.6)
        axes[0].annotate(label, (table["rms_ratio"], exponents[label]),
                         textcoords="offset points", xytext=(9, 6), fontsize=9, color=colour)
        axes[1].scatter([table["rms_ratio"]], [table["spectrum_correlation"]], s=90,
                        marker="*", color=colour, zorder=5, edgecolors="white", linewidths=0.6)
        axes[1].annotate(label, (table["rms_ratio"], table["spectrum_correlation"]),
                         textcoords="offset points", xytext=(9, 6), fontsize=9, color=colour)

    axes[0].axhline(0.0, color="0.3", lw=1.0)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set(xlabel="Attractor rms / truth rms",
                ylabel=r"$\lambda_{max}$ per time unit (symlog)",
                title="Both shrinkage forms keep chaos; the networks do not")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set(xlabel="Attractor rms / truth rms", ylabel="Spectrum correlation",
                title="Spectral shape at matched amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Form probe: the same shrinkage applied in both parameterisations "
                 "(stars: the trained networks, both far below the null curves)", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("wrote %s", path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_tables(
    budgets: list[dict[str, Any]],
    exponents: dict[str, float],
    dynamics: dict[str, Any],
    stats: dict[str, dict[str, Any]],
    probe: list[dict[str, float]],
    output: Path,
) -> None:
    """Write the three numeric tables and the key/value report."""
    with (output / "one_step_budget.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ["label", "mode", "beta", "increment_r2", "increment_nrmse",
                "increment_correlation", "state_nrmse", "bias_energy_fraction",
                "residual_skew", "residual_kurtosis", "true_increment_rms", "state_rms"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(budgets)

    with (output / "mode_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "true_dz"] + [f"pred_dz_{b['mode']}" for b in budgets]
                         + [f"error_{b['mode']}" for b in budgets])
        for k in range(budgets[0]["spectrum_true_dz"].size):
            writer.writerow(
                [k, budgets[0]["spectrum_true_dz"][k]]
                + [b["spectrum_pred_dz"][k] for b in budgets]
                + [b["spectrum_error"][k] for b in budgets]
            )

    with (output / "invariant_measure.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ["mean_ratio", "rms_ratio", "variance_ratio", "skew", "kurtosis",
                "spectrum_correlation", "spectrum_peak_mode", "tail_spread_ratio",
                "acf_time"]
        writer = csv.writer(handle)
        writer.writerow(["map"] + keys)
        for label, table in stats.items():
            writer.writerow([label] + [f"{table[key]:.6g}" for key in keys])

    with (output / "attenuation_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = list(probe[0])
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(probe)

    rows: list[tuple[str, str]] = []
    rows.append(("--- one-step error budget ---", ""))
    for budget in budgets:
        rows.append((f"{budget['mode']}: attenuation beta", f"{budget['beta']:.4f}"))
        rows.append((f"{budget['mode']}: increment R2", f"{budget['increment_r2']:.4f}"))
        rows.append((f"{budget['mode']}: state NRMSE", f"{budget['state_nrmse']:.4f}"))
        rows.append((f"{budget['mode']}: bias share of MSE",
                     f"{100 * budget['bias_energy_fraction']:.1f} %"))
    rows.append(("", ""))
    rows.append(("--- Lyapunov spectrum at dt=0.1 (QR) ---", ""))
    for label, value in exponents.items():
        table = dynamics.get(label, {})
        extra = ""
        if table:
            extra = (f", {table['n_unstable']} unstable dirs, "
                     f"{'limit cycle p=%d' % table['period'] if table['periodic'] else 'aperiodic'}")
        rows.append((f"{label}: lambda_max", f"{value:+.3f} per time unit{extra}"))
        if table:
            rows.append((f"{label}: jacobian sigma_max / rho / kappa",
                         f"{table['sigma_max']:.2f} / {table['rho']:.2f} / {table['kappa']:.1f}"))
    rows.append(("", ""))
    rows.append(("--- invariant measure (long free ensemble) ---", ""))
    for label, table in stats.items():
        rows.append((f"{label}: rms ratio", f"{table['rms_ratio']:.3f}"))
        rows.append((f"{label}: mean ratio", f"{table['mean_ratio']:.3f}"))
        rows.append((f"{label}: spectrum correlation", f"{table['spectrum_correlation']:.3f}"))
        rows.append((f"{label}: spectrum peak mode", f"{table['spectrum_peak_mode']:.0f}"))
        rows.append((f"{label}: 1/e autocorrelation time", f"{table['acf_time']:.2f}"))
        rows.append((f"{label}: tail spread / truth", f"{table['tail_spread_ratio']:.3f}"))
        rows.append(("", ""))
    rows.append(("--- attenuation probe on the true map ---", ""))
    for row in probe:
        rows.append((f"beta={row['beta']:.2f}",
                     f"lambda {row['lambda_max']:+.2f}, rms {row['rms_ratio']:.3f}, "
                     f"spec r {row['spectrum_correlation']:.3f}, peak {row['spectrum_peak_mode']:.0f}"))
    width = max(len(key) for key, _ in rows) + 2
    lines = ["Codec diagnosis: direct vs delta target encoding, Lorenz-96 F=64", "=" * 78]
    lines += [f"{key:<{width}} {value}" for key, value in rows]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", output / "summary.txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=DIR",
                        help="one or more label=directory pairs")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "flowmap_v3_codec"))
    parser.add_argument("--ensemble", type=int, default=16, help="number of initial conditions")
    parser.add_argument("--steps", type=int, default=4000, help="iterations per member")
    parser.add_argument("--burn", type=int, default=800, help="discarded transient")
    parser.add_argument("--probe-steps", type=int, default=4000)
    parser.add_argument("--warm-up", type=int, default=2000,
                        help="iterations discarded before attaching the tangent frame")
    parser.add_argument("--betas", default="1.0,0.95,0.90,0.85,0.80,0.70,0.60,0.50,0.40")
    parser.add_argument("--form-values", default="1.0,0.92,0.84,0.76,0.68,0.60,0.52")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the whole diagnostic battery and write figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    split = build_split(DATA, stride=10, train_fraction=2.0 / 3.0, burn_in=5.0,
                        standardization="global")
    runs: list[Run] = []
    for item in args.runs:
        label, directory = item.split("=", 1)
        resolved = Path(directory)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        runs.append(load_run(label, resolved, split))

    budgets = [one_step_budget(run) for run in runs]
    for budget in budgets:
        LOGGER.info("%-8s beta=%.4f  incr R2=%.4f  state NRMSE=%.4f",
                    budget["mode"], budget["beta"], budget["increment_r2"],
                    budget["state_nrmse"])

    # -- 3. learned dynamics -------------------------------------------------
    start = split.test_input[0]
    exponents: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    decay: dict[str, np.ndarray] = {}
    dynamics: dict[str, Any] = {}
    truth_map = lambda x: true_flow_map(x)
    truth_result = tangent_spectrum(truth_map, start, n_steps=500)
    exponents["truth"] = truth_result["lambda_max"]
    curves["truth"] = truth_result["running"]
    dynamics["truth"] = {**recurrence(truth_map, start, n_steps=6000, max_period=2000),
                         "n_unstable": truth_result["n_unstable"],
                         **jacobian_diagnostics(truth_map, start, n_points=20)}
    learned_maps: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for run in runs:
        map_fn = (lambda x, r=run: r.codec.decode(r.codec.standardize(x),
                                                  predict64(r.model, r.codec.standardize(x))))
        learned_maps[run.label] = map_fn
        result = tangent_spectrum(map_fn, start, n_steps=200, warm_up=args.warm_up)
        exponents[run.label] = result["lambda_max"]
        curves[run.label] = result["running"]
        cycle = recurrence(map_fn, start, n_steps=5000 + args.warm_up, burn_in=args.warm_up,
                           max_period=3000)
        LOGGER.info("%-8s lambda_max = %+.3f (%d unstable)  periodic=%s (p=%d)",
                    run.label, result["lambda_max"], result["n_unstable"],
                    cycle["periodic"], cycle["period"])
        dynamics[run.label] = {**cycle, "n_unstable": result["n_unstable"],
                               "exponents": result["exponents"],
                               **jacobian_diagnostics(map_fn, start)}

    # -- 4. invariant measure ------------------------------------------------
    index = np.linspace(0, split.test_input.shape[0] - 1, num=args.ensemble).astype(int)
    initial = split.test_input[index]
    reference = long_truth_reference(cache=output / "truth_reference.npz")
    LOGGER.info("reference: %d samples, rms %.3f (test segment rms %.3f)",
                reference.shape[0], np.sqrt((reference**2).mean()),
                np.sqrt((split.test_target**2).mean()))

    for label, map_fn in learned_maps.items():
        decay[label] = variance_decay(map_fn, initial, n_steps=200)
    decay["truth"] = variance_decay(truth_map, initial, n_steps=200)

    samples: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, Any]] = {}
    stats["truth"] = invariant_statistics(reference[None, :, :], reference)
    for run in runs:
        sampled = ensemble_rollout(learned_maps[run.label], initial, args.steps, args.burn)
        if not np.all(np.isfinite(sampled)):
            LOGGER.error("%s diverged during the free ensemble", run.label)
        samples[run.label] = sampled
        stats[run.label] = invariant_statistics(sampled, reference)
        LOGGER.info("%-8s rms ratio %.3f  spectrum r %.3f  peak %2.0f  acf %.2f",
                    run.label, stats[run.label]["rms_ratio"],
                    stats[run.label]["spectrum_correlation"],
                    stats[run.label]["spectrum_peak_mode"], stats[run.label]["acf_time"])

    # -- 5. attenuation probe ------------------------------------------------
    betas = [float(value) for value in args.betas.split(",")]
    probe = beta_probe(initial, betas, n_steps=args.probe_steps, burn_in=args.burn,
                       reference=reference)
    forms = [float(value) for value in args.form_values.split(",")]
    form_rows = form_probe(initial, forms, reference, n_steps=args.probe_steps,
                           burn_in=args.burn)

    # -- outputs -------------------------------------------------------------
    write_tables(budgets, exponents, dynamics, stats, probe, output)
    figure_one_step(budgets, output / "fig01_one_step_budget.png")
    figure_wavenumbers(budgets, output / "fig02_wavenumber_error.png")
    figure_lyapunov(curves, exponents, decay, output / "fig03_learned_lyapunov.png", split.dt)
    figure_invariant(runs, samples, reference, stats, split.dt,
                     output / "fig04_invariant_measure.png")
    figure_beta_probe(probe, budgets, output / "fig05_attenuation_probe.png")
    figure_spectrum_convergence(reference, split.test_target,
                                output / "fig06_spectrum_convergence.png")
    figure_form_probe(form_rows, stats, exponents, output / "fig08_form_probe.png")
    with (output / "form_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(form_rows[0]))
        writer.writeheader()
        writer.writerows(form_rows)

    scalars = {
        "budgets": [{k: v for k, v in b.items()
                     if isinstance(v, (int, float, str))} for b in budgets],
        "lyapunov": exponents,
        "cycle": {label: {k: v for k, v in table.items() if isinstance(v, (int, float, bool))}
                  for label, table in dynamics.items()},
        "invariant": {label: {k: float(v) for k, v in table.items()
                              if isinstance(v, (int, float))} for label, table in stats.items()},
        "probe": probe,
        "form_probe": form_rows,
        "decay_after_200_steps": {label: float(curve[-1]) for label, curve in decay.items()},
        "dt": split.dt,
        "tau_lyapunov": 1.0 / exponents["truth"] if exponents["truth"] > 0 else None,
        "reference_samples": int(reference.shape[0]),
    }
    (output / "diagnosis.json").write_text(
        json.dumps(scalars, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("done; artefacts in %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

