"""Resampling, chronological splitting and target encoding for flow-map learning.

The RK4 solver writes snapshots on the grid ``dt = 0.01``.  A *flow map* is the
one-step propagator of the ODE sampled on a coarser grid,

    Phi_dt(x(t)) = x(t + dt),

and this module turns a trajectory record into supervised ``(input, target)``
pairs for that map.  Four regression strategies are supported behind a single
interface (:class:`TargetCodec`), so that the data pipeline, the network and the
metrics are literally identical across them:

``direct``
    Regress the next state itself, ``y = x(t + dt)``.
``delta``
    Regress the increment, ``y = x(t + dt) - x(t)``.
``residual_linear``
    Fit a linear flow map ``A z + b`` by ridge regression on the training split
    and regress whatever the linear model cannot explain.
``residual_linearized``
    Use the exact solution of the linearised system ``x' = -x + F``, namely
    ``x0(dt) = F + (x(t) - F) exp(-dt)``, and regress its residual.

Everything is expressed in *standardised* coordinates, ``z = (x - mu) / sigma``,
with ``mu`` and ``sigma`` estimated on the training split only.  Standardisation
is not cosmetic here: the state amplitude is ``O(1e2)`` while the interesting
structure lives in ``O(1)`` differences, so an unstandardised network would need
either a minuscule learning rate or an enormous output bias.

Chronology is respected everywhere.  The record is split at a single time
instant; the transition that would straddle that instant is *dropped*, so no
target ever leaks information about the test segment into training.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

LOGGER = logging.getLogger("flowmap.data")

#: Regression strategies understood by :func:`build_codec`.
SUPPORTED_MODES: tuple[str, ...] = (
    "direct",
    "delta",
    "residual_linear",
    "residual_linearized",
)


class TargetCodec:
    """Two-way map between physical next states and what the network regresses.

    The network always operates on standardised inputs ``z = (x - mu) / sigma``
    and always emits an array shaped like ``z``.  Subclasses decide what that
    array means: the next state, an increment, or a residual against a baseline
    predictor.

    Attributes:
        mean: training mean ``mu`` — a scalar (stored as a 0-d array) when all
            components are pooled (``standardization="global"``), or a
            per-component vector of shape ``(d,)`` otherwise.
        scale: training standard deviation ``sigma``, same convention as ``mean``.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, dt: float) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.dt = float(dt)

    # -- helpers ---------------------------------------------------------
    def standardize(self, physical: np.ndarray) -> np.ndarray:
        """Map physical states to the standardised space the network sees."""
        return (np.asarray(physical, dtype=np.float64) - self.mean) / self.scale

    def destandardize(self, standardized: np.ndarray) -> np.ndarray:
        """Map standardised states back to physical units."""
        return np.asarray(standardized, dtype=np.float64) * self.scale + self.mean

    # -- interface -------------------------------------------------------
    def encode(self, z_input: np.ndarray, physical_target: np.ndarray) -> np.ndarray:
        """Build the regression target for a batch of standardised inputs."""
        raise NotImplementedError

    def decode(self, z_input: np.ndarray, network_output: np.ndarray) -> np.ndarray:
        """Turn a network output into a physical next state."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Human-readable one-line description used in reports."""
        raise NotImplementedError


class DirectCodec(TargetCodec):
    """Regress the standardised next state ``z(t + dt)`` directly."""

    def encode(self, z_input: np.ndarray, physical_target: np.ndarray) -> np.ndarray:
        return self.standardize(physical_target)

    def decode(self, z_input: np.ndarray, network_output: np.ndarray) -> np.ndarray:
        return self.destandardize(network_output)

    @property
    def description(self) -> str:
        return "direct: network output is the standardised next state z(t+dt)"


class DeltaCodec(TargetCodec):
    """Regress the standardised increment ``z(t + dt) - z(t)``."""

    def encode(self, z_input: np.ndarray, physical_target: np.ndarray) -> np.ndarray:
        return self.standardize(physical_target) - z_input

    def decode(self, z_input: np.ndarray, network_output: np.ndarray) -> np.ndarray:
        return self.destandardize(z_input + network_output)

    @property
    def description(self) -> str:
        return "delta: network output is the standardised increment dz"


class LinearResidualCodec(TargetCodec):
    """Regress the residual of the best-fit affine flow map ``A z + b``.

    ``A`` and ``b`` are obtained by ridge regression on the *training* pairs, so
    the baseline itself never sees the test segment.
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        dt: float,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        super().__init__(mean, scale, dt)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)

    def linear_flow(self, z_input: np.ndarray) -> np.ndarray:
        """Evaluate the affine baseline ``A z + b``."""
        return z_input @ self.weights.T + self.bias

    def encode(self, z_input: np.ndarray, physical_target: np.ndarray) -> np.ndarray:
        return self.standardize(physical_target) - self.linear_flow(z_input)

    def decode(self, z_input: np.ndarray, network_output: np.ndarray) -> np.ndarray:
        return self.destandardize(self.linear_flow(z_input) + network_output)

    @property
    def description(self) -> str:
        return "residual_linear: network output is the residual of a ridge-fit affine flow map"


class LinearizedResidualCodec(TargetCodec):
    """Regress the residual of the analytic solution of ``x' = -x + F``.

    Dropping the nonlinear advection term leaves a scalar linear ODE per grid
    point, solved exactly by ``x0(dt) = F + (x - F) exp(-dt)``.  Because the
    equation is diagonal in physical space there is no fitted parameter at all:
    the baseline is a fixed function of the state and the forcing.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, dt: float, forcing: float) -> None:
        super().__init__(mean, scale, dt)
        self.forcing = float(forcing)

    def linear_flow(self, physical_input: np.ndarray) -> np.ndarray:
        """Evaluate ``F + (x - F) exp(-dt)`` pointwise."""
        return self.forcing + (np.asarray(physical_input) - self.forcing) * np.exp(-self.dt)

    def encode(self, z_input: np.ndarray, physical_target: np.ndarray) -> np.ndarray:
        baseline = self.standardize(self.linear_flow(self.destandardize(z_input)))
        return self.standardize(physical_target) - baseline

    def decode(self, z_input: np.ndarray, network_output: np.ndarray) -> np.ndarray:
        baseline = self.linear_flow(self.destandardize(z_input))
        return baseline + network_output * self.scale

    @property
    def description(self) -> str:
        return "residual_linearized: network output is the residual of x0(dt) = F + (x - F) exp(-dt)"


@dataclass(frozen=True)
class FlowMapSplit:
    """Chronologically split supervised pairs for the flow map.

    Attributes:
        train_input: physical states at the start of each training transition.
        train_target: physical states one step later (never crossing the split).
        test_input: same, for the held-out tail.
        test_target: same, for the held-out tail.
        train_time: physical time of each ``train_input`` row.
        test_time: physical time of each ``test_input`` row.
        mean: training mean — 0-d array (global pooling) or ``(d,)`` vector.
        scale: training standard deviation — same convention as ``mean``.
        standardization: which pooling was used, ``"global"`` or
            ``"per_component"``.
        stride: resampling stride applied to the raw solver output.
        dt: sampling interval of the pairs, ``stride * raw_dt``.
        n_points: number of resampled snapshots before pairing.
        split_index: index (in resampled units) where the test segment starts.
    """

    train_input: np.ndarray
    train_target: np.ndarray
    test_input: np.ndarray
    test_target: np.ndarray
    train_time: np.ndarray
    test_time: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    standardization: str
    stride: int
    dt: float
    n_points: int
    split_index: int

    @property
    def n_dim(self) -> int:
        """State dimension of the system."""
        return int(self.train_input.shape[1])

    def summary_rows(self) -> list[tuple[str, str]]:
        """Key/value rows describing the split, for text reports."""
        return [
            ("state dimension", f"{self.n_dim}"),
            ("resampling stride", f"{self.stride}"),
            ("sampling interval dt", f"{self.dt:.6g}"),
            ("standardization", f"{self.standardization} "
                                f"(mean={float(self.mean):.4f}, sd={float(self.scale):.4f})"),
            ("resampled snapshots", f"{self.n_points}"),
            ("split index", f"{self.split_index}"),
            ("train transitions", f"{self.train_input.shape[0]}"),
            ("test transitions", f"{self.test_input.shape[0]}"),
            ("train time span", f"{self.train_time[0]:.2f} .. {self.train_time[-1]:.2f}"),
            ("test time span", f"{self.test_time[0]:.2f} .. {self.test_time[-1]:.2f}"),
            ("train mean of x", f"{self.train_input.mean():.4f}"),
            ("test mean of x", f"{self.test_input.mean():.4f}"),
            ("train rms of x", f"{np.sqrt((self.train_input ** 2).mean()):.4f}"),
            ("test rms of x", f"{np.sqrt((self.test_input ** 2).mean()):.4f}"),
        ]


def load_snapshots(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a solver CSV with a ``time`` column followed by ``x_0 ... x_{N-1}``.

    Args:
        path: CSV written by :mod:`solve_lorenz96`.

    Returns:
        ``(time, states)`` with shapes ``(n,)`` and ``(n, d)``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file has fewer than two columns or no rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trajectory file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    if len(header) < 2:
        raise ValueError(f"{path} needs a time column and at least one state column")
    data = np.asarray(rows, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"{path} contains no data rows")
    LOGGER.info(
        "loaded %d snapshots x %d components from %s", data.shape[0], data.shape[1] - 1, path.name
    )
    return data[:, 0], data[:, 1:]


def build_split(
    path: Path,
    stride: int = 10,
    train_fraction: float = 2.0 / 3.0,
    burn_in: float = 0.0,
    standardization: str = "global",
) -> FlowMapSplit:
    """Resample a trajectory, split it chronologically and build step pairs.

    The transition that would straddle the split instant is discarded, so every
    training pair ends at or before the split and every test pair begins at or
    after it.

    Args:
        path: solver CSV.
        stride: keep every ``stride``-th raw snapshot.
        train_fraction: fraction of the resampled record used for training.
        burn_in: physical time to discard from the start of the record.
        standardization: how ``mu`` and ``sigma`` are estimated on the training
            split.  ``"global"`` pools all components into one scalar mean and
            standard deviation (the right choice for systems with a permutation
            symmetry such as Lorenz-96: every component shares the same marginal
            law, so pooling turns ~1e3 samples into ~4e4); ``"per_component"``
            estimates each component separately.

    Returns:
        A populated :class:`FlowMapSplit`.

    Raises:
        ValueError: on non-positive ``stride``, a ``train_fraction`` outside
            ``(0, 1)``, a split too short to contain any pair, or an unknown
            ``standardization``.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")
    if standardization not in ("global", "per_component"):
        raise ValueError(
            f"standardization must be 'global' or 'per_component', got {standardization!r}"
        )

    time, states = load_snapshots(path)
    if burn_in > 0.0:
        keep = time >= burn_in
        if keep.sum() < 3:
            raise ValueError(f"burn_in={burn_in} leaves fewer than 3 snapshots")
        time, states = time[keep], states[keep]

    raw_dt = float(np.median(np.diff(time)))
    index = np.arange(0, time.size, stride)
    time = time[index]
    states = states[index]
    n_points = int(time.size)

    n_train_points = int(np.floor(n_points * train_fraction))
    if n_train_points < 2 or n_points - n_train_points < 2:
        raise ValueError(
            f"split of {n_points} points at {train_fraction:.3f} leaves no usable pairs"
        )

    # Pair index i with i + 1.  Training pairs stop one short of the boundary so
    # that no training target lies inside the test segment; the boundary-straddling
    # pair is dropped entirely.
    train_rows = np.arange(0, n_train_points - 1)
    test_rows = np.arange(n_train_points, n_points - 1)

    train_input = states[train_rows]
    train_target = states[train_rows + 1]
    test_input = states[test_rows]
    test_target = states[test_rows + 1]

    if standardization == "global":
        # Pool every component into one scalar: by the cyclic symmetry all
        # components share the same marginal distribution, so the pooled
        # estimates are both less noisy and consistent with the shift
        # augmentation used during training.
        mean = np.asarray(float(train_input.mean()))
        scale = np.asarray(float(train_input.std()))
        if float(scale) <= 1e-12:
            scale = np.asarray(1.0)
    else:
        mean = train_input.mean(axis=0)
        scale = train_input.std(axis=0)
        # A degenerate component (constant in the training split) would blow up
        # the standardisation; fall back to unit scale so it stays invertible.
        scale = np.where(scale > 1e-12, scale, 1.0)

    split = FlowMapSplit(
        train_input=train_input,
        train_target=train_target,
        test_input=test_input,
        test_target=test_target,
        train_time=time[train_rows],
        test_time=time[test_rows],
        mean=mean,
        scale=scale,
        standardization=standardization,
        stride=stride,
        dt=raw_dt * stride,
        n_points=n_points,
        split_index=n_train_points,
    )
    LOGGER.info(
        "split: %d train pairs, %d test pairs, dt=%.4g, boundary at t=%.2f",
        train_input.shape[0],
        test_input.shape[0],
        split.dt,
        time[n_train_points],
    )
    return split


def fit_ridge_flow_map(
    z_input: np.ndarray, z_target: np.ndarray, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of ``z_target ~= z_input @ A.T + b`` with an L2 penalty.

    Args:
        z_input: standardised inputs, shape ``(n, d)``.
        z_target: standardised targets, shape ``(n, d)``.
        ridge: Tikhonov penalty on ``A`` (the bias is unpenalised).

    Returns:
        ``(weights, bias)`` with shapes ``(d, d)`` and ``(d,)``.
    """
    n, d = z_input.shape
    design = np.hstack([z_input, np.ones((n, 1))])
    penalty = ridge * np.eye(d + 1)
    penalty[-1, -1] = 0.0  # do not shrink the intercept
    gram = design.T @ design + penalty
    solution = np.linalg.solve(gram, design.T @ z_target)
    return solution[:d].T, solution[d]


def build_codec(mode: str, split: FlowMapSplit, ridge: float = 1e-6) -> TargetCodec:
    """Instantiate the target codec for one of :data:`SUPPORTED_MODES`.

    Args:
        mode: one of ``direct``, ``delta``, ``residual_linear``,
            ``residual_linearized``.
        split: the split whose training statistics and pairs parametrise the codec.
        ridge: Tikhonov penalty used by ``residual_linear``.

    Returns:
        A ready-to-use :class:`TargetCodec`.

    Raises:
        ValueError: for an unknown mode.
    """
    if mode == "direct":
        return DirectCodec(split.mean, split.scale, split.dt)
    if mode == "delta":
        return DeltaCodec(split.mean, split.scale, split.dt)
    if mode == "residual_linear":
        z_in = (split.train_input - split.mean) / split.scale
        z_out = (split.train_target - split.mean) / split.scale
        weights, bias = fit_ridge_flow_map(z_in, z_out, ridge=ridge)
        LOGGER.info("ridge flow map fitted: ||A||_F = %.4f", float(np.linalg.norm(weights)))
        return LinearResidualCodec(split.mean, split.scale, split.dt, weights, bias)
    if mode == "residual_linearized":
        forcing = float(np.round(split.train_input.mean(axis=1).mean()))
        LOGGER.info("linearised baseline uses forcing F = %.4f", forcing)
        return LinearizedResidualCodec(split.mean, split.scale, split.dt, forcing)
    raise ValueError(f"unknown mode {mode!r}; expected one of {SUPPORTED_MODES}")


def encode_dataset(
    split: FlowMapSplit, codec: TargetCodec
) -> dict[str, np.ndarray]:
    """Standardise inputs and encode targets for both segments.

    Args:
        split: the chronological split.
        codec: target codec, usually from :func:`build_codec`.

    Returns:
        A dict with ``train_x``, ``train_y``, ``test_x``, ``test_y`` in the
        network space, plus the physical inputs/targets for convenience.
    """
    return {
        "train_x": codec.standardize(split.train_input),
        "train_y": codec.encode(codec.standardize(split.train_input), split.train_target),
        "test_x": codec.standardize(split.test_input),
        "test_y": codec.encode(codec.standardize(split.test_input), split.test_target),
    }
