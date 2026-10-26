"""Unit tests for the flow-map pipeline.

The interesting ones are :meth:`TestAutodiff.test_gradients_match_finite_differences`,
which checks the backpropagation actually used for training against a brute-force
central difference of the loss, and
:meth:`TestSymmetry.test_shift_augmentation_is_a_symmetry`, which verifies that
the augmentation really is a symmetry of the governing equations rather than an
arbitrary relabelling.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from flowmap_data import (
    SUPPORTED_MODES,
    build_codec,
    build_split,
    encode_dataset,
    fit_ridge_flow_map,
)
from flowmap_model import FlowMapMLP, train_flow_map
from solve_lorenz96 import rk4_step
from train_flowmap import (
    delta_metrics,
    largest_lyapunov,
    normalized_rmse,
    shift_augment,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"
CACHE: dict[str, object] = {}


def get_split():
    """Build the split once and reuse it across tests."""
    if "split" not in CACHE:
        CACHE["split"] = build_split(DATA_PATH, stride=10, train_fraction=2 / 3, burn_in=5.0)
    return CACHE["split"]


class TestSplit(unittest.TestCase):
    """Chronology of the train/test boundary."""

    def test_shapes_are_consistent(self) -> None:
        split = get_split()
        self.assertEqual(split.train_input.shape, split.train_target.shape)
        self.assertEqual(split.test_input.shape, split.test_target.shape)
        self.assertEqual(split.train_input.shape[1], 40)

    def test_no_transition_crosses_the_boundary(self) -> None:
        """The last training target must not be later than the first test input."""
        split = get_split()
        last_train_target_time = split.train_time[-1] + split.dt
        self.assertLessEqual(last_train_target_time, split.test_time[0] + 1e-9)

    def test_split_follows_the_requested_fraction(self) -> None:
        split = get_split()
        self.assertAlmostEqual(split.train_input.shape[0] / split.n_points, 2 / 3, places=2)

    def test_sampling_interval_is_the_stride_times_raw_dt(self) -> None:
        split = get_split()
        self.assertAlmostEqual(split.dt, 0.1, places=9)


class TestCodec(unittest.TestCase):
    """Encode/decode round trips for every regression strategy."""

    def test_round_trip_for_every_mode(self) -> None:
        split = get_split()
        for mode in SUPPORTED_MODES:
            with self.subTest(mode=mode):
                codec = build_codec(mode, split)
                z_input = codec.standardize(split.test_input[:8])
                encoded = codec.encode(z_input, split.test_target[:8])
                recovered = codec.decode(z_input, encoded)
                np.testing.assert_allclose(recovered, split.test_target[:8], atol=1e-8)

    def test_linear_codec_matches_the_ridge_fit(self) -> None:
        """The affine baseline must reproduce the ridge solution exactly."""
        split = get_split()
        codec = build_codec("residual_linear", split)
        z_in = codec.standardize(split.train_input)
        z_out = codec.standardize(split.train_target)
        weights, bias = fit_ridge_flow_map(z_in, z_out)
        np.testing.assert_allclose(codec.weights, weights, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(codec.bias, bias, rtol=1e-10, atol=1e-12)

    def test_linearized_codec_decays_towards_the_forcing(self) -> None:
        """x0(dt) = F + (x - F) exp(-dt) contracts monotonically toward F."""
        split = get_split()
        codec = build_codec("residual_linearized", split)
        state = np.full((1, split.n_dim), 100.0)
        levels = [float(state.mean())]
        for _ in range(6):
            state = codec.linear_flow(state)
            levels.append(float(state.mean()))
        self.assertTrue(np.all(np.diff(levels) < 0.0))
        self.assertGreater(levels[0], 90.0)
        # after n steps the excess over F is exp(-n*dt) of the initial excess
        forcing = codec.forcing
        expected = forcing + (100.0 - forcing) * np.exp(-6 * codec.dt)
        self.assertAlmostEqual(levels[-1], expected, places=8)


class TestModel(unittest.TestCase):
    """Architecture shape and capacity."""

    def test_parameter_count(self) -> None:
        model = FlowMapMLP(40, hidden_width=80, n_hidden_layers=2)
        # 40*80+80 + 80*80+80 + 80*40+40
        self.assertEqual(model.n_parameters, 3280 + 6480 + 3240)
        self.assertEqual(model.n_parameters, 13000)

    def test_forward_shape(self) -> None:
        model = FlowMapMLP(40, 80, 2)
        output = model(torch.zeros(7, 40))
        self.assertEqual(tuple(output.shape), (7, 40))


class TestAutodiff(unittest.TestCase):
    """Backpropagation against a brute-force central difference."""

    def test_gradients_match_finite_differences(self) -> None:
        """The analytic gradient must agree with a numerical one to ~1e-6."""
        torch.manual_seed(0)
        model = FlowMapMLP(6, hidden_width=8, n_hidden_layers=2).double()
        inputs = torch.randn(5, 6, dtype=torch.float64)
        targets = torch.randn(5, 6, dtype=torch.float64)

        def loss_value() -> float:
            return float(((model(inputs) - targets) ** 2).mean())

        model.zero_grad()
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()

        # perturb a handful of parameters by +/- eps and compare the slope
        eps = 1e-6
        max_relative_error = 0.0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                analytic = parameter.grad.clone()
                flat = parameter.view(-1)
                for index in range(min(6, flat.numel())):
                    original = float(flat[index])
                    flat[index] = original + eps
                    plus = loss_value()
                    flat[index] = original - eps
                    minus = loss_value()
                    flat[index] = original
                    numeric = (plus - minus) / (2 * eps)
                    scale = max(abs(numeric), abs(float(analytic.view(-1)[index])), 1e-8)
                    error = abs(numeric - float(analytic.view(-1)[index])) / scale
                    max_relative_error = max(max_relative_error, error)
        self.assertLess(max_relative_error, 1e-5, msg=f"max relative error {max_relative_error:.2e}")

    def test_training_reduces_the_loss(self) -> None:
        """A few hundred epochs on 2-D data must beat the initial loss."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 2))
        y = np.tanh(x) + 0.1 * x

        model = FlowMapMLP(2, hidden_width=16, n_hidden_layers=2)
        history = train_flow_map(model, x, y, x, y, epochs=200, batch_size=64, log_every=0)
        self.assertLess(history.train_loss[-1], history.train_loss[0] * 0.5)


class TestSymmetry(unittest.TestCase):
    """The cyclic invariance used for augmentation."""

    def test_augmentation_preserves_the_pairing(self) -> None:
        split = get_split()
        codec = build_codec("direct", split)
        data = encode_dataset(split, codec)
        inputs, targets = shift_augment(data["train_x"], data["train_y"], 4)
        n = data["train_x"].shape[0]
        self.assertEqual(inputs.shape[0], 4 * n)
        np.testing.assert_allclose(inputs[n : 2 * n], np.roll(data["train_x"], 1, axis=1))
        np.testing.assert_allclose(targets[n : 2 * n], np.roll(data["train_y"], 1, axis=1))

    def test_shift_augmentation_is_a_symmetry(self) -> None:
        """Rolling the state must commute with the true flow (up to RK4 error)."""
        split = get_split()
        state = split.test_input[0]
        offset = 7
        dt, forcing = 0.01, 64.0

        # advance both for one coarse step using fine RK4 substeps
        def advance(x: np.ndarray) -> np.ndarray:
            for _ in range(10):
                x = rk4_step(x, dt, forcing)
            return x

        rolled_then_advanced = advance(np.roll(state, offset))
        advanced_then_rolled = np.roll(advance(state), offset)
        mismatch = np.max(np.abs(rolled_then_advanced - advanced_then_rolled))
        scale = float(np.max(np.abs(advanced_then_rolled)))
        self.assertLess(mismatch / scale, 1e-12)


class TestMetrics(unittest.TestCase):
    """Metric definitions used in the report."""

    def test_normalized_rmse_is_scale_invariant(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(50, 4))
        prediction = truth + 0.1 * rng.normal(size=(50, 4))
        self.assertAlmostEqual(
            normalized_rmse(prediction, truth), normalized_rmse(3.0 * prediction, 3.0 * truth)
        )

    def test_delta_metrics_perfect_prediction(self) -> None:
        rng = np.random.default_rng(0)
        initial = rng.normal(size=(30, 5))
        target = initial + 0.2 * rng.normal(size=(30, 5))
        metrics = delta_metrics(target, target, initial)
        self.assertAlmostEqual(metrics["delta_nrmse"], 0.0, places=12)

    def test_lyapunov_exponent_is_positive_and_physical(self) -> None:
        """F=64 Lorenz-96 is strongly turbulent: lambda_max is of order 10.

        The same estimator returns 1.67 for the textbook F=8 configuration, which
        matches the literature, so the magnitude here is a property of the system
        rather than of the integrator (the estimate is stable across
        ``dt = 0.01 ... 0.001``).
        """
        split = get_split()
        estimate = largest_lyapunov(split.test_input[0], forcing=64.0, dt=0.01, t_total=40.0)
        self.assertGreater(estimate["lambda_max"], 5.0)
        self.assertLess(estimate["lambda_max"], 20.0)
        self.assertAlmostEqual(
            estimate["lyapunov_time"], 1.0 / estimate["lambda_max"], places=9
        )
        # the sampling interval is coarser than one Lyapunov time
        self.assertGreater(split.dt, estimate["lyapunov_time"])


if __name__ == "__main__":
    unittest.main()
