"""Checks for the Fourier mode analysis."""

import unittest

import numpy as np

from analyze_fourier_modes import (
    brute_force_complex_acf,
    complex_acf,
    degrees_of_freedom,
    independent_mode_indices,
    mode_energies,
    spatial_dft,
    verify_identities,
)


class FourierModeTests(unittest.TestCase):
    def test_unitary_dft_preserves_the_norm(self) -> None:
        rng = np.random.default_rng(3)
        states = rng.normal(size=(7, 40))
        modes = spatial_dft(states)
        np.testing.assert_allclose(np.sum(np.abs(modes) ** 2, axis=1),
                                   np.sum(states**2, axis=1), rtol=1e-12)

    def test_real_state_has_conjugate_symmetric_coefficients(self) -> None:
        rng = np.random.default_rng(11)
        states = rng.normal(size=(5, 40))
        modes = spatial_dft(states)
        reflected = modes[:, (40 - np.arange(40)) % 40]
        np.testing.assert_allclose(reflected, np.conjugate(modes), atol=1e-12)

    def test_zero_and_nyquist_modes_are_real(self) -> None:
        rng = np.random.default_rng(5)
        modes = spatial_dft(rng.normal(size=(4, 40)))
        np.testing.assert_allclose(modes[:, 0].imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(modes[:, 20].imag, 0.0, atol=1e-12)

    def test_degrees_of_freedom_sum_to_the_state_dimension(self) -> None:
        keep = independent_mode_indices(40)
        self.assertEqual(int(np.sum(degrees_of_freedom(40, keep))), 40)

    def test_energy_partition_reproduces_half_the_squared_norm(self) -> None:
        rng = np.random.default_rng(13)
        states = rng.normal(size=(9, 40))
        modes_all = spatial_dft(states)
        keep = independent_mode_indices(40)
        energy = mode_energies(modes_all[:, keep], degrees_of_freedom(40, keep))
        np.testing.assert_allclose(np.sum(energy, axis=1), 0.5 * np.sum(states**2, axis=1),
                                   rtol=1e-12)

    def test_complex_acf_matches_direct_evaluation(self) -> None:
        rng = np.random.default_rng(17)
        series = rng.normal(size=400) + 1j * rng.normal(size=400)
        fast = complex_acf(series, 30)
        slow = brute_force_complex_acf(series, 30)
        np.testing.assert_allclose(fast, slow, atol=1e-10)

    def test_acf_of_a_pure_rotation_has_linear_phase(self) -> None:
        # z_t = exp(i * omega * t) gives rho(tau) = (1 - tau/n) exp(i * omega * tau)
        # with the biased estimator, because only n - tau products are available.
        omega = 0.37
        n = 4096
        series = np.exp(1j * omega * np.arange(n))
        rho = complex_acf(series, 12)
        expected = (1.0 - np.arange(13) / n) * np.exp(1j * omega * np.arange(13))
        np.testing.assert_allclose(rho, expected, atol=5e-4)
        # The phase must be linear in tau; unwrap by removing the expected phase.
        residual_phase = np.angle(rho[1:] * np.exp(-1j * omega * np.arange(1, 13)))
        np.testing.assert_allclose(residual_phase, 0.0, atol=5e-3)

    def test_verify_identities_reports_machine_precision_errors(self) -> None:
        rng = np.random.default_rng(23)
        states = rng.normal(size=(6, 40))
        modes_all = spatial_dft(states)
        keep = independent_mode_indices(40)
        energy = mode_energies(modes_all[:, keep], degrees_of_freedom(40, keep))
        report = verify_identities(states, modes_all, keep, energy)
        self.assertLess(report["max_abs_parseval_error"], 1e-9)
        self.assertLess(report["max_abs_conjugate_symmetry_error"], 1e-12)
        self.assertLess(report["max_abs_energy_partition_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
