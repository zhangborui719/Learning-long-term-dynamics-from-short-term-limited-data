"""Checks for the autocorrelation and effective-sample-size analysis."""

import unittest

import numpy as np

from analyze_correlations import acf_fft, geyer_initial_positive_sequence, ljung_box


class AnalysisTests(unittest.TestCase):
    def test_constant_series_has_unit_acf(self) -> None:
        np.testing.assert_allclose(acf_fft(np.ones(20), 5), np.ones(6))

    def test_white_noise_has_approximately_one_sample_per_step(self) -> None:
        rng = np.random.default_rng(7)
        acf = acf_fft(rng.normal(size=10000), 100)
        tau, cutoff = geyer_initial_positive_sequence(acf)
        self.assertGreaterEqual(tau, 1.0)
        self.assertLess(tau, 1.2)
        self.assertGreaterEqual(cutoff, 0)

    def test_geyer_ips_stops_at_first_nonpositive_pair(self) -> None:
        acf = np.array([1.0, 0.5, 0.2, 0.1, 0.02, -0.04, 0.3])
        tau, cutoff = geyer_initial_positive_sequence(acf)
        # Gamma_0=1.5, Gamma_1=0.3, Gamma_2=-0.02; keep first two pairs.
        self.assertAlmostEqual(tau, -1.0 + 2.0 * (1.5 + 0.3))
        self.assertEqual(cutoff, 3)

    def test_ljung_box_is_small_for_a_short_uncorrelated_sequence(self) -> None:
        rng = np.random.default_rng(9)
        acf = acf_fft(rng.normal(size=2000), 20)
        _, p_value = ljung_box(acf, 2000, 10)
        self.assertGreater(p_value, 0.01)


if __name__ == "__main__":
    unittest.main()
