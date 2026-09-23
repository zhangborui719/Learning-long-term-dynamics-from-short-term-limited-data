"""Checks for the autocorrelation and effective-sample-size analysis."""

import unittest

import numpy as np

from analyze_correlations import acf_fft, integrated_autocorrelation_time, ljung_box


class AnalysisTests(unittest.TestCase):
    def test_constant_series_has_unit_acf(self) -> None:
        np.testing.assert_allclose(acf_fft(np.ones(20), 5), np.ones(6))

    def test_white_noise_has_approximately_one_sample_per_step(self) -> None:
        rng = np.random.default_rng(7)
        acf = acf_fft(rng.normal(size=10000), 100)
        tau, cutoff = integrated_autocorrelation_time(acf)
        self.assertGreaterEqual(tau, 1.0)
        self.assertLess(tau, 1.2)
        self.assertGreaterEqual(cutoff, 0)

    def test_ljung_box_is_small_for_a_short_uncorrelated_sequence(self) -> None:
        rng = np.random.default_rng(9)
        acf = acf_fft(rng.normal(size=2000), 20)
        _, p_value = ljung_box(acf, 2000, 10)
        self.assertGreater(p_value, 0.01)


if __name__ == "__main__":
    unittest.main()
