"""Numerical checks for the Lorenz-96 RK4 implementation."""

import unittest

import numpy as np

from solve_lorenz96 import integrate_lorenz96, lorenz96_rhs, rk4_step


class SolverTests(unittest.TestCase):
    def test_periodic_rhs_matches_direct_indexing(self) -> None:
        x = np.arange(1.0, 7.0)
        forcing = 2.5
        expected = np.array(
            [
                (x[(j + 1) % 6] - x[(j - 2) % 6]) * x[(j - 1) % 6]
                - x[j]
                + forcing
                for j in range(6)
            ]
        )
        np.testing.assert_allclose(lorenz96_rhs(x, forcing), expected)

    def test_equilibrium_is_preserved(self) -> None:
        forcing = 8.0
        state = np.full(40, forcing)
        np.testing.assert_allclose(rk4_step(state, 0.01, forcing), state)

    def test_rk4_exhibits_fourth_order_convergence(self) -> None:
        # With N=1 the nonlinear term cancels, so x'=F-x has a known solution.
        x0 = np.array([1.25])
        forcing = 3.0
        exact = forcing + (x0[0] - forcing) * np.exp(-1.0)
        errors = []
        for dt in (0.1, 0.05):
            _, states = integrate_lorenz96(x0, forcing, 1.0, dt)
            errors.append(abs(states[-1, 0] - exact))
        self.assertGreater(errors[0] / errors[1], 14.0)


if __name__ == "__main__":
    unittest.main()
