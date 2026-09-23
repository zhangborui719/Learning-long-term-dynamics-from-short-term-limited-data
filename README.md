# Lorenz-96 with RK4

This project solves the 40-dimensional Lorenz-96 system

```math
\dot{x}_j=(x_{j+1}-x_{j-2})x_{j-1}-x_j+F,
\qquad x_{j+40}=x_j,
```

using the classical fixed-step fourth-order Runge-Kutta method.

![Lorenz-96 RK4 space-time solution, F=8](results/lorenz96_rk4_F8.png)

## Reproducible run

The committed result uses:

- `N = 40`
- `F = 8` and `F = 64` (same numerical settings)
- `t in [0, 150]` (extended so both datasets have ESS >= 200)
- `dt = 0.01`
- `seed = 20260917`
- `x_j(0) = F + epsilon_j`, where `epsilon_j ~ Normal(0, 0.01^2)`

The fixed seed makes the randomly perturbed initial state reproducible. The
exact values are stored in `results/initial_condition_F8.csv` and
`results/initial_condition_F64.csv`.

## Run

```bash
python -m pip install -r requirements.txt
python solve_lorenz96.py --forcing 8
python solve_lorenz96.py --forcing 64
```

Optional parameters:

```bash
python solve_lorenz96.py --forcing 8 --n 40 --t-end 150 --dt 0.01 --seed 20260917
```

Outputs:

- `results/lorenz96_rk4_F8.png` and `results/lorenz96_rk4_F64.png`: space-time heatmaps
- `results/lorenz96_rk4_F8.csv` and `results/lorenz96_rk4_F64.csv`: time and all 40 state components
- `results/initial_condition_F8.csv` and `results/initial_condition_F64.csv`: initial states

## Decorrelation and effective sample size

Run the analysis after generating both trajectories:

```bash
python analyze_correlations.py
```

The analysis discards the initial `t < 5` transient, computes a time ACF for
each of the 40 spatial components, applies the Ljung–Box test at lag 50, and
estimates the integrated autocorrelation time with Geyer's initial positive
sequence (IPS):

```text
Gamma_k = rho_(2k) + rho_(2k+1)
tau_int = -1 + 2 * sum(Gamma_k)
```

The sum stops before the first non-positive `Gamma_k`. For each component,

```text
ESS = n_samples / tau_int
```

Results are written to `results/correlation/`, including per-component CSV
files, ACF arrays, a comparison figure, and `summary.txt`. A Ljung–Box
`p > 0.05` means the test does not reject the null hypothesis of no temporal
autocorrelation through lag 50; it is not proof that the samples are exactly
independent.

For the final run (`n_samples = 14501` per component after burn-in):

| Forcing | Median ESS | ESS range | Median integrated autocorrelation time | Ljung–Box Q(50) p > 0.05 |
|---:|---:|---:|---:|---:|
| 8 | 306.90 | 281.69–337.26 | 0.4725 | 40 / 40 |
| 64 | 1171.29 | 980.15–1298.53 | 0.1238 | 40 / 40 |

The higher forcing case decorrelates faster in this simulation, so its
effective sample size is larger despite the same raw number of time points.

### Ljung–Box confidence and interpretation

The test uses `alpha = 0.05`, i.e. a 95% confidence level. Its null hypothesis
is that the autocorrelations through lag 50 are jointly zero. All 40 components
for both forcing values reject that null. The reported p-values are shown as
`0.000e+00` because they underflow IEEE double precision; they mean “far below
machine precision”, not an exactly zero probability.

The relatively high temporal autocorrelation is expected here. The integrator
samples a smooth deterministic ODE every `dt = 0.01`, so adjacent states differ
only slightly. The Lorenz–96 nearest-neighbour coupling transports structures
between spatial sites, while the `-x_j` term damps them on an O(1) time scale.
At `F=8`, coherent waves persist longer and the estimated correlation time is
larger. At `F=64`, the stronger nonlinear activity creates faster fluctuations,
so the correlation time is shorter and the ESS is larger. A small Ljung–Box
p-value detects any residual serial dependence; it does not by itself quantify
how many effectively independent samples remain, which is why the Geyer ESS is
reported alongside it.

## Test

```bash
python -m unittest -v
```

The tests verify periodic indexing, preservation of the constant equilibrium,
and the expected fourth-order convergence on a reduced problem with a known
exact solution.
