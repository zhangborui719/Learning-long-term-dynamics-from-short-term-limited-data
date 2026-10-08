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

## Energy correlations for F=64

The energy analysis uses

```text
E_j(t) = x_j(t)^2 / 2
```

after the same `t < 5` burn-in. Run:

```bash
python analyze_energy_correlations.py
```

This writes `results/energy_correlation/`:

- `energy_F64.csv` and `energy_space_time_F64.png`: energy samples and the space-time energy field.
- `energy_temporal_acf_F64.png`: temporal ACF, averaged over spatial components with the component range.
- `energy_spatial_acf_definitions_F64.png`: three periodic spatial ACF definitions. `snapshot_centered` centers each time snapshot over space; `global_centered` uses one global mean and variance; `component_anomaly` removes each component's temporal mean before forming spatial covariance.
- `energy_spacetime_acf_3d_F64.png`: three-dimensional ACF surface using the global-centered periodic correlation `C(delta t, delta j)`, averaged over all valid time pairs and spatial shifts.
- CSV files containing the plotted numerical values.

## Fourier representation, mode ACF and mode energy (F=64)

The spatial DFT of the state is

```text
xhat_k(t) = N^(-1/2) * sum_j x_j(t) exp(-2 pi i (j - 1) k / N),   k = 0 .. N-1
```

with `N = 40`. Because `x_j` is real, the coefficients satisfy the conjugate
symmetry `xhat_(N-k) = conj(xhat_k)`, so only `k = 0 .. N/2` are independent:
mode `0` and the Nyquist mode `k = N/2 = 20` are real, and the `k = 1 .. 19` are
genuine complex series. Every mode therefore contributes a complex-valued time
series `{xhat_k(t)}`.

Run:

```bash
python analyze_fourier_modes.py
```

The temporal autocorrelation of a mode uses the Hermitian convention

```text
rho_k(tau) = E[conj(xhat_k(t)) xhat_k(t + tau)] / E[|xhat_k(t)|^2]
```

with the temporal mean removed, so `rho_k` is complex with
`rho_k(-tau) = conj(rho_k(tau))` and `rho_k(0) = 1`. `Re rho_k` is itself a real
autocorrelation sequence (it is the variance-weighted average of the ACFs of
`Re xhat_k` and `Im xhat_k`), so Geyer's initial-positive sequence is applied to
it to obtain the integrated time.

Energy is assigned per mode. Since the transform above is unitary, Parseval reads
`sum_k |xhat_k|^2 = ||x||^2`, and with `dof_k = 1` for `k = 0, N/2` and
`dof_k = 2` otherwise,

```text
E_k(t) = dof_k * |xhat_k(t)|^2 / 2 ,     sum_k E_k(t) = ||x(t)||^2 / 2
```

The reported share is `mean_t E_k / mean_t ||x||^2 / 2`, which sums to 100% by
construction. Run with `--self-check` to compare the FFT autocorrelation against
a direct `O(n * lag)` loop.

This writes `results/fourier_modes/`:

- `fourier_energy_spectrum_F64.png`: energy share per mode, log-scale mode energy and the cumulative share.
- `fourier_wavenumber_time_F64.png`: mode energy density `10 log10(E_k / Ebar)` as a function of time and wavenumber.
- `fourier_acf_F64.png`: mode ACF magnitude (log scale), its heat map, and the integrated decorrelation time per mode.
- `fourier_acf_real_imag_F64.png`: real and imaginary parts of the ACF for `k = 0, 1, 5, 10, 20`.
- `mode_amplitudes_F64.csv`, `mode_real_imag_F64.csv`, `mode_energy_F64.csv`: the mode series and their energies.
- `fourier_acf_curves_F64.csv`, `fourier_mode_summary_F64.csv`, `summary.txt`: ACF curves and the per-mode table.

For the committed run (`n_samples = 14501` per mode, `t < 5` discarded):

| Quantity | Value |
|---|---|
| Parseval check `max abs(sum_k |xhat_k|^2 - ||x||^2)` | 1.5e-08 |
| Conjugate-symmetry check `max abs(xhat_(N-k) - conj(xhat_k))` | 2.2e-14 |
| Energy-partition check `max abs(sum_k E_k - ||x||^2/2)` | 5.5e-12 |
| Time-averaged total energy `||x||^2 / 2` | 6265.96 |
| Largest mode share | `k = 0`, 9.59% |
| Energy-containing band | `k = 4 .. 13`, 55.7% |
| Smallest mode share | `k = 20`, 1.42% |
| Median / min / max `tau_int` | 0.1217 / 0.0916 / 0.1627 |

The spectrum is red: the largest share belongs to the spatial mean (`k = 0`)
and the share decreases towards the Nyquist mode. Per degree of freedom the same
ordering holds, so the large scales carry more energy than the small ones. The
integrated decorrelation time follows the opposite ordering overall, from
`0.1627` at `k = 0` down to `0.0916` at `k = 19`, with a secondary maximum near
`k = 10 .. 13` that coincides with the energy-containing band.

