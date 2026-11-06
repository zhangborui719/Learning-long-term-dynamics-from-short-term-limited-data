"""Compare how well the direct and delta flow maps reproduce the energy spectrum.

Both training runs share everything except the regression target: ``direct``
regresses the standardised next state ``z(t + dt)``, ``delta`` regresses the
standardised increment ``z(t + dt) - z(t)``.  The hypothesis under test is that
learning the *increment* is markedly better at reproducing the statistical
structure of the attractor - in particular the energy spectrum

    E_k = dof_k * <|x^_k|^2> / 2,

because the state itself is dominated by the forcing level F = 64 while the
interesting dynamics live in O(1) fluctuations around it; a regression network
trained on the state can spend its capacity pinning down that constant offset
instead of the spectral shape.

Inputs are the rollout CSVs written by :mod:`train_flowmap` (one row per time
step, ``time`` followed by ``pred_0..pred_{d-1}`` and ``true_0..true_{d-1}``).
Everything here is diagnostic: the spectra are compared on the *same* rollout
length, and the truth columns of both files must agree bit-for-bit, which is
asserted before any statistic is computed.

Usage::

    python compare_spectrum.py \
        --direct results/flowmap_v2_global_direct \
        --delta  results/flowmap_v2_global_delta \
        --output results/flowmap_v2_compare
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger("flowmap.compare")

#: wavenumbers included in the log-log slope fit (exclude k=0 and the tail)
SLOPE_MODES: tuple[int, int] = (1, 20)


def load_rollout(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a rollout CSV written by :func:`train_flowmap.run`.

    Args:
        directory: run directory containing ``flowmap_rollout_F64.csv``.

    Returns:
        ``(prediction, truth)`` arrays of shape ``(n, d)``.

    Raises:
        FileNotFoundError: if the rollout CSV is missing.
    """
    path = directory / "flowmap_rollout_F64.csv"
    if not path.exists():
        raise FileNotFoundError(f"rollout CSV not found: {path}")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    n_col = data.shape[1]
    d = (n_col - 1) // 2
    pred = data[:, 1 : 1 + d]
    truth = data[:, 1 + d :]
    expected = ["time"] + [f"pred_{j}" for j in range(d)] + [f"true_{j}" for j in range(d)]
    if header != expected:
        LOGGER.warning("unexpected header in %s; parsed by column position", path.name)
    return pred, truth


def mode_energies(field: np.ndarray) -> np.ndarray:
    """Time-averaged energy of each spatial Fourier mode.

    With the unitary normalisation ``x^_k = (1/sqrt(d)) sum_j x_j exp(-2 pi i jk/d)``
    the energy of mode ``k`` is ``E_k = dof_k |x^_k|^2 / 2`` where ``dof_k = 2``
    for the conjugate-symmetric pairs ``1 <= k < d/2`` and ``1`` otherwise; the
    zero mode carries the forcing offset and is reported but excluded from the
    shape statistics.

    Args:
        field: states, shape ``(n, d)``.

    Returns:
        ``E_k`` averaged over time, shape ``(d // 2 + 1,)``.
    """
    coeff = np.fft.fft(field, axis=1) / np.sqrt(field.shape[1])
    power = np.abs(coeff) ** 2
    independent = field.shape[1] // 2 + 1
    dof = np.ones(independent)
    dof[1 : field.shape[1] // 2] = 2.0
    return (power[:, :independent] * dof / 2.0).mean(axis=0)


def spectral_slope(spectrum: np.ndarray, first: int = SLOPE_MODES[0],
                   last: int = SLOPE_MODES[1]) -> float:
    """Ordinary least-squares slope of ``log E_k`` against ``log k``.

    Args:
        spectrum: mode energies ``E_k``.
        first: smallest wavenumber in the fit.
        last: largest wavenumber in the fit.

    Returns:
        The slope (a negative number: energy decays with wavenumber).
    """
    k = np.arange(first, last + 1, dtype=np.float64)
    log_e = np.log(spectrum[first : last + 1])
    return float(np.polyfit(np.log(k), log_e, deg=1)[0])


def shape_statistics(spectrum: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Spectrum *shape* diagnostics, all offset-invariant.

    Args:
        spectrum: predicted mode energies.
        reference: true mode energies.

    Returns:
        Correlation of log-energies (k >= 1), mean absolute log10 ratio over
        k >= 1, total-energy ratio (k >= 1), peak mode, and the fitted slope.
    """
    log_p = np.log(spectrum[1:])
    log_r = np.log(reference[1:])
    return {
        "log_spectrum_correlation": float(np.corrcoef(log_p, log_r)[0, 1]),
        "mean_abs_log10_ratio": float(np.mean(np.abs(log_p - log_r) / np.log(10.0))),
        "total_energy_ratio": float(spectrum[1:].sum() / reference[1:].sum()),
        "peak_mode": float(np.argmax(spectrum[1:]) + 1),
        "spectral_slope": spectral_slope(spectrum),
    }


def compare_run(direct_dir: Path, delta_dir: Path, output: Path) -> dict[str, Any]:
    """Full spectrum comparison of two rollout directories.

    Args:
        direct_dir: run directory of the ``direct`` mode.
        delta_dir: run directory of the ``delta`` mode.
        output: directory for figures, tables and the summary JSON.

    Returns:
        A dict of everything written to ``spectrum_comparison.json``.
    """
    pred_direct, truth_direct = load_rollout(direct_dir)
    pred_delta, truth_delta = load_rollout(delta_dir)
    # The two runs share the same split, so their truth columns must coincide.
    if not np.array_equal(truth_direct, truth_delta):
        raise ValueError("truth columns of the two rollouts differ; the runs are not comparable")
    n = min(len(pred_direct), len(pred_delta))
    LOGGER.info("comparing %d rollout steps of each run", n)

    truth_spectrum = mode_energies(truth_direct[:n])
    direct_spectrum = mode_energies(pred_direct[:n])
    delta_spectrum = mode_energies(pred_delta[:n])
    reference_slope = spectral_slope(truth_spectrum)

    result: dict[str, Any] = {
        "n_rollout_steps": int(n),
        "n_dim": int(truth_direct.shape[1]),
        "slope_mode_range": list(SLOPE_MODES),
        "reference_peak_mode": float(np.argmax(truth_spectrum[1:]) + 1),
        "reference_spectral_slope": reference_slope,
        "reference_total_energy_k_ge_1": float(truth_spectrum[1:].sum()),
    }
    for name, spectrum in (("direct", direct_spectrum), ("delta", delta_spectrum)):
        result[name] = shape_statistics(spectrum, truth_spectrum)

    # per-mode table
    modes = np.arange(truth_spectrum.size)
    table = np.column_stack([
        modes,
        truth_spectrum,
        direct_spectrum,
        delta_spectrum,
        direct_spectrum / np.maximum(truth_spectrum, 1e-12),
        delta_spectrum / np.maximum(truth_spectrum, 1e-12),
    ])
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "spectrum_per_mode.csv",
        table,
        delimiter=",",
        header="mode,E_k_truth,E_k_direct,E_k_delta,ratio_direct,ratio_delta",
        comments="",
        fmt="%.6e",
    )

    # figure: spectra overlay + ratio panel
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(9.6, 7.4), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax.loglog(modes[1:], truth_spectrum[1:], "o-", ms=4.5, lw=1.7, color="0.15", label="truth")
    ax.loglog(modes[1:], direct_spectrum[1:], "s--", ms=4, lw=1.4, color="#2166ac",
              label="direct: regress $z(t+\\Delta t)$")
    ax.loglog(modes[1:], delta_spectrum[1:], "^--", ms=4, lw=1.4, color="#b2182b",
              label="delta: regress $z(t+\\Delta t)-z(t)$")
    ax.set_ylabel(r"Time-averaged mode energy $E_k$")
    ax.set_title("Energy spectrum of free-running rollouts, F = 64, "
                 "global standardisation, lr = 1e-2")
    ax.grid(alpha=0.25, which="both")
    ax.legend()

    ax_ratio.axhline(1.0, color="0.35", lw=1.0, ls="--")
    ax_ratio.semilogx(modes[1:], table[1:, 4], "s-", ms=3.5, lw=1.3, color="#2166ac",
                      label="direct / truth")
    ax_ratio.semilogx(modes[1:], table[1:, 5], "^-", ms=3.5, lw=1.3, color="#b2182b",
                      label="delta / truth")
    ax_ratio.set(xlabel="Wavenumber $k$", ylabel=r"$E_k$ ratio",
                 title="Per-mode energy ratio (1 = perfect)")
    ax_ratio.grid(alpha=0.25, which="both")
    ax_ratio.legend()

    fig.savefig(output / "spectrum_comparison_F64.png", dpi=220)
    plt.close(fig)

    # headline text table
    lines = [
        "Energy-spectrum comparison: direct vs delta flow map (F=64, global sd, lr=1e-2)",
        "=" * 78,
        f"rollout steps compared          {result['n_rollout_steps']}",
        f"reference peak mode             {result['reference_peak_mode']:.0f}",
        f"reference spectral slope        {reference_slope:+.3f}  (modes "
        f"{SLOPE_MODES[0]}..{SLOPE_MODES[1]})",
        "",
        f"{'statistic':<28}{'direct':>12}{'delta':>12}{'delta gain':>14}",
        "-" * 66,
    ]
    keys = [
        ("log_spectrum_correlation", "log-spectrum correlation", higher_better := True),
        ("total_energy_ratio", "total energy ratio (k>=1)", True),
        ("mean_abs_log10_ratio", "mean |log10 E ratio|", False),
        ("spectral_slope", f"spectral slope ({SLOPE_MODES[0]}..{SLOPE_MODES[1]})", True),
        ("peak_mode", "peak mode", True),
    ]
    for key, label, higher_better in keys:
        value_direct = result["direct"][key]
        value_delta = result["delta"][key]
        gain = value_delta - value_direct
        mark = " (+)" if (gain > 0) == higher_better and abs(gain) > 1e-12 else " (-)"
        lines.append(f"{label:<28}{value_direct:>12.4f}{value_delta:>12.4f}{gain:>+12.4f}{mark}")
    lines.append("")
    lines.append("(+) means delta improves on direct for this statistic")
    (output / "spectrum_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # carry over the headline one-step metrics of each run for context
    for name, directory in (("direct", direct_dir), ("delta", delta_dir)):
        metrics_path = directory / "metrics.json"
        if metrics_path.exists():
            run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result[name + "_run"] = {
                key: run_metrics[key]
                for key in ("test_nrmse", "delta_correlation", "rms_ratio",
                            "spectrum_correlation", "rollout_nrmse",
                            "learning_rate", "standardization")
                if key in run_metrics
            }

    output_json = output / "spectrum_comparison.json"
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote %s and figure to %s", output_json.name, output)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the command-line interface."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--direct", default=str(project_root / "results" / "flowmap_v2_global_direct"),
                        help="run directory of the direct-mode training")
    parser.add_argument("--delta", default=str(project_root / "results" / "flowmap_v2_global_delta"),
                        help="run directory of the delta-mode training")
    parser.add_argument("--output", default=str(project_root / "results" / "flowmap_v2_compare"),
                        help="directory for the comparison artefacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)
    result = compare_run(Path(args.direct), Path(args.delta), Path(args.output))
    print("\nSpectrum comparison (direct -> delta)")
    print("-" * 60)
    for key, label, higher_better in [
        ("log_spectrum_correlation", "log-spectrum correlation", True),
        ("total_energy_ratio", "total energy ratio (k>=1)", True),
        ("mean_abs_log10_ratio", "mean |log10 E ratio|", False),
        ("spectral_slope", "spectral slope", True),
    ]:
        gain = result["delta"][key] - result["direct"][key]
        verdict = "delta better" if (gain > 0) == higher_better else "direct better"
        print(f"  {label:<28} {result['direct'][key]:>10.4f} -> "
              f"{result['delta'][key]:>10.4f}   ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
