"""How long is the chaotic transient of a learned map?

The learned maps are not chaotic repellers: they have a long transient during
which the tangent frame expands, and then settle onto a much thinner invariant
set (sometimes an exact limit cycle).  A Lyapunov exponent measured from the test
state without a warm-up therefore describes the transient, not the attractor.

This helper prints the leading exponent and the closest recurrence at several
warm-up lengths, so the table in ``REPORT_v3.md`` §4.4 can be reproduced.

Usage::

    python probe_transient.py --runs direct=DIR_D delta=DIR_L
    python probe_transient.py --runs direct=DIR --warm-ups 0,2000,20000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from diagnose_codecs import load_run, recurrence, tangent_spectrum
from flowmap_data import build_split
from scan_dynamics import learned_map

LOGGER = logging.getLogger("flowmap.transient")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "results" / "lorenz96_rk4_F64.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=DIR")
    parser.add_argument("--warm-ups", default="0,2000,8000,20000",
                        help="comma-separated numbers of discarded iterations")
    parser.add_argument("--steps", type=int, default=200, help="QR steps after the warm-up")
    parser.add_argument("--probe-steps", type=int, default=4000,
                        help="iterations searched for a recurrence after the warm-up")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = parse_args(argv)
    split = build_split(DATA, stride=10, train_fraction=2.0 / 3.0, burn_in=5.0,
                        standardization="global")
    warm_ups = [int(value) for value in args.warm_ups.split(",")]

    for item in args.runs:
        label, directory = item.split("=", 1)
        resolved = Path(directory)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        run = load_run(label, resolved, split)
        map_fn = learned_map(run)
        print(f"--- {label} ({run.mode}) ---")
        for warm_up in warm_ups:
            spectrum = tangent_spectrum(map_fn, split.test_input[0], n_steps=args.steps,
                                        warm_up=warm_up)
            cycle = recurrence(map_fn, split.test_input[0],
                               n_steps=warm_up + args.probe_steps, burn_in=warm_up,
                               max_period=3000)
            print(f"  warm {warm_up:>6}: lambda {spectrum['lambda_max']:+0.3f}  "
                  f"unstable {spectrum['n_unstable']:>2}  periodic {cycle['periodic']}  "
                  f"p={cycle['period']:>4}  dist {cycle['distance']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
