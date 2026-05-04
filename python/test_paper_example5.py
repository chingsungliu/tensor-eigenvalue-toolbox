"""Reproduce Liu / Guo / Lin (Numer. Math. 2017) §7 Example 5 — paper Table 2.

Z-tensor smallest eigenpair via NNI on ``A = s·I − Z`` with ``s = max
diag Z``. Smallest eigenpair of ``Z`` is recovered as ``μ* = s − ρ(A)``.

Paper Table 2 reports NNI iter ≤ 7 across 9 ``(m, n)`` cases, illustrating
quadratic convergence of NNI on weakly-primitive nonneg tensors.

The Z-tensor construction includes a random trap potential
``V[v,…,v] = 0.1·u_v``. Python ``np.random.default_rng`` differs from
MATLAB ``rand()``, so the random ``v`` differs across implementations
and exact iter counts may vary by 1.

Day 15 MATLAB Octave verification (5 small cases, ``rand('seed', 42)``):

  (3, 20): MATLAB nit=5, paper Table 2 = 5    ✓
  (3, 50): MATLAB nit=6, paper Table 2 = 6    ✓
  (3,100): MATLAB nit=7, paper Table 2 = 7    ✓
  (4, 20): MATLAB nit=5, paper Table 2 = 5    ✓
  (4, 50): MATLAB nit=5, paper Table 2 = 6    (RNG -1)

``ALLOWED_NIT_DIFF = 2`` covers RNG differences and MATLAB-vs-Python LU
trajectory variations.

Run::

    .venv/bin/python test_paper_example5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from streamlit_app.problems.tensor_eigenvalue.paper_examples import (
    build_liu2017_example5,
)
from tensor_utils import nni


PAPER_TABLE_2 = {
    (3, 20): 5, (3, 50): 6, (3, 100): 7,
    (4, 20): 5, (4, 50): 6, (4, 100): 7,
    (5, 20): 5, (5, 50): 5, (5, 100): 6,
}

ALLOWED_NIT_DIFF = 2


def test_paper_example5_table2():
    """All 9 ``(m, n)`` cases of Liu 2017 Example 5 Table 2."""
    print(
        f"{'m':>2} {'n':>4}  {'paper':>5} {'got':>4}  "
        f"{'lambda(A)':>14}  {'min_eig(Z)':>14}  {'res':>10}  {'status':>10}"
    )
    print("-" * 80)

    failures = []
    for (m, n), expected_nit in sorted(PAPER_TABLE_2.items()):
        AA, x0, s = build_liu2017_example5(m, n, seed=42)

        lam_U, x, nit, lam_L, res, lam_hist = nni(
            AA, m, tol=1e-13,
            initial_vector=x0,
            maxit=100,
            linear_solver="spsolve",
            halving=False,        # paper §7 Ex 5: theta_k = 1 throughout
        )
        final_res = (lam_U - lam_L) / lam_U if lam_U != 0 else float("nan")
        smallest_eig_Z = s - lam_U
        nit_diff = abs(nit - expected_nit)
        status = "PASS" if nit_diff <= ALLOWED_NIT_DIFF else f"FAIL d={nit_diff}"

        print(
            f"{m:>2} {n:>4}  {expected_nit:>5} {nit:>4}  "
            f"{lam_U:>14.6f}  {smallest_eig_Z:>14.6e}  "
            f"{final_res:>10.2e}  {status:>10}"
        )
        if nit_diff > ALLOWED_NIT_DIFF:
            failures.append((m, n, expected_nit, nit, lam_U, final_res))

    if failures:
        print()
        print(
            f"FAIL: {len(failures)} of {len(PAPER_TABLE_2)} cases differ from "
            f"paper Table 2 by > {ALLOWED_NIT_DIFF} iter"
        )
        for m, n, exp, got, lam, res in failures:
            print(
                f"  m={m}, n={n}: expected nit={exp}, got nit={got}, "
                f"lambda={lam:.6f}, res={res:.2e}"
            )
        raise AssertionError(
            f"{len(failures)} cases out of allowed range "
            f"(ALLOWED_NIT_DIFF={ALLOWED_NIT_DIFF})"
        )

    print()
    print(
        f"PASS: All {len(PAPER_TABLE_2)} cases match paper Table 2 within "
        f"±{ALLOWED_NIT_DIFF} iter."
    )


def main():
    print("test_paper_example5 — Liu / Guo / Lin (Numer. Math. 2017) §7 Example 5")
    print("=" * 80)
    test_paper_example5_table2()


if __name__ == "__main__":
    main()
