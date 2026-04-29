"""Reproduce Liu / Guo / Lin (Numer. Math. 2017) §7 Example 1 — wind-power Markov chain.

Paper Figure 1 specifies:
- NNI converges to relative error < 1e-13 in 5 iterations
- NQZ converges to the same target in 22 iterations (not tested here —
  NQZ is not part of the toolbox)

This file gates Phase B B1: the test fails if our reading of the paper's
A_{ijk} formula is wrong (see ``paper_examples.build_liu2017_example1``
docstring for the disambiguation argument).

Run::

    .venv/bin/python test_paper_example1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from streamlit_app.problems.tensor_eigenvalue.paper_examples import (
    build_liu2017_example1,
)
from tensor_utils import nni

EXPECTED_NIT = 5  # paper Figure 1


def test_paper_example1_nni_5_iter():
    """NNI must converge in exactly 5 iterations on Liu 2017 §7 Example 1."""
    A, x0, m = build_liu2017_example1()

    lam_U, x, nit, lam_L, res, lam_hist = nni(
        A, m, tol=1e-13,
        initial_vector=x0,
        maxit=100,
        linear_solver="spsolve",
        halving=False,  # paper §7 Example 1 uses θ = 1 (Algorithm 1 canonical)
    )

    final_res = (lam_U - lam_L) / lam_U if lam_U != 0 else float("nan")

    print(f"  nit          = {nit}")
    print(f"  lambda_U     = {lam_U:.12f}")
    print(f"  lambda_L     = {lam_L:.12f}")
    print(f"  final res    = {final_res:.3e}")
    print(f"  lambda_history (5 iter):")
    for k, lam in enumerate(lam_hist[: nit + 1]):
        print(f"    k={k}: {lam:.12f}")

    assert nit == EXPECTED_NIT, (
        f"Paper Figure 1 reports NNI converges in {EXPECTED_NIT} iterations, "
        f"got nit={nit}. Possible disambiguation error in "
        f"build_liu2017_example1() — see its docstring."
    )
    assert final_res < 1e-13, (
        f"Final residual {final_res:.3e} > 1e-13 (paper §7 target tol)."
    )

    print()
    print("PASS: Liu 2017 §7 Example 1 reproduced.")


def main():
    print("test_paper_example1 — Liu / Guo / Lin (Numer. Math. 2017) §7 Example 1")
    print("=" * 72)
    test_paper_example1_nni_5_iter()


if __name__ == "__main__":
    main()
