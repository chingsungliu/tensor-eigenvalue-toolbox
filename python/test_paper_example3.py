"""Reproduce Liu / Guo / Lin (Numer. Math. 2017) §7 Example 3 + Figure 2.

Two reproductions on the same `100·D + C` tensor (m=4, n=20):

1. **NNI canonical** (``halving=False``, θ_k = 1): paper §7 text states
   convergence in "no more than 20 iterations". MATLAB output: 19 iter.
   Toolbox output: 18 iter. PASS within paper bound.

2. **NNI-hav** (``halving=True``): paper Figure 2 reports ~74 iter. Day
   15 MATLAB verification (Octave on paper's `NNI_hav.m`) reveals that
   the paper's own MATLAB code does NOT reach 74-iter convergence on
   this input — it runs 200 iter (cap) with bracket residual stuck at
   ~5×10⁻¹¹, halving firing 17 times per outer iteration. The toolbox
   shows essentially identical behaviour (199 iter, residual ~9×10⁻¹¹).

   Paper Figure 2's 74-iter target is unreproducible from the MATLAB
   source provided. Possible explanations: paper used a different
   version of the MATLAB code, a different tol, or hand-traced figure
   estimation. The toolbox is faithfully aligned with the MATLAB code
   as provided. See docs/papers/liu2017_alignment_audit.md §7 for the
   three-way comparison (paper text / MATLAB code / toolbox port).

Run::

    .venv/bin/python test_paper_example3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from streamlit_app.problems.tensor_eigenvalue.paper_examples import (
    build_liu2017_example3,
)
from tensor_utils import nni


PAPER_HALVING_NIT = 74          # paper §7 Example 3 / Figure 2
PAPER_CANONICAL_NIT_MAX = 20    # paper §7 Example 3 ("no more than 20")
ALLOWED_NIT_DIFF = 5            # MATLAB vs Python LU pivot floating-point
                                 # noise-floor accomodation


def test_paper_example3_halving_matches_matlab():
    """NNI-hav matches paper MATLAB behaviour (both fail Figure 2's 74-iter target).

    Day 15 verification (Octave on paper's NNI_hav.m): the paper's own
    MATLAB code on this input runs 200 iter (cap) with bracket residual
    stuck at ~5e-11, halving firing many times per outer iter. Paper
    Figure 2's 74-iter target is unreproducible from the source.

    The toolbox shows essentially identical behaviour to the MATLAB
    code: 199 iter (cap), bracket residual stuck at ~9e-11. This is
    the expected toolbox output, not a bug — we assert that the
    toolbox stays in this paper-MATLAB-matching regime so any future
    accidental "fix" that produces a different (e.g., paper-text-
    matching) trajectory will trip this test and prompt review.

    See docs/papers/liu2017_alignment_audit.md §7.
    """
    AA, x0 = build_liu2017_example3()

    lam_U, x, nit, lam_L, res, lam_hist = nni(
        AA, m=4, tol=1e-13,
        initial_vector=x0,
        maxit=200,
        linear_solver="spsolve",
        halving=True,
    )
    final_res = (lam_U - lam_L) / lam_U if lam_U != 0 else float("nan")

    print(f"  halving=True (toolbox)  : nit={nit:>3d}, λ={lam_U:.10f}, res={final_res:.2e}")
    print(f"  paper MATLAB NNI_hav.m  : nit=200 (cap), res ~5e-11 (NOT CONVERGED)")
    print(f"  paper Figure 2 target   : ~{PAPER_HALVING_NIT} iter (unreproducible)")
    print()
    print("  Toolbox NNI-hav matches MATLAB NNI_hav.m behaviour: both fail to reach")
    print("  paper Figure 2's 74-iter target. See audit §7.")

    assert nit > 100, (
        f"Expected toolbox NNI-hav to stay in MATLAB-matching NOT-CONVERGED "
        f"regime (>100 iter); got nit={nit}. If this test starts failing, "
        f"some commit changed halving behaviour — review against MATLAB."
    )
    assert final_res > 1e-12, (
        f"Expected toolbox NNI-hav to be stuck at noise floor (res > 1e-12); "
        f"got res={final_res:.2e}. If this test starts failing, some commit "
        f"changed halving behaviour — review against MATLAB."
    )


def test_paper_example3_canonical():
    """NNI canonical (halving=False, θ=1) reproduces ≤20 iter."""
    AA, x0 = build_liu2017_example3()

    lam_U, x, nit, lam_L, res, lam_hist = nni(
        AA, m=4, tol=1e-13,
        initial_vector=x0,
        maxit=200,
        linear_solver="spsolve",
        halving=False,          # canonical θ=1, non-monotone allowed
    )
    final_res = (lam_U - lam_L) / lam_U if lam_U != 0 else float("nan")

    print(
        f"  canonical   : nit={nit:>3d}  (paper <={PAPER_CANONICAL_NIT_MAX})"
        f"               λ={lam_U:.6f}  res={final_res:.2e}"
    )

    assert nit <= PAPER_CANONICAL_NIT_MAX, (
        f"Paper §7 Ex 3 expects <={PAPER_CANONICAL_NIT_MAX} iter for "
        f"canonical θ=1; got nit={nit}."
    )
    assert final_res < 1e-13, (
        f"final residual {final_res:.2e} >= 1e-13"
    )


def main():
    print("test_paper_example3 — Liu / Guo / Lin (Numer. Math. 2017) §7 Example 3")
    print("=" * 78)
    print("Canonical (θ=1, paper §7 Ex 3 expects ≤20 iter) — strict test:")
    test_paper_example3_canonical()
    print()
    print("Halving (paper §7 Ex 3 / Figure 2 ~74 iter unreproducible from MATLAB):")
    test_paper_example3_halving_matches_matlab()
    print()
    print("PASS: Canonical reproduction matches paper §7 Example 3.")
    print("Halving matches paper MATLAB NNI_hav.m (both fail Figure 2's 74-iter target).")


if __name__ == "__main__":
    main()
