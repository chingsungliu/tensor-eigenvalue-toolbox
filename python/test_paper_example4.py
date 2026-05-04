"""Reproduce Liu / Guo / Lin (Numer. Math. 2017) §7 Example 4 — weakly
irreducible non-weakly-primitive tensor.

3rd-order, 3-dim tensor with hardcoded entries

    A[1,2,2] = A[2,3,3] = A[3,1,1] = 1   (1-indexed; all other 24 entries 0)

Paper Figure 3: NNI converges to the Perron pair ``λ* = 1``,
``x* = (1/√3)·[1, 1, 1]``; NQZ fails (permutation cycle for any positive
initial vector other than the Perron vector). Paper §7 text: "(37) holds
with theta_k=1 for each iteration of NNI and the halving procedure is
not used".

Day 15 MATLAB Octave verification with ``rand('seed', 42)`` initial
vector ``[0.311981; 0.103585; 0.563460]``:

- MATLAB: nit=10, λ=1.0, Perron=(1/√3)·[1,1,1] perfect.

Python ``np.random.default_rng`` produces a different initial vector
than MATLAB ``rand()``, so iter count may differ. The example converges
from any positive initial vector to the same Perron pair, so this test
asserts on ``(λ, Perron vector)`` rather than exact nit:

- NNI converges within 30 iter (margin over MATLAB's 10).
- Final λ matches 1.0 to 1e-12.
- Final Perron vector matches (1/√3)·[1, 1, 1] to 1e-10 (sign-folded).

Run::

    .venv/bin/python test_paper_example4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from streamlit_app.problems.tensor_eigenvalue.paper_examples import (
    build_liu2017_example4,
)
from tensor_utils import nni

EXPECTED_LAMBDA = 1.0
EXPECTED_X = np.ones(3) / np.sqrt(3)
MAX_NIT = 30           # MATLAB does 10; margin for Python RNG difference
LAMBDA_TOL = 1e-12
X_TOL = 1e-10


def test_paper_example4_nni():
    """NNI converges to Perron pair λ=1, x=(1/√3)·[1,1,1]."""
    AA, x0 = build_liu2017_example4(seed=42)

    print(f"  initial vector x0 = {x0}")
    print(f"  ||x0||            = {np.linalg.norm(x0):.6f}")

    lam_U, x, nit, lam_L, res, lam_hist = nni(
        AA, m=3, tol=1e-13,
        initial_vector=x0,
        maxit=100,
        linear_solver="spsolve",
        halving=False,        # paper §7 Ex 4: theta_k=1 throughout
    )
    final_res = (lam_U - lam_L) / lam_U if lam_U != 0 else float("nan")

    print(f"  nit               = {nit}     (MATLAB nit=10, margin to {MAX_NIT})")
    print(f"  lambda_U          = {lam_U:.15f}")
    print(f"  lambda_L          = {lam_L:.15f}")
    print(f"  final res         = {final_res:.3e}")
    print(f"  Perron vector x   = {x}")
    print(f"  expected x*       = {EXPECTED_X}")

    assert nit <= MAX_NIT, (
        f"Expected NNI converges within {MAX_NIT} iter (MATLAB does 10); "
        f"got nit={nit}. If nit > 25 the builder may be wrong."
    )
    assert abs(lam_U - EXPECTED_LAMBDA) < LAMBDA_TOL, (
        f"Expected λ=1.0 within {LAMBDA_TOL}; got λ={lam_U}."
    )

    # Perron vector match up to sign (NNI returns positive vector for
    # M-tensor inputs, but fold by |x| as a defensive measure).
    x_pos = np.abs(x)
    x_diff = np.linalg.norm(x_pos - EXPECTED_X)
    assert x_diff < X_TOL, (
        f"Expected Perron x = (1/√3)·[1,1,1]; got x = {x_pos}, "
        f"||diff|| = {x_diff:.2e} > {X_TOL:.0e}. If any component is "
        f"negative the toolbox may have left the positive cone."
    )

    print()
    print("PASS: Liu 2017 §7 Example 4 reproduced.")


def main():
    print("test_paper_example4 — Liu / Guo / Lin (Numer. Math. 2017) §7 Example 4")
    print("=" * 78)
    test_paper_example4_nni()


if __name__ == "__main__":
    main()
