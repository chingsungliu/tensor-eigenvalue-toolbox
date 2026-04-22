"""Per-iteration parity test for sqrt2_newton.

This POC establishes the **分岔點報告 (divergence-point report)** framework
that Multi / HONI / NNI parity tests will reuse.

Key design: don't just report overall max_err. Find the **first iteration**
where the difference exceeds tolerance — that's the divergence point. Once
iteration paths diverge, downstream iterations carry the error forward so
max_err alone doesn't tell you where the bug is.

Prerequisite:
    Run matlab_ref/poc_iteration/sqrt2_newton.m in MATLAB first to produce
    sqrt2_newton_reference.mat.
"""
from pathlib import Path

from scipy.io import loadmat

from sqrt2_newton import sqrt2_newton
from parity_utils import TOLERANCE, find_divergence, print_neighborhood, report

REF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "matlab_ref"
    / "poc_iteration"
    / "sqrt2_newton_reference.mat"
)


def test_sqrt2_newton_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"{REF_PATH} not found. Run matlab_ref/poc_iteration/sqrt2_newton.m "
            "in MATLAB first."
        )

    data = loadmat(str(REF_PATH))
    x_matlab = data["x_history"].ravel()
    res_matlab = data["res_history"].ravel()

    print(
        f"Loaded {REF_PATH.name}: "
        f"x_history shape {x_matlab.shape},  res_history shape {res_matlab.shape}"
    )

    # Run Python version with the same initial conditions
    x_py, res_py = sqrt2_newton(x0=1.0, n_iter=10, matlab_compat=True)

    if x_py.shape != x_matlab.shape:
        raise AssertionError(
            f"x_history shape mismatch: MATLAB {x_matlab.shape} vs Python {x_py.shape}"
        )
    if res_py.shape != res_matlab.shape:
        raise AssertionError(
            f"res_history shape mismatch: MATLAB {res_matlab.shape} vs Python {res_py.shape}"
        )

    # Per-sequence divergence check
    x_result = find_divergence(x_matlab, x_py, TOLERANCE, "x_history")
    res_result = find_divergence(res_matlab, res_py, TOLERANCE, "res_history")

    print("\n=== Per-iteration parity ===")
    report(x_result)
    report(res_result)

    # If any divergence, show neighborhood around earliest divergence
    divergence_iters = [
        r["first_bad_iter"] for r in [x_result, res_result]
        if r["first_bad_iter"] is not None
    ]

    if not divergence_iters:
        print("\n=== Summary ===")
        print("All 11 iterations within tolerance. Per-iteration parity passed.")
        return

    earliest = min(divergence_iters)
    print(f"\n=== 分岔點報告 ===")
    print(f"Earliest divergence at iteration {earliest}")
    print_neighborhood(x_matlab, x_py, earliest, radius=2, label="x_history")
    print_neighborhood(res_matlab, res_py, earliest, radius=2, label="res_history")

    raise AssertionError(
        f"per-iteration parity FAILED: earliest divergence at iter {earliest}"
    )


if __name__ == "__main__":
    test_sqrt2_newton_parity()
