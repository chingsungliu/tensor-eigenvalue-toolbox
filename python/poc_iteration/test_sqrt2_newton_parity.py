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

import numpy as np
from scipy.io import loadmat

from sqrt2_newton import sqrt2_newton

REF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "matlab_ref"
    / "poc_iteration"
    / "sqrt2_newton_reference.mat"
)

TOLERANCE = 1e-10  # pass threshold; expect 0 or machine epsilon here


def find_divergence(matlab_seq, python_seq, tolerance, name=""):
    """Step-by-step compare two sequences; report first-divergence iteration.

    Returns a dict with:
      - name: label for the sequence (e.g. "x_history")
      - passed: True if all |matlab - python| <= tolerance
      - max_err, max_err_iter: overall max diff and where it hits
      - first_bad_iter: iteration index where diff first exceeds tolerance
                        (None if passed)
      - matlab_val, python_val, diff_at_first_bad: values at first divergence
    """
    diff = np.abs(matlab_seq - python_seq)
    max_err = float(diff.max())
    max_err_iter = int(np.argmax(diff))

    over = np.where(diff > tolerance)[0]
    if len(over) == 0:
        return {
            "name": name,
            "passed": True,
            "max_err": max_err,
            "max_err_iter": max_err_iter,
            "first_bad_iter": None,
        }

    first_bad = int(over[0])
    return {
        "name": name,
        "passed": False,
        "max_err": max_err,
        "max_err_iter": max_err_iter,
        "first_bad_iter": first_bad,
        "matlab_val": float(matlab_seq[first_bad]),
        "python_val": float(python_seq[first_bad]),
        "diff_at_first_bad": float(diff[first_bad]),
    }


def report(result):
    """One-line (or multi-line if failed) summary of a find_divergence result."""
    name = result["name"]
    if result["passed"]:
        print(
            f"  [{name}] PASS  max_err = {result['max_err']:.3e}  "
            f"(at iter {result['max_err_iter']}, within tolerance {TOLERANCE:.0e})"
        )
    else:
        print(
            f"  [{name}] FAIL  first divergence at iteration {result['first_bad_iter']}\n"
            f"          MATLAB value = {result['matlab_val']:.15e}\n"
            f"          Python value = {result['python_val']:.15e}\n"
            f"          diff at that iter = {result['diff_at_first_bad']:.3e}  "
            f"(tolerance {TOLERANCE:.0e})\n"
            f"          overall max_err   = {result['max_err']:.3e}  "
            f"(at iter {result['max_err_iter']})"
        )


def print_neighborhood(matlab_seq, python_seq, center, radius=2, label="seq"):
    """Show per-iteration values around the divergence point for context."""
    lo = max(0, center - radius)
    hi = min(len(matlab_seq), center + radius + 1)
    print(f"\n  --- {label} 分岔鄰近 (iteration {center} 前後各 {radius} 步) ---")
    print(f"  {'iter':>4s}  {'MATLAB':>25s}  {'Python':>25s}  {'diff':>12s}")
    for i in range(lo, hi):
        marker = "  <-- DIVERGE" if i == center else ""
        d = abs(matlab_seq[i] - python_seq[i])
        print(
            f"  {i:4d}  {matlab_seq[i]:25.15e}  {python_seq[i]:25.15e}  "
            f"{d:12.3e}{marker}"
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
