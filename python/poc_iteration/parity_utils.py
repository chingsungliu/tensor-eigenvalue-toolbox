"""Reusable per-iteration parity helpers (三件組).

Originally inlined in `test_sqrt2_newton_parity.py` (POC). Extracted so
other per-iteration parity tests (Multi / HONI / NNI) can import them
directly without a sys.path hack.

Public API:
    find_divergence(matlab_seq, python_seq, tolerance, name="")
        → dict with first_bad_iter / max_err / etc.
    report(result, tolerance=TOLERANCE)
        → print a one-line PASS or a multi-line FAIL summary.
    print_neighborhood(matlab_seq, python_seq, center, radius=2, label="seq")
        → print per-iter values around the divergence point.

Tolerance convention:
    TOLERANCE = 1e-10 — pass threshold. Expect actual diff to be 0
    (bit-identical) or ~1e-14..1e-16 (float accumulation order).
"""
import numpy as np

TOLERANCE = 1e-10


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


def report(result, tolerance=TOLERANCE):
    """One-line (or multi-line if failed) summary of a find_divergence result."""
    name = result["name"]
    if result["passed"]:
        print(
            f"  [{name}] PASS  max_err = {result['max_err']:.3e}  "
            f"(at iter {result['max_err_iter']}, within tolerance {tolerance:.0e})"
        )
    else:
        print(
            f"  [{name}] FAIL  first divergence at iteration {result['first_bad_iter']}\n"
            f"          MATLAB value = {result['matlab_val']:.15e}\n"
            f"          Python value = {result['python_val']:.15e}\n"
            f"          diff at that iter = {result['diff_at_first_bad']:.3e}  "
            f"(tolerance {tolerance:.0e})\n"
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
