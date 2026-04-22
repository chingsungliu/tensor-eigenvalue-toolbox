"""Per-iteration parity test for Multi port (Layer 3 step 1/2).

Prerequisite:
    Run matlab_ref/hni/generate_multi_reference.m in MATLAB first to produce
    matlab_ref/hni/multi_reference.mat.

Compares 5 history fields + final u between MATLAB and Python:
    u_history      (n, K)    — state after each outer iter
    res_history    (K,)      — residual after each outer iter
    theta_history  (K,)      — final θ per iter  [0] is NaN on both sides
    hal_history    (K,)      — halving count per iter
    v_history      (n, K)    — Newton direction per iter  [:, 0] is NaN
    u (scalar output)        — final solution

For 2-D fields (u_history, v_history), divergence is reported per-iter using
max |diff| across rows. NaN init slots are skipped (both sides agree).

`nit` scalar relation:  MATLAB nit (1-based) == Python nit (0-based) + 1.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from poc_iteration import TOLERANCE, find_divergence, print_neighborhood, report
from tensor_utils import multi

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "multi_reference.mat"
)


def _per_iter_max_abs_diff(ml_2d, py_2d):
    """For (n, K) history arrays, return (K,) per-iter max|diff|. NaN-safe
    (NaN on both sides → treated as 0)."""
    diff = ml_2d - py_2d
    diff = np.where(np.isnan(diff), 0.0, diff)
    return np.max(np.abs(diff), axis=0)


def _compare_2d(ml_2d, py_2d, tolerance, name):
    """find_divergence-compatible dict for 2-D history via per-iter max|diff|."""
    per_iter = _per_iter_max_abs_diff(ml_2d, py_2d)
    # Compare per_iter against zero sequence: first_bad_iter semantics preserved;
    # matlab_val = per_iter[k], python_val = 0 in the returned dict.
    return find_divergence(per_iter, np.zeros_like(per_iter), tolerance, name)


def _print_neighborhood_2d(ml_2d, py_2d, center, radius=2, label="seq"):
    """Per-iter max|diff| neighborhood for 2-D, with argmax-row hint."""
    per_iter = _per_iter_max_abs_diff(ml_2d, py_2d)
    lo = max(0, center - radius)
    hi = min(per_iter.size, center + radius + 1)
    print(f"\n  --- {label} 分岔鄰近 (iter {center} ±{radius} 步, per-iter max|diff|) ---")
    print(f"  {'iter':>4s}  {'max|diff|':>15s}  {'argmax row':>12s}")
    for i in range(lo, hi):
        marker = "  <-- DIVERGE" if i == center else ""
        col_diff = ml_2d[:, i] - py_2d[:, i]
        col_diff = np.where(np.isnan(col_diff), 0.0, col_diff)
        n_idx = int(np.argmax(np.abs(col_diff)))
        print(f"  {i:4d}  {per_iter[i]:15.6e}  n={n_idx:>10d}{marker}")


def test_multi_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"{REF_PATH} not found. Run matlab_ref/hni/generate_multi_reference.m "
            "in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    # --- Inputs ---
    AA = data["AA"]
    b = data["b"].ravel()
    m = int(data["m"].item())
    tol = float(data["tol"].item())
    print(
        f"Loaded {REF_PATH.name}: "
        f"AA {'sparse' if issparse(AA) else 'dense'} {AA.shape} "
        f"nnz={AA.nnz if issparse(AA) else '-'}, b {b.shape}, m={m}, tol={tol:.0e}"
    )

    # --- MATLAB outputs ---
    u_ml = data["u"].ravel()
    nit_ml = int(data["nit"].item())

    # --- MATLAB history (slot semantics: 1-based, slot 1 = init) ---
    u_history_ml     = data["u_history"]                  # (n, nit_ml)
    res_history_ml   = data["res_history"].ravel()        # (nit_ml,)
    theta_history_ml = data["theta_history"].ravel()
    hal_history_ml   = data["hal_history"].ravel()
    v_history_ml     = data["v_history"]                  # (n, nit_ml)

    print(f"MATLAB: nit={nit_ml} (1-based → expect Python nit_py = {nit_ml - 1})")

    # --- Run Python multi() ---
    u_py, nit_py, hal_py, history_py = multi(
        AA, b, m, tol, record_history=True, matlab_compat=True,
    )
    print(f"Python: nit={nit_py} (0-based)")

    # --- nit scalar check: MATLAB nit_ml == Python nit_py + 1 ---
    if nit_py + 1 != nit_ml:
        raise AssertionError(
            f"nit mismatch: MATLAB nit={nit_ml}, Python nit+1={nit_py + 1} "
            "(they must be equal under 1-based ↔ 0-based offset)"
        )

    # --- Shape sanity ---
    K = nit_ml  # both sides have K slots (init + K-1 outer iters)
    n = len(u_ml)
    shape_checks = [
        (u_history_ml, history_py["u_history"], (n, K), "u_history"),
        (res_history_ml, history_py["res_history"], (K,), "res_history"),
        (theta_history_ml, history_py["theta_history"], (K,), "theta_history"),
        (hal_history_ml, history_py["hal_history"], (K,), "hal_history"),
        (v_history_ml, history_py["v_history"], (n, K), "v_history"),
    ]
    for ml_arr, py_arr, expected, name in shape_checks:
        if ml_arr.shape != expected or py_arr.shape != expected:
            raise AssertionError(
                f"{name} shape mismatch: MATLAB {ml_arr.shape}, "
                f"Python {py_arr.shape}, expected {expected}"
            )

    # --- Per-iter parity ---
    print(f"\n=== Per-iteration parity (tolerance = {TOLERANCE:.0e}) ===")

    res_r = find_divergence(res_history_ml, history_py["res_history"], TOLERANCE, "res_history")
    hal_r = find_divergence(hal_history_ml, history_py["hal_history"], TOLERANCE, "hal_history")
    # theta_history: slot [0] is NaN on both sides — skip it.
    theta_r = find_divergence(
        theta_history_ml[1:],
        history_py["theta_history"][1:],
        TOLERANCE, "theta_history[1:]",
    )
    u_r = _compare_2d(u_history_ml, history_py["u_history"], TOLERANCE, "u_history (2D)")
    # v_history: column [:, 0] is NaN on both sides — skip it.
    v_r = _compare_2d(
        v_history_ml[:, 1:],
        history_py["v_history"][:, 1:],
        TOLERANCE, "v_history[:, 1:] (2D)",
    )

    all_results = [res_r, hal_r, theta_r, u_r, v_r]
    for r in all_results:
        report(r)

    # --- Final u sanity ---
    u_diff = float(np.max(np.abs(u_ml - u_py)))
    u_passed = u_diff <= TOLERANCE
    print(
        f"  [final u]  {'PASS' if u_passed else 'FAIL'}  "
        f"max|u_ml - u_py| = {u_diff:.3e}  (tolerance {TOLERANCE:.0e})"
    )

    # --- Summary ---
    divergence_iters = [r["first_bad_iter"] for r in all_results if r["first_bad_iter"] is not None]
    if not divergence_iters and u_passed:
        print("\n=== Summary ===")
        print(
            f"All {K} history slots within tolerance across u / res / theta / hal / v_history. "
            f"Final u matches (max|diff| = {u_diff:.3e}). Per-iteration parity PASSED."
        )
        return

    # --- Divergence report ---
    print("\n=== 分岔點報告 ===")
    if divergence_iters:
        earliest_slice_idx = min(divergence_iters)
        print(f"Earliest history divergence at (sliced) iteration {earliest_slice_idx}")

        # For each failing field, pick the right re-mapped center and print neighborhood.
        def _report_field(r, is_2d, mapping):
            if r["passed"]:
                return
            center = mapping(r["first_bad_iter"])
            if is_2d:
                full_ml, full_py = mapping.full_2d
                _print_neighborhood_2d(full_ml, full_py, center, 2, r["name"])
            else:
                full_ml, full_py = mapping.full_1d
                print_neighborhood(full_ml, full_py, center, 2, r["name"])

        # Inline mapping with simple objects (avoid boilerplate class)
        class _M:
            def __init__(self, shift, full_1d=None, full_2d=None):
                self.shift, self.full_1d, self.full_2d = shift, full_1d, full_2d
            def __call__(self, k):
                return k + self.shift

        _report_field(res_r, False, _M(0, full_1d=(res_history_ml, history_py["res_history"])))
        _report_field(hal_r, False, _M(0, full_1d=(hal_history_ml, history_py["hal_history"])))
        _report_field(theta_r, False, _M(1, full_1d=(theta_history_ml, history_py["theta_history"])))
        _report_field(u_r, True, _M(0, full_2d=(u_history_ml, history_py["u_history"])))
        _report_field(v_r, True, _M(1, full_2d=(v_history_ml, history_py["v_history"])))

    if not u_passed:
        n_idx = int(np.argmax(np.abs(u_ml - u_py)))
        print(
            f"\n  [final u] FAIL  max |u_ml - u_py| = {u_diff:.3e}  "
            f"(argmax row = {n_idx}, u_ml[{n_idx}]={u_ml[n_idx]:.15e}, "
            f"u_py[{n_idx}]={u_py[n_idx]:.15e})"
        )

    raise AssertionError("Multi per-iteration parity FAILED — see report above")


if __name__ == "__main__":
    test_multi_parity()
