"""Per-iteration parity test for HONI port (Layer 3 step 2/2).

Two test functions, one per branch:
    test_honi_parity_exact()    — load honi_reference_exact.mat
    test_honi_parity_inexact()  — load honi_reference_inexact.mat

Both cases share the same AA and initial_vector (built from Multi Q5 rng(42)
sequence); only linear_solver differs.

Tiered parity assertions (see memory/feedback_honi_multi_fragility_propagation.md):

  Tier 1  STRICT
      - ABS tol 1e-10:
          final lambda, final x
          x_history, lambda_history, res, inner_tol_history   (all slots)
          chit_history, hal_per_outer_history                  (slots 0..K-2)
      - REL tol 1e-8 (y_history uses RELATIVE because magnitude grows 10^6×;
                      inexact branch drifts up to 5e-9 even at iter 4):
          y_history                                            (slots 1..K-2)

  Tier 2  APPROX
      - REL tol 1e-2 (last-iter y; inexact case hits 1e-3 at iter K-1):
          y_history[:, K-1]  (last-iter, near-singular shift-invert fragility)

  Tier 3  INFORMATIONAL (no-assert)
      - chit_history[K-1], hal_per_outer_history[K-1], hal_accum_history[K-1]
      - nit, innit, hal scalars (MATLAB vs Python)

PASS condition: Tier 1 + Tier 2 all pass.

Why y_history uses relative error: as lambda_U → true eigenvalue,
`lambda_U * II - AA` becomes near-singular, and y = inv(shifted) @ x^(m-1)
grows exponentially (O(1) at iter 1 → O(10^6) at iter K-1). Absolute diff
scales with |y|; only relative diff reflects the true divergence. See
memory/feedback_honi_multi_fragility_propagation.md for the full rationale.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from poc_iteration import TOLERANCE, find_divergence, print_neighborhood, report
from tensor_utils import honi

REF_DIR = Path(__file__).resolve().parent.parent / "matlab_ref" / "hni"

# Tier thresholds
TIER1_TOL = TOLERANCE        # 1e-10, strict band for ABS-metric fields (x/lambda/res/chit/hal)
TIER1_Y_RTOL = 1e-8          # strict band for REL-metric y_history (inexact branch needs 1e-8)
TIER2_RTOL = 1e-2            # approximate relative, last-iter y under shift-invert fragility


def _per_iter_max_abs_diff(ml_2d, py_2d):
    """(n, K) history → (K,) per-iter max|diff|, NaN-safe."""
    diff = ml_2d - py_2d
    diff = np.where(np.isnan(diff), 0.0, diff)
    return np.max(np.abs(diff), axis=0)


def _per_iter_max_magnitude(ml_2d):
    """(n, K) → (K,) per-iter max|ml|, NaN-safe."""
    ml_safe = np.where(np.isnan(ml_2d), 0.0, ml_2d)
    return np.max(np.abs(ml_safe), axis=0)


def _compare_2d(ml_2d, py_2d, tolerance, name):
    """find_divergence dict for 2-D history via per-iter max ABSOLUTE diff.
    Use for fields with bounded magnitude (x_history, lambda-like fields)."""
    per_iter = _per_iter_max_abs_diff(ml_2d, py_2d)
    return find_divergence(per_iter, np.zeros_like(per_iter), tolerance, name)


def _compare_2d_relative(ml_2d, py_2d, rtol, name):
    """find_divergence dict for 2-D history via per-iter max RELATIVE diff.

    For each slot k: per_iter_rel[k] = max_i |ml[i,k] - py[i,k]| / max_i |ml[i,k]|.
    Use for fields where magnitude grows exponentially (y_history in shift-invert
    eigenvalue iteration: |y| scales ~ 1/(lambda_U - true_eigenvalue), so absolute
    diff at late iters is dominated by magnitude inflation, not by algorithm error).
    """
    per_iter_abs = _per_iter_max_abs_diff(ml_2d, py_2d)
    per_iter_mag = _per_iter_max_magnitude(ml_2d)
    # Guard div-by-zero (init-slot where both sides NaN → mag=0)
    per_iter_rel = per_iter_abs / np.maximum(per_iter_mag, 1e-300)
    return find_divergence(per_iter_rel, np.zeros_like(per_iter_rel), rtol, name)


def _print_tier_header():
    print(
        f"\nTier semantics (metric choice per field):\n"
        f"  Tier 1 STRICT — mixed abs/rel depending on magnitude behavior:\n"
        f"    ABS tol {TIER1_TOL:.0e}:\n"
        f"      - final lambda, final x\n"
        f"      - x_history, lambda_history, res, inner_tol_history  (all slots)\n"
        f"      - chit_history, hal_per_outer_history                 (slots 0..K-2)\n"
        f"    REL tol {TIER1_Y_RTOL:.0e}:  [y_history exponentially scales → needs rel]\n"
        f"      - y_history                                           (slots 1..K-2)\n"
        f"  Tier 2 APPROX — REL tol {TIER2_RTOL:.0e}:\n"
        f"      - y_history[:, K-1]  (last-iter, near-singular shift-invert fragility)\n"
        f"  Tier 3 INFORMATIONAL (no-assert):\n"
        f"      - chit_history[K-1], hal_per_outer_history[K-1], hal_accum_history[K-1]\n"
        f"      - nit, innit, hal scalars\n"
        f"  PASS = Tier 1 + Tier 2 all pass; Tier 3 prints divergence info only."
    )


def _run_parity(ref_filename, branch_label):
    ref_path = REF_DIR / ref_filename
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} not found. Run matlab_ref/hni/generate_honi_reference.m "
            "in MATLAB first."
        )

    data = loadmat(str(ref_path), squeeze_me=True)

    # --- Inputs ---
    AA = data["AA"]
    m = int(data["m"])
    tol = float(data["tol"])
    linear_solver_ml = str(data["linear_solver"])
    initial_vector = np.asarray(data["initial_vector"], dtype=np.float64).ravel()
    maxit = int(data["maxit"])
    assert linear_solver_ml == branch_label, (
        f"{ref_filename}: expected linear_solver='{branch_label}', got '{linear_solver_ml}'"
    )

    n = AA.shape[0]
    print(
        f"Loaded {ref_filename}: "
        f"AA {'sparse' if issparse(AA) else 'dense'} {AA.shape} "
        f"nnz={AA.nnz if issparse(AA) else '-'}, "
        f"m={m}, tol={tol:.0e}, maxit={maxit}, linear_solver={linear_solver_ml!r}"
    )

    # --- MATLAB outputs ---
    x_ml      = np.asarray(data["x"], dtype=np.float64).ravel()
    lambda_ml = float(data["lambda"])
    res_ml    = np.asarray(data["res"], dtype=np.float64).ravel()
    nit_ml    = int(data["nit"])
    innit_ml  = int(data["innit"])
    hal_ml    = int(data["hal"])

    x_history_ml             = np.asarray(data["x_history"], dtype=np.float64)
    lambda_history_ml        = np.asarray(data["lambda_history"], dtype=np.float64).ravel()
    y_history_ml             = np.asarray(data["y_history"], dtype=np.float64)
    inner_tol_history_ml     = np.asarray(data["inner_tol_history"], dtype=np.float64).ravel()
    chit_history_ml          = np.asarray(data["chit_history"], dtype=np.float64).ravel()
    hal_per_outer_history_ml = np.asarray(data["hal_per_outer_history"], dtype=np.float64).ravel()
    innit_history_ml         = np.asarray(data["innit_history"], dtype=np.float64).ravel()
    hal_accum_history_ml     = np.asarray(data["hal_accum_history"], dtype=np.float64).ravel()

    print(f"MATLAB: nit={nit_ml} (1-based → expect Python nit_py = {nit_ml - 1})")
    print(f"        innit={innit_ml}, hal={hal_ml}, lambda={lambda_ml:.12e}")

    # --- Run Python HONI with record_history=True ---
    lambda_py, x_py, nit_py, innit_py, res_py, lambda_hist_py, history_py = honi(
        AA, m, tol,
        linear_solver=linear_solver_ml,
        initial_vector=initial_vector,
        maxit=maxit,
        record_history=True,
        matlab_compat=True,
    )
    hal_py = int(history_py["hal_accum_history"][-1])
    print(f"Python: nit={nit_py} (0-based), innit={innit_py}, hal={hal_py}, lambda={lambda_py:.12e}")

    # --- Structural check: nit must be compatible (1-based offset) ---
    if nit_py + 1 != nit_ml:
        raise AssertionError(
            f"STRUCTURAL: nit mismatch (history arrays have different lengths); "
            f"MATLAB nit={nit_ml}, Python nit+1={nit_py + 1}"
        )

    K = nit_ml          # total slots on both sides
    last = K - 1        # last outer iter slot index

    # --- Shape sanity ---
    shape_checks = [
        (x_history_ml, history_py["x_history"], (n, K), "x_history"),
        (lambda_history_ml, history_py["lambda_history"], (K,), "lambda_history"),
        (res_ml, res_py, (K,), "res"),
        (y_history_ml, history_py["y_history"], (n, K), "y_history"),
        (inner_tol_history_ml, history_py["inner_tol_history"], (K,), "inner_tol_history"),
        (chit_history_ml, history_py["chit_history"], (K,), "chit_history"),
        (hal_per_outer_history_ml, history_py["hal_per_outer_history"], (K,), "hal_per_outer_history"),
        (innit_history_ml, history_py["innit_history"], (K,), "innit_history"),
        (hal_accum_history_ml, history_py["hal_accum_history"], (K,), "hal_accum_history"),
    ]
    for ml_arr, py_arr, expected, name in shape_checks:
        if ml_arr.shape != expected or py_arr.shape != expected:
            raise AssertionError(
                f"{name} shape mismatch: MATLAB {ml_arr.shape}, "
                f"Python {py_arr.shape}, expected {expected}"
            )

    _print_tier_header()

    # ============================================================
    # Tier 1: STRICT (machine epsilon)
    # ============================================================
    print(f"\n=== Tier 1  STRICT  (|diff| <= {TIER1_TOL:.0e}) ===")
    tier1_fails = []

    # 1a: full-slot 1-D arrays
    for ml, py, name in [
        (res_ml, res_py, "res"),
        (lambda_history_ml, history_py["lambda_history"], "lambda_history"),
    ]:
        r = find_divergence(ml, py, TIER1_TOL, name)
        report(r)
        if not r["passed"]:
            tier1_fails.append(r)

    # 1b: inner_tol_history[1:] (slot 0 is NaN both sides)
    r = find_divergence(
        inner_tol_history_ml[1:], history_py["inner_tol_history"][1:],
        TIER1_TOL, "inner_tol_history[1:]",
    )
    report(r)
    if not r["passed"]:
        tier1_fails.append(r)

    # 1c: full x_history (2-D)
    r = _compare_2d(x_history_ml, history_py["x_history"], TIER1_TOL, "x_history (all slots, 2D)")
    report(r)
    if not r["passed"]:
        tier1_fails.append(r)

    # 1d: iter [0..K-2] chit/hal 1-D
    for ml, py, name in [
        (chit_history_ml, history_py["chit_history"], f"chit_history[:{last}]"),
        (hal_per_outer_history_ml, history_py["hal_per_outer_history"], f"hal_per_outer_history[:{last}]"),
    ]:
        r = find_divergence(ml[:last], py[:last], TIER1_TOL, name)
        report(r)
        if not r["passed"]:
            tier1_fails.append(r)

    # 1e: y_history iter 1..K-2 (2-D; slot 0 init NaN, slot K-1 = last excluded)
    #     Uses RELATIVE tolerance because |y| grows exponentially in shift-invert
    #     eigenvalue iteration (O(1) at iter 1 → O(10^6) at iter K-1).
    if last >= 2:
        r = _compare_2d_relative(
            y_history_ml[:, 1:last], history_py["y_history"][:, 1:last],
            TIER1_Y_RTOL, f"y_history[:, 1:{last}] (2D, RELATIVE, exclude init and last)",
        )
        report(r)
        if not r["passed"]:
            tier1_fails.append(r)

    # 1f: final scalars (lambda, x)
    lambda_diff = abs(lambda_ml - lambda_py)
    lambda_pass = lambda_diff <= TIER1_TOL
    print(
        f"  [final lambda] {'PASS' if lambda_pass else 'FAIL'}  "
        f"|λ_ml - λ_py| = {lambda_diff:.3e}  (tolerance {TIER1_TOL:.0e})"
    )
    if not lambda_pass:
        tier1_fails.append({"name": "final lambda", "fail": True, "diff": lambda_diff,
                            "ml": lambda_ml, "py": lambda_py})

    x_diff = float(np.max(np.abs(x_ml - x_py)))
    x_pass = x_diff <= TIER1_TOL
    print(
        f"  [final x]      {'PASS' if x_pass else 'FAIL'}  "
        f"max|x_ml - x_py| = {x_diff:.3e}"
    )
    if not x_pass:
        tier1_fails.append({"name": "final x", "fail": True, "diff": x_diff})

    # ============================================================
    # Tier 2: APPROX (relative tolerance for last-iter y, fragility zone)
    # ============================================================
    print(f"\n=== Tier 2  APPROX  (REL tol = {TIER2_RTOL:.0e}) ===")
    tier2_fails = []

    # Use the same per-iter relative machinery applied to the single last slot.
    # Reshape single column to (n, 1) for _compare_2d_relative compatibility.
    r_tier2 = _compare_2d_relative(
        y_history_ml[:, last:last + 1],
        history_py["y_history"][:, last:last + 1],
        TIER2_RTOL,
        f"y_history[:, {last}] (last iter, RELATIVE)",
    )
    report(r_tier2)
    if not r_tier2["passed"]:
        tier2_fails.append(r_tier2)

    # ============================================================
    # Tier 3: INFORMATIONAL (no assert, expected divergence)
    # ============================================================
    print(f"\n=== Tier 3  INFORMATIONAL  (no-assert) ===")
    print(
        f"  Expected divergence in these metrics due to near-singular shift-invert\n"
        f"  fragility at HONI's last outer iter (scipy.sparse.linalg.spsolve vs\n"
        f"  MATLAB mldivide halving path divergence on ill-conditioned systems).\n"
        f"  See memory/feedback_honi_multi_fragility_propagation.md."
    )
    print()

    def _info(name, ml_val, py_val):
        print(f"    {name:<32s} MATLAB={ml_val:>7d}  Python={py_val:>7d}  diff={abs(ml_val - py_val):>7d}")

    _info(f"chit_history[{last}]",         int(chit_history_ml[last]),          int(history_py["chit_history"][last]))
    _info(f"hal_per_outer_history[{last}]", int(hal_per_outer_history_ml[last]), int(history_py["hal_per_outer_history"][last]))
    _info(f"hal_accum_history[{last}]",     int(hal_accum_history_ml[last]),     int(history_py["hal_accum_history"][last]))
    _info("nit (MATLAB 1-based)",           nit_ml,                               nit_py + 1)
    _info("innit",                          innit_ml,                             innit_py)
    _info("hal (scalar)",                   hal_ml,                               hal_py)

    # ============================================================
    # Final pass/fail
    # ============================================================
    if tier1_fails or tier2_fails:
        print(f"\n=== FAIL ({branch_label}) ===")
        print(f"  Tier 1 failures: {len(tier1_fails)}")
        print(f"  Tier 2 failures: {len(tier2_fails)}")
        # Neighborhood for any Tier 1 1-D find_divergence failure (has first_bad_iter)
        for r in tier1_fails:
            if isinstance(r, dict) and r.get("first_bad_iter") is not None:
                # find the right full array pair to show neighborhood
                nm = r["name"]
                if nm == "res":
                    print_neighborhood(res_ml, res_py, r["first_bad_iter"], 2, nm)
                elif nm == "lambda_history":
                    print_neighborhood(lambda_history_ml, history_py["lambda_history"], r["first_bad_iter"], 2, nm)
                # (other fields: user can dig further if needed)
        raise AssertionError(f"HONI parity FAILED at Tier 1/2 ({branch_label})")

    print(f"\n=== PASS ({branch_label}) ===")
    print(f"  Tier 1 STRICT : {K} slots within abs {TIER1_TOL:.0e} / y_history within rel {TIER1_Y_RTOL:.0e}")
    print(f"  Tier 2 APPROX : last-iter y within rtol = {TIER2_RTOL:.0e}  (actual {r_tier2['max_err']:.3e})")
    print(f"  Tier 3 INFO   : last-iter chit/hal diverged as documented (fragility propagation)")


def test_honi_parity_exact():
    _run_parity("honi_reference_exact.mat", "exact")


def test_honi_parity_inexact():
    _run_parity("honi_reference_inexact.mat", "inexact")


if __name__ == "__main__":
    print("=" * 70)
    print("HONI parity — Case 1: linear_solver = 'exact'")
    print("=" * 70)
    test_honi_parity_exact()
    print()
    print("=" * 70)
    print("HONI parity — Case 2: linear_solver = 'inexact'")
    print("=" * 70)
    test_honi_parity_inexact()
