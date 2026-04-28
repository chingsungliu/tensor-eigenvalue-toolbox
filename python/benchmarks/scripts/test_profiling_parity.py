"""Verify that enabling the profiler does not perturb numerical output.

Runs Multi / HONI exact / HONI inexact / NNI / NNI_ha on the Q7 baseline
twice — once with profiling disabled and once enabled — and asserts that
final λ, final x, and iteration counts are bit-identical between the
two runs. Any divergence indicates the timing instrumentation introduced
data-flow side effects (which it must not).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from benchmarks.profiling import (
    disable_profiling,
    enable_profiling,
    reset_profiler,
)
from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from tensor_utils import honi, multi, nni


def _run_all(AA, vec, m, tol):
    """Run the four canonical algorithms and capture observable outputs."""
    out = {}

    u, nit_m, _hal = multi(AA, vec, m, tol)
    out["multi"] = {"final_u": u.copy(), "nit": int(nit_m)}

    lam_e, x_e, nit_he, in_he, _r, _l = honi(
        AA, m, tol, linear_solver="exact", maxit=200, initial_vector=vec,
    )
    out["honi_exact"] = {
        "final_lambda": float(lam_e),
        "final_x": x_e.copy(),
        "outer_nit": int(nit_he),
        "inner_nit": int(in_he),
    }

    lam_i, x_i, nit_hi, in_hi, _r, _l = honi(
        AA, m, tol, linear_solver="inexact", maxit=200, initial_vector=vec,
    )
    out["honi_inexact"] = {
        "final_lambda": float(lam_i),
        "final_x": x_i.copy(),
        "outer_nit": int(nit_hi),
        "inner_nit": int(in_hi),
    }

    lam_U, x_n, nit_n, lam_L, _r, _l = nni(
        AA, m, tol, linear_solver="spsolve", maxit=200,
        initial_vector=vec, halving=False,
    )
    out["nni_canonical"] = {
        "final_lambda_U": float(lam_U),
        "final_lambda_L": float(lam_L),
        "final_x": x_n.copy(),
        "nit": int(nit_n),
    }

    lam_Uh, x_nh, nit_nh, lam_Lh, _r, _l = nni(
        AA, m, tol, linear_solver="spsolve", maxit=200,
        initial_vector=vec, halving=True,
    )
    out["nni_halving"] = {
        "final_lambda_U": float(lam_Uh),
        "final_lambda_L": float(lam_Lh),
        "final_x": x_nh.copy(),
        "nit": int(nit_nh),
    }
    return out


def _compare(disabled, enabled):
    failures = []
    for algo, da in disabled.items():
        ea = enabled[algo]
        for key, dval in da.items():
            eval_ = ea[key]
            if isinstance(dval, np.ndarray):
                if not np.array_equal(dval, eval_):
                    diff = float(np.max(np.abs(dval - eval_)))
                    failures.append(f"{algo}.{key}: array diff max={diff:.3e}")
            else:
                if dval != eval_:
                    failures.append(f"{algo}.{key}: scalar disabled={dval} enabled={eval_}")
    return failures


def main():
    print("Profiling parity test — Q7 baseline (n=20, m=3, seed=42)")
    print("=" * 72)
    AA, vec = build_q7_tensor(n=20, m=3, rng_seed=42)

    print("\n  Run 1: profiling DISABLED")
    disable_profiling()
    reset_profiler()
    disabled_run = _run_all(AA, vec, m=3, tol=1e-10)
    for algo, vals in disabled_run.items():
        if "final_lambda" in vals:
            print(f"    {algo:<16s} λ={vals['final_lambda']:.10f}")
        elif "final_lambda_U" in vals:
            print(f"    {algo:<16s} λ_U={vals['final_lambda_U']:.10f}")
        else:
            print(f"    {algo:<16s} nit={vals.get('nit')}")

    print("\n  Run 2: profiling ENABLED")
    reset_profiler()
    enable_profiling()
    enabled_run = _run_all(AA, vec, m=3, tol=1e-10)
    disable_profiling()
    for algo, vals in enabled_run.items():
        if "final_lambda" in vals:
            print(f"    {algo:<16s} λ={vals['final_lambda']:.10f}")
        elif "final_lambda_U" in vals:
            print(f"    {algo:<16s} λ_U={vals['final_lambda_U']:.10f}")
        else:
            print(f"    {algo:<16s} nit={vals.get('nit')}")

    print("\n  Comparing bit-by-bit ...")
    failures = _compare(disabled_run, enabled_run)
    if failures:
        print(f"\n=== FAIL ({len(failures)} divergences) ===")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n=== PASS ===")
        print("  All algorithms bit-identical between profiling enabled / disabled.")


if __name__ == "__main__":
    main()
