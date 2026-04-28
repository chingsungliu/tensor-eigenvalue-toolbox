"""Sub-step 1.1.5 — F1 root-cause analysis.

F1 (from Sub-step 1.1 baseline run): on ``Q7_large`` (n=50, m=3, seed=42),
HONI_exact returned λ=9.333980 with residual 1.87e-14 and 99 halving
warnings, while HONI_inexact and NNI_spsolve both converged to
λ=10.756272 from the same initial vector. HONI_exact "looks converged"
yet locks onto a different fixed point — a silent wrong answer.

This script:

  §1  Reproduces F1 and records the HONI_exact trajectory
       (lambda_U_history, res_history, hal_per_outer_history,
        inner_histories per outer iter).

  §2  Runs HONI_inexact / NNI_spsolve / NNI_ha on the **same tensor and
       initial vector** and records their final λ.

  §3  Inspects Multi's inner trace inside HONI_exact: where halving
       starts, condition number of the shifted system at trap, residual
       evolution per inner Newton iter.

  §4  Multi-restart spectrum exploration: 20 random initial vectors fed
       to HONI_exact and NNI_spsolve. Tallies which λ each lands on so
       we can decide whether 9.333980 is a real H-eigenvalue or pure
       numerical noise.

Outputs:

  python/benchmarks/results/f1_analysis/f1_trajectory.json
  python/benchmarks/results/f1_analysis/f1_summary.md

Run from python/:

    .venv/bin/python benchmarks/scripts/repro_f1_honi_exact_silent_failure.py

Stdout is suppressed during algorithm calls (Multi prints noisy
"Can't find a suitible step length" warnings — they would otherwise
drown the trace).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_PY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from tensor_utils import honi, multi, nni, sp_Jaco_Ax, tpv


_HALVING_NEEDLE = "Can't find a suitible step length"
_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "f1_analysis"

# Q7_large config (matches Sub-step 1.1 baseline)
N = 50
M = 3
SEED = 42
TOL = 1e-10
MAXIT = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silent_call(fn, *args, **kwargs):
    """Run ``fn`` with stdout silenced and return (result, warning_count)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue().count(_HALVING_NEEDLE)


def _shifted_jacobian(AA, m, x, lam):
    """Form the shifted matrix M = (1/(m-1)) * sp_Jaco_Ax(AA, x, m) - λ·diag(x^(m-2)).

    This mirrors HONI's inner solve operator for the system
    ``(λ·I − A)·y^(m-1) = x^(m-1)`` written through Multi's Jacobian.
    Only used here for condition-number diagnostics.
    """
    n = x.shape[0]
    J = sp_Jaco_Ax(AA, x, m).tocsc() / (m - 1)
    diag_xpow = sp.diags((x ** (m - 2)).astype(np.float64), format="csc")
    return (J - lam * diag_xpow).tocsc()


def _try_cond_estimate(M):
    """Cheap upper bound on condition number via norm and inverse-norm.

    For an n=50 matrix this is fine; for larger, switch to power iteration
    on M and M^{-1}.
    """
    try:
        norm_M = spla.norm(M, ord=np.inf)
    except Exception:
        norm_M = float(np.linalg.norm(M.toarray(), ord=np.inf))
    try:
        # inverse via small dense solve
        inv = np.linalg.inv(M.toarray())
        norm_inv = float(np.linalg.norm(inv, ord=np.inf))
    except Exception:
        norm_inv = float("nan")
    return norm_M, norm_inv, norm_M * norm_inv


# ---------------------------------------------------------------------------
# §1 Reproduce F1 with full HONI_exact trace
# ---------------------------------------------------------------------------


def section_1_reproduce(AA, x0):
    print("§1  Reproducing F1 — HONI_exact on Q7_large with record_history=True")
    (result, warn_count) = _silent_call(
        honi, AA, M, TOL,
        linear_solver="exact",
        maxit=MAXIT,
        initial_vector=x0,
        record_history=True,
        record_inner_history=True,
    )
    lam, x, outer_nit, inner_nit, outer_res, lam_hist, history = result
    print(f"    final λ        = {lam:.10f}")
    print(f"    final residual = {outer_res[-1]:.3e}")
    print(f"    outer nit      = {outer_nit}")
    print(f"    inner nit (Σ)  = {inner_nit}")
    print(f"    halving warns  = {warn_count}")
    print()
    print("    outer trajectory (first row = init):")
    print("    iter |     λ_U      |    residual    |  halvings  |  inner nit")
    print("    -----+--------------+----------------+------------+-----------")
    hal_per_outer = history.get("hal_per_outer_history",
                               history.get("hal_per_outer", []))
    innit_per_outer = history.get("innit_history",
                                  history.get("innit", []))
    for k in range(len(lam_hist)):
        hal = int(hal_per_outer[k]) if k < len(hal_per_outer) else 0
        inn = int(innit_per_outer[k]) if k < len(innit_per_outer) else 0
        res_k = outer_res[k] if k < len(outer_res) else float("nan")
        print(f"    {k:4d} | {lam_hist[k]:12.6f} | {res_k:14.3e} | {hal:10d} | {inn:10d}")
    print()
    return {
        "final_lambda": float(lam),
        "final_residual": float(outer_res[-1]),
        "outer_nit": int(outer_nit),
        "inner_nit_total": int(inner_nit),
        "halving_warning_count": int(warn_count),
        "lambda_U_history": [float(v) for v in lam_hist],
        "outer_res_history": [float(v) for v in outer_res],
        "hal_per_outer_history": [int(v) for v in hal_per_outer],
        "innit_history": [int(v) for v in innit_per_outer],
        "history_keys": list(history.keys()),
        "x_final": x.tolist(),
        "_inner_histories": history.get("inner_histories"),
    }


# ---------------------------------------------------------------------------
# §2 Cross-check on same tensor + same initial vector
# ---------------------------------------------------------------------------


def section_2_crosscheck(AA, x0):
    print("§2  Cross-check — same AA, same x0")
    rows = []

    # HONI_inexact
    (res_in, warn_in) = _silent_call(
        honi, AA, M, TOL,
        linear_solver="inexact", maxit=MAXIT, initial_vector=x0,
    )
    lam_in, _x_in, outer_in, inner_in, outer_res_in, _ = res_in
    rows.append({
        "algorithm": "HONI_inexact",
        "final_lambda": float(lam_in),
        "final_residual": float(outer_res_in[-1]),
        "outer_nit": int(outer_in),
        "inner_nit_total": int(inner_in),
        "halving_warnings": int(warn_in),
    })

    # NNI_spsolve (canonical, halving=False)
    (res_nni, warn_nni) = _silent_call(
        nni, AA, M, TOL,
        linear_solver="spsolve", maxit=MAXIT, initial_vector=x0, halving=False,
    )
    lam_U, _x_n, nit_n, lam_L, outer_res_n, _ = res_nni
    rows.append({
        "algorithm": "NNI_spsolve (canonical)",
        "final_lambda": float(lam_U),
        "final_lambda_lower": float(lam_L),
        "final_residual": float(outer_res_n[-1]),
        "outer_nit": int(nit_n),
        "halving_warnings": int(warn_nni),
    })

    # NNI_ha (halving=True)
    (res_nha, warn_nha) = _silent_call(
        nni, AA, M, TOL,
        linear_solver="spsolve", maxit=MAXIT, initial_vector=x0, halving=True,
    )
    lam_U_h, _x_h, nit_h, lam_L_h, outer_res_h, _ = res_nha
    rows.append({
        "algorithm": "NNI_ha (halving=True)",
        "final_lambda": float(lam_U_h),
        "final_lambda_lower": float(lam_L_h),
        "final_residual": float(outer_res_h[-1]),
        "outer_nit": int(nit_h),
        "halving_warnings": int(warn_nha),
    })

    print(f"    {'algorithm':<26s}  λ              residual      nit   warnings")
    for r in rows:
        print(
            f"    {r['algorithm']:<26s}  "
            f"{r['final_lambda']:<14.10f} "
            f"{r['final_residual']:<13.3e} "
            f"{r['outer_nit']:<5d} "
            f"{r['halving_warnings']}"
        )
    print()
    return rows


# ---------------------------------------------------------------------------
# §3 Multi inner trace inside HONI_exact
# ---------------------------------------------------------------------------


def section_3_inner_trace(AA, x0, sec1):
    print("§3  Multi inner trace — when and where halving fires")
    inner_histories = sec1.get("_inner_histories")
    hal_per_outer = sec1["hal_per_outer_history"]
    lam_hist = sec1["lambda_U_history"]

    if inner_histories is None:
        print("    inner_histories not captured — skipping deep trace")
        return {"available": False}

    # Find the first outer iter where halving > 0
    first_hal_iter = next(
        (k for k, h in enumerate(hal_per_outer) if h > 0), None
    )
    print(f"    first outer iter with halving > 0 : "
          f"{first_hal_iter if first_hal_iter is not None else 'none'}")
    print()

    per_iter_diag = []
    for k, inner_hist in enumerate(inner_histories):
        if inner_hist is None:
            continue
        # Multi's history has res_history (per inner iter), hal_history, theta_history
        res_h = inner_hist.get("res_history")
        hal_h = inner_hist.get("hal_history")
        theta_h = inner_hist.get("theta_history")
        if res_h is None:
            continue
        res_arr = np.asarray(res_h, dtype=float)
        hal_arr = np.asarray(hal_h, dtype=float) if hal_h is not None else np.array([])
        theta_arr = np.asarray(theta_h, dtype=float) if theta_h is not None else np.array([])
        # Find inner iters that triggered halving
        max_inner = res_arr.shape[0] - 1 if res_arr.ndim else 0
        # Multi pre-fills res with na+nb sentinel; "real" entries are those <= sentinel
        # (we just count nonzero hal entries instead)
        n_hal_inner = int(np.sum(hal_arr > 0)) if hal_arr.size else 0
        per_iter_diag.append({
            "outer_iter": k,
            "lambda_U_at_iter": float(lam_hist[k]) if k < len(lam_hist) else None,
            "inner_max_iter_recorded": int(max_inner),
            "inner_min_residual": float(np.min(res_arr[res_arr > 0])) if np.any(res_arr > 0) else None,
            "inner_max_residual": float(np.max(res_arr)) if res_arr.size else None,
            "inner_halving_count": n_hal_inner,
            "min_theta": float(np.min(theta_arr[theta_arr > 0])) if np.any(theta_arr > 0) else None,
        })

    print(f"    {'outer':>5s}  {'λ_U at start':>14s}  {'inner max it':>13s}"
          f"  {'inner res min':>14s}  {'inner res max':>14s}  {'hal in inner':>13s}"
          f"  {'min θ':>10s}")
    for d in per_iter_diag:
        lam_str = f"{d['lambda_U_at_iter']:.6f}" if d['lambda_U_at_iter'] is not None else "—"
        rmin = f"{d['inner_min_residual']:.3e}" if d['inner_min_residual'] is not None else "—"
        rmax = f"{d['inner_max_residual']:.3e}" if d['inner_max_residual'] is not None else "—"
        theta = f"{d['min_theta']:.3e}" if d['min_theta'] is not None else "—"
        print(f"    {d['outer_iter']:5d}  {lam_str:>14s}  "
              f"{d['inner_max_iter_recorded']:13d}  {rmin:>14s}  {rmax:>14s}  "
              f"{d['inner_halving_count']:13d}  {theta:>10s}")
    print()

    # Condition number of the shifted system (λ·I − A) Jacobian at the locked λ
    # We use lam_hist[-1] (final λ) as a stand-in
    final_lam = lam_hist[-1]
    final_x = np.asarray(sec1["x_final"], dtype=float)
    M_shift = _shifted_jacobian(AA, M, final_x, final_lam)
    norm_M, norm_inv, cond = _try_cond_estimate(M_shift)
    print(f"    Shifted system at locked λ={final_lam:.6f}:")
    print(f"      ||M||_inf      = {norm_M:.3e}")
    print(f"      ||M^-1||_inf   = {norm_inv:.3e}")
    print(f"      cond_inf(M)    = {cond:.3e}")
    print()

    return {
        "available": True,
        "first_outer_iter_with_halving": first_hal_iter,
        "per_outer_inner_diagnostics": per_iter_diag,
        "shifted_system_at_locked_lambda": {
            "norm_inf": float(norm_M),
            "norm_inverse_inf": float(norm_inv),
            "cond_inf": float(cond),
        },
    }


# ---------------------------------------------------------------------------
# §4 Multi-restart spectrum exploration
# ---------------------------------------------------------------------------


def section_4_spectrum(AA, n_trials=20):
    print(f"§4  Spectrum exploration — {n_trials} random initial vectors")
    rng = np.random.default_rng(99)
    lam_he = []
    lam_ne = []

    for trial in range(n_trials):
        x0 = np.abs(rng.random(N)) + 0.1
        # HONI_exact
        try:
            (res_he, warn_he) = _silent_call(
                honi, AA, M, TOL,
                linear_solver="exact", maxit=MAXIT, initial_vector=x0,
            )
            lam_h = float(res_he[0])
        except Exception as exc:
            lam_h = float("nan")
        # NNI canonical
        try:
            (res_ne, _) = _silent_call(
                nni, AA, M, TOL,
                linear_solver="spsolve", maxit=MAXIT, initial_vector=x0,
                halving=False,
            )
            lam_n = float(res_ne[0])
        except Exception as exc:
            lam_n = float("nan")
        lam_he.append(lam_h)
        lam_ne.append(lam_n)

    def _bucket(values, decimals=3):
        rounded = [round(v, decimals) if not np.isnan(v) else None for v in values]
        return Counter(rounded)

    bucket_he = _bucket(lam_he)
    bucket_ne = _bucket(lam_ne)

    print(f"    HONI_exact   λ buckets (rounded to 3 dec): "
          f"{dict(bucket_he)}")
    print(f"    NNI_spsolve  λ buckets (rounded to 3 dec): "
          f"{dict(bucket_ne)}")
    print()

    return {
        "n_trials": n_trials,
        "honi_exact_lambdas": lam_he,
        "nni_spsolve_lambdas": lam_ne,
        "honi_exact_buckets": {str(k): v for k, v in bucket_he.items()},
        "nni_spsolve_buckets": {str(k): v for k, v in bucket_ne.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"F1 root-cause analysis — Q7_large (n={N}, m={M}, seed={SEED})")
    print("=" * 78)

    AA, x0 = build_q7_tensor(n=N, m=M, rng_seed=SEED)
    print(f"  AA: {type(AA).__name__}, shape {AA.shape}, nnz={AA.nnz}")
    print(f"  x0: shape {x0.shape}, range [{x0.min():.3f}, {x0.max():.3f}]")
    print()

    sec1 = section_1_reproduce(AA, x0)
    sec2 = section_2_crosscheck(AA, x0)
    sec3 = section_3_inner_trace(AA, x0, sec1)
    sec4 = section_4_spectrum(AA, n_trials=20)

    # Drop the inner_histories from JSON (not serializable, large)
    sec1_for_json = {k: v for k, v in sec1.items() if k != "_inner_histories"}

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "n": N, "m": M, "seed": SEED, "tol": TOL, "maxit": MAXIT,
        },
        "section_1_reproduce": sec1_for_json,
        "section_2_crosscheck": sec2,
        "section_3_inner_trace": sec3,
        "section_4_spectrum": sec4,
    }
    json_path = _OUT_DIR / "f1_trajectory.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON written to: {json_path}")

    # Quick markdown summary
    md_path = _OUT_DIR / "f1_summary.md"
    write_summary(md_path, payload)
    print(f"  Markdown summary: {md_path}")


def write_summary(path, p):
    s = p["section_1_reproduce"]
    cs = p["section_2_crosscheck"]
    inner = p["section_3_inner_trace"]
    spec = p["section_4_spectrum"]

    lines = [
        "# F1 root-cause analysis — auto summary",
        "",
        f"- **Generated**: {p['timestamp']}",
        f"- **Config**: n={p['config']['n']}, m={p['config']['m']}, "
        f"seed={p['config']['seed']}, tol={p['config']['tol']}, "
        f"maxit={p['config']['maxit']}",
        "",
        "## §1 HONI_exact reproduction",
        "",
        f"- final λ           : `{s['final_lambda']:.10f}`",
        f"- final residual    : `{s['final_residual']:.3e}`",
        f"- outer iterations  : `{s['outer_nit']}`",
        f"- inner iterations  : `{s['inner_nit_total']}`",
        f"- halving warnings  : `{s['halving_warning_count']}`",
        "",
        "λ_U trajectory (every outer iter):",
        "",
        "| iter | λ_U | residual | halvings | inner nit |",
        "|---:|---:|---:|---:|---:|",
    ]
    n_iters = len(s["lambda_U_history"])
    for k in range(n_iters):
        hal = s["hal_per_outer_history"][k] if k < len(s["hal_per_outer_history"]) else 0
        inn = s["innit_history"][k] if k < len(s["innit_history"]) else 0
        res_k = s["outer_res_history"][k] if k < len(s["outer_res_history"]) else float("nan")
        lines.append(
            f"| {k} | {s['lambda_U_history'][k]:.6f} | "
            f"{res_k:.3e} | {hal} | {inn} |"
        )
    lines += [
        "",
        "## §2 Cross-check on same input",
        "",
        "| algorithm | final λ | residual | outer nit | halving warns |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cs:
        lines.append(
            f"| {r['algorithm']} | {r['final_lambda']:.10f} | "
            f"{r['final_residual']:.3e} | {r['outer_nit']} | "
            f"{r['halving_warnings']} |"
        )

    lines += ["", "## §3 Inner trace (Multi inside HONI_exact)", ""]
    if not inner.get("available"):
        lines.append("_inner_histories not captured_")
    else:
        lines.append(
            f"- first outer iter with halving > 0: "
            f"**{inner['first_outer_iter_with_halving']}**"
        )
        sm = inner["shifted_system_at_locked_lambda"]
        lines.append(
            f"- shifted system condition at locked λ: "
            f"||M||={sm['norm_inf']:.3e}, "
            f"||M^-1||={sm['norm_inverse_inf']:.3e}, "
            f"cond={sm['cond_inf']:.3e}"
        )
        lines.append("")
        lines.append("Per-outer Multi inner diagnostics:")
        lines.append("")
        lines.append("| outer | λ_U | inner max it | inner res min | inner res max | hal | min θ |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for d in inner["per_outer_inner_diagnostics"]:
            lam = (f"{d['lambda_U_at_iter']:.6f}"
                   if d["lambda_U_at_iter"] is not None else "—")
            rmin = (f"{d['inner_min_residual']:.3e}"
                    if d["inner_min_residual"] is not None else "—")
            rmax = (f"{d['inner_max_residual']:.3e}"
                    if d["inner_max_residual"] is not None else "—")
            theta = (f"{d['min_theta']:.3e}"
                     if d["min_theta"] is not None else "—")
            lines.append(
                f"| {d['outer_iter']} | {lam} | "
                f"{d['inner_max_iter_recorded']} | {rmin} | {rmax} | "
                f"{d['inner_halving_count']} | {theta} |"
            )

    lines += [
        "",
        "## §4 Multi-restart spectrum exploration",
        "",
        f"- `{spec['n_trials']}` random initial vectors",
        f"- HONI_exact λ buckets : `{spec['honi_exact_buckets']}`",
        f"- NNI_spsolve λ buckets: `{spec['nni_spsolve_buckets']}`",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
