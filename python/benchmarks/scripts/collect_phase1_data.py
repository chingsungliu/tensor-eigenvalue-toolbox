"""Phase 1 Sub-step 1.3 supplemental data collection.

Produces a single JSON artefact consumed by the baseline report
(``docs/performance_baseline.md``):

  python/benchmarks/results/phase1_supplemental.json

Three datasets are collected:

  D1 (per-case F1 detection flags) — for each test case, run HONI exact
       and HONI inexact with profiling enabled and record the
       ``flag.honi.inner_trap`` / ``flag.honi.lambda_nonmonotone``
       counts together with the resulting λ.

  D2 (Q7_baseline + Q7_large profiling breakdown) — for each algorithm
       (Multi, HONI exact, HONI inexact, NNI canonical, NNI_ha), run
       once with profiling enabled and dump every category's stats.
       This extends the Q7_baseline-only breakdown produced by
       ``profile_baseline.py`` to include Q7_large so the report can
       compare bottleneck shifts as ``n`` grows.

  D3 (NNI_ha vs NNI canonical confirmation) — for each of the five
       Q7-style cases, run both halving=False and halving=True three
       times each with profiling DISABLED (cleanest wall-clock), report
       median wall in milliseconds. Confirms whether the NNI_ha
       speedup observed at Q7_baseline (Sub-step 1.2 finding F-1.2.3)
       is algorithmic or case-specific.

Run from python/:

    .venv/bin/python benchmarks/scripts/collect_phase1_data.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from benchmarks.profiling import (
    disable_profiling,
    enable_profiling,
    get_report,
    reset_profiler,
)
from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from tensor_utils import honi, multi, nni


_HALVING_NEEDLE = "Can't find a suitible step length"
_OUT_DIR = Path(__file__).resolve().parent.parent / "results"

TEST_CASES = [
    {"name": "Q7_baseline",  "n": 20, "m": 3, "seed":  42},
    {"name": "Q7_large",     "n": 50, "m": 3, "seed":  42},
    {"name": "Q7_small",     "n": 10, "m": 3, "seed":  42},
    {"name": "Q7_seed_alt1", "n": 20, "m": 3, "seed":   7},
    {"name": "Q7_seed_alt2", "n": 20, "m": 3, "seed": 137},
]
TOL = 1e-10
MAXIT = 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    failure = None
    result = None
    try:
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:300]}
    return result, buf.getvalue().count(_HALVING_NEEDLE), failure


def _flags_from_report(report):
    return {k: v["count"] for k, v in report.items() if k.startswith("flag.")}


def _trim_breakdown(report):
    """Drop flags from the timing dict for cleaner JSON."""
    return {k: v for k, v in report.items() if not k.startswith("flag.")}


# ---------------------------------------------------------------------------
# D1 — flags per case (HONI exact + inexact)
# ---------------------------------------------------------------------------


def collect_flags():
    print("D1  Per-case F1 detection flags (profiling enabled)")
    out = []
    for case in TEST_CASES:
        AA, x0 = build_q7_tensor(n=case["n"], m=case["m"], rng_seed=case["seed"])
        for branch in ("exact", "inexact"):
            reset_profiler()
            enable_profiling()
            res, warn, fail = _silent(
                honi, AA, case["m"], TOL,
                linear_solver=branch, maxit=MAXIT, initial_vector=x0,
            )
            disable_profiling()
            report = get_report()
            entry = {
                "case": case["name"],
                "n": case["n"], "seed": case["seed"],
                "algorithm": f"HONI_{branch}",
                "halving_warning_count": warn,
                "flags": _flags_from_report(report),
                "failure": fail,
            }
            if res is not None:
                entry["final_lambda"] = float(res[0])
                entry["outer_nit"] = int(res[2])
                entry["inner_nit"] = int(res[3])
            out.append(entry)
            flagstr = (
                ", ".join(f"{k}={v}" for k, v in entry["flags"].items())
                if entry["flags"] else "—"
            )
            lam_str = (
                f"λ={entry.get('final_lambda', float('nan')):.6f}"
                if "final_lambda" in entry else "FAIL"
            )
            print(f"    {case['name']:<14s}  HONI_{branch:<7s}  {lam_str:<18s}  "
                  f"warn={warn:<5d}  flags=[{flagstr}]")
    return out


# ---------------------------------------------------------------------------
# D2 — full breakdown for Q7_baseline + Q7_large
# ---------------------------------------------------------------------------


def collect_breakdown():
    print()
    print("D2  Profiling breakdown — Q7_baseline + Q7_large × 5 algorithms")
    targets = [c for c in TEST_CASES if c["name"] in ("Q7_baseline", "Q7_large")]
    out = {}
    for case in targets:
        AA, x0 = build_q7_tensor(n=case["n"], m=case["m"], rng_seed=case["seed"])
        case_name = case["name"]
        out[case_name] = {}
        runs = [
            ("Multi",        lambda AA=AA, x0=x0, m=case["m"]: multi(AA, x0, m, TOL)),
            ("HONI_exact",   lambda AA=AA, x0=x0, m=case["m"]: honi(
                AA, m, TOL, linear_solver="exact", maxit=MAXIT, initial_vector=x0,
            )),
            ("HONI_inexact", lambda AA=AA, x0=x0, m=case["m"]: honi(
                AA, m, TOL, linear_solver="inexact", maxit=MAXIT, initial_vector=x0,
            )),
            ("NNI_spsolve",  lambda AA=AA, x0=x0, m=case["m"]: nni(
                AA, m, TOL, linear_solver="spsolve", maxit=MAXIT,
                initial_vector=x0, halving=False,
            )),
            ("NNI_ha",       lambda AA=AA, x0=x0, m=case["m"]: nni(
                AA, m, TOL, linear_solver="spsolve", maxit=MAXIT,
                initial_vector=x0, halving=True,
            )),
        ]
        for algo, fn in runs:
            reset_profiler()
            enable_profiling()
            t0 = time.perf_counter()
            _res, warn, fail = _silent(fn)
            wall_s = time.perf_counter() - t0
            disable_profiling()
            report = get_report()
            out[case_name][algo] = {
                "wall_clock_s": wall_s,
                "halving_warning_count": warn,
                "failure": fail,
                "flags": _flags_from_report(report),
                "categories": _trim_breakdown(report),
            }
            print(f"    {case_name:<14s}  {algo:<14s}  wall={wall_s*1000:8.2f}ms  "
                  f"warn={warn}{'  FAILED' if fail else ''}")
    return out


# ---------------------------------------------------------------------------
# D3 — NNI_ha vs NNI canonical, all cases, profiling DISABLED
# ---------------------------------------------------------------------------


def collect_nni_compare(n_runs=3):
    print()
    print(f"D3  NNI_ha vs NNI canonical, {n_runs} runs each (profiling DISABLED)")
    disable_profiling()
    out = []
    for case in TEST_CASES:
        AA, x0 = build_q7_tensor(n=case["n"], m=case["m"], rng_seed=case["seed"])
        row = {
            "case": case["name"],
            "n": case["n"], "seed": case["seed"],
            "canonical": {"runs_ms": [], "nit": None,
                          "final_lambda": None, "failure": None},
            "halving":   {"runs_ms": [], "nit": None,
                          "final_lambda": None, "failure": None},
        }
        for variant, halv in (("canonical", False), ("halving", True)):
            for _ in range(n_runs):
                t0 = time.perf_counter()
                res, _, fail = _silent(
                    nni, AA, case["m"], TOL,
                    linear_solver="spsolve", maxit=MAXIT,
                    initial_vector=x0, halving=halv,
                )
                row[variant]["runs_ms"].append((time.perf_counter() - t0) * 1000)
                if fail is not None:
                    row[variant]["failure"] = fail
                    break
                if res is not None:
                    row[variant]["nit"] = int(res[2])
                    row[variant]["final_lambda"] = float(res[0])
        c_runs = row["canonical"]["runs_ms"]
        h_runs = row["halving"]["runs_ms"]
        c_med = float(np.median(c_runs)) if c_runs and not row["canonical"]["failure"] else None
        h_med = float(np.median(h_runs)) if h_runs and not row["halving"]["failure"] else None
        row["canonical"]["median_ms"] = c_med
        row["halving"]["median_ms"] = h_med
        if c_med is not None and h_med is not None:
            row["speedup_ha_over_canonical"] = c_med / h_med
            verdict = (
                f"ha {c_med / h_med:.2f}× faster"
                if h_med < c_med
                else f"canonical {h_med / c_med:.2f}× faster"
            )
        else:
            row["speedup_ha_over_canonical"] = None
            verdict = (
                "CANONICAL FAILED" if row["canonical"]["failure"]
                else "HALVING FAILED" if row["halving"]["failure"]
                else "—"
            )
        out.append(row)
        print(
            f"    {case['name']:<14s}  "
            f"canonical={c_med if c_med is None else f'{c_med:6.2f}'}ms  "
            f"halving={h_med if h_med is None else f'{h_med:6.2f}'}ms  "
            f"→ {verdict}"
        )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "phase1_supplemental.json"

    print("Phase 1 supplemental data collection")
    print(f"  cases: {[c['name'] for c in TEST_CASES]}")
    print(f"  tol={TOL}, maxit={MAXIT}")
    print()

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"tol": TOL, "maxit": MAXIT},
        "test_cases": TEST_CASES,
        "D1_flags": collect_flags(),
        "D2_breakdown": collect_breakdown(),
        "D3_nni_compare": collect_nni_compare(n_runs=3),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
