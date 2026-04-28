"""Benchmark suite for Layer 3 tensor-eigenvalue algorithms.

Phase 1 Sub-step 1.1 — collects baseline metrics (iteration count, wall
clock, peak memory, final residual / eigenvalue) for the four canonical
algorithms across five Q7-style M-tensor cases. Outputs are JSON (for
Sub-step 1.3 plotting) and Markdown (for human review).

Run from the ``python/`` directory:

    .venv/bin/python benchmarks/benchmark_suite.py

Outputs land in ``python/benchmarks/results/``:
    benchmark_results_YYYYMMDD.json
    benchmark_results_YYYYMMDD.md

Notes
-----
- Q1 / Q3 ill-conditioned builders do not exist in this codebase, so the
  five cases are all Q7-style M-tensors (well-conditioned) with varying
  ``(n, seed)``. Fragility regimes (halving / shift-invert / Rayleigh)
  do not surface here; document for Sub-step 1.2 / 1.3.
- ``tracemalloc`` adds ~10-20% overhead, so absolute wall-clock numbers
  are slightly inflated. Relative comparisons across algorithms remain
  valid because the overhead applies uniformly.
- Multi does not return a residual scalar; it is recomputed post-hoc as
  ``||A · u^(m-1) - b^(m-1)||``. HONI / NNI return ``outer_res_history``
  unconditionally, so their final residual is read directly.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

# Make sibling packages importable when running this file directly.
_PY_ROOT = Path(__file__).resolve().parent.parent
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

import numpy as np

from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from tensor_utils import honi, multi, nni, tpv


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"name": "Q7_baseline",   "n": 20, "m": 3, "seed":  42, "note": "Well-conditioned default; matches Streamlit demo Q7"},
    {"name": "Q7_large",      "n": 50, "m": 3, "seed":  42, "note": "Scale up — measures how cost grows with n"},
    {"name": "Q7_small",      "n": 10, "m": 3, "seed":  42, "note": "Scale down — overhead vs. computation ratio"},
    {"name": "Q7_seed_alt1",  "n": 20, "m": 3, "seed":   7, "note": "Same dimensions, different random tensor (seed variability)"},
    {"name": "Q7_seed_alt2",  "n": 20, "m": 3, "seed": 137, "note": "Same dimensions, different random tensor (seed variability)"},
]


# ---------------------------------------------------------------------------
# Algorithm wrappers — return a dict of correctness metrics
# ---------------------------------------------------------------------------


def run_multi(AA, vec, m, tol):
    u, nit, hal = multi(AA, vec, m, tol)
    b_raised = np.asarray(vec, dtype=np.float64) ** (m - 1)
    final_res = float(np.linalg.norm(tpv(AA, u, m) - b_raised))
    return {
        "iteration_count": int(nit),
        "inner_iteration_count": None,
        "final_residual": final_res,
        "final_lambda": None,
        "lambda_lower": None,
    }


def run_honi(AA, vec, m, tol, branch):
    lam, x, outer_nit, inner_nit, outer_res, _lam_hist = honi(
        AA, m, tol, linear_solver=branch, maxit=200, initial_vector=vec,
    )
    return {
        "iteration_count": int(outer_nit),
        "inner_iteration_count": int(inner_nit),
        "final_residual": float(outer_res[-1]),
        "final_lambda": float(lam),
        "lambda_lower": None,
    }


def run_nni(AA, vec, m, tol):
    lam_U, x, nit, lam_L, outer_res, _lam_hist = nni(
        AA, m, tol, linear_solver="spsolve", maxit=200, initial_vector=vec,
    )
    return {
        "iteration_count": int(nit),
        "inner_iteration_count": None,
        "final_residual": float(outer_res[-1]),
        "final_lambda": float(lam_U),
        "lambda_lower": float(lam_L),
    }


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


_HALVING_WARNING_NEEDLE = "Can't find a suitible step length"


def benchmark_call(call_fn, n_runs):
    """Time + memory profile a callable across ``n_runs`` invocations.

    Stdout from the algorithm is captured and inspected for the Multi
    halving-fragility warning ("Can't find a suitible step length...");
    its occurrence count is returned alongside timing data so we can flag
    runs where the inner Newton solver trapped without raising.
    """
    times = []
    peaks = []
    warning_counts = []
    correctness = None
    failure = None
    for _ in range(n_runs):
        buf = io.StringIO()
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                correctness = call_fn()
            elapsed = time.perf_counter() - t0
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(elapsed)
            peaks.append(peak)
            captured = buf.getvalue()
            warning_counts.append(captured.count(_HALVING_WARNING_NEEDLE))
        except (AssertionError, Exception) as exc:
            elapsed = time.perf_counter() - t0
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(elapsed)
            peaks.append(peak)
            captured = buf.getvalue()
            warning_counts.append(captured.count(_HALVING_WARNING_NEEDLE))
            failure = {
                "exception_type": type(exc).__name__,
                "message": str(exc)[:300],
            }
            # Do not retry — same input, same outcome. Record one entry and bail.
            break

    base = {
        "wall_clock_median_s": float(np.median(times)) if times else None,
        "wall_clock_all_runs_s": [float(t) for t in times],
        "peak_memory_median_mb": (float(np.median(peaks)) / (1024 * 1024)) if peaks else None,
        "peak_memory_all_runs_mb": [p / (1024 * 1024) for p in peaks],
        "halving_warning_count_median": int(np.median(warning_counts)) if warning_counts else 0,
        "halving_warning_count_all_runs": [int(c) for c in warning_counts],
        "failure": failure,
    }
    if correctness is None:
        base.update({
            "iteration_count": None,
            "inner_iteration_count": None,
            "final_residual": None,
            "final_lambda": None,
            "lambda_lower": None,
        })
    else:
        base.update(correctness)
    return base


def main(tol: float = 1e-10, n_runs: int = 3):
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    json_path = out_dir / f"benchmark_results_{timestamp}.json"
    md_path = out_dir / f"benchmark_results_{timestamp}.md"

    print(f"Benchmark suite — {len(TEST_CASES)} cases × 4 algorithms × {n_runs} runs each")
    print(f"  tol={tol}")
    print()

    results = []
    for case in TEST_CASES:
        print(f"  {case['name']} (n={case['n']}, m={case['m']}, seed={case['seed']})")
        AA, vec = build_q7_tensor(n=case["n"], m=case["m"], rng_seed=case["seed"])

        runners = [
            ("Multi",        lambda: run_multi(AA, vec, case["m"], tol)),
            ("HONI_exact",   lambda: run_honi(AA, vec, case["m"], tol, "exact")),
            ("HONI_inexact", lambda: run_honi(AA, vec, case["m"], tol, "inexact")),
            ("NNI_spsolve",  lambda: run_nni(AA, vec, case["m"], tol)),
        ]
        for algo, runner in runners:
            print(f"    {algo:<13s} ...", end=" ", flush=True)
            entry = benchmark_call(runner, n_runs=n_runs)
            entry["case"] = case["name"]
            entry["algorithm"] = algo
            entry["n"] = case["n"]
            entry["m"] = case["m"]
            entry["seed"] = case["seed"]
            entry["tol"] = tol
            results.append(entry)
            if entry["failure"] is not None:
                fail = entry["failure"]
                print(
                    f"FAILED ({fail['exception_type']}: "
                    f"{fail['message'][:80]}{'…' if len(fail['message']) > 80 else ''})"
                )
                continue

            wall_ms = entry["wall_clock_median_s"] * 1000
            mem_mb = entry["peak_memory_median_mb"]
            lam_str = (
                f"λ={entry['final_lambda']:.6f}"
                if entry["final_lambda"] is not None else "λ=—"
            )
            warn = entry["halving_warning_count_median"]
            warn_str = f"  ⚠halving={warn}" if warn > 0 else ""
            print(
                f"nit={entry['iteration_count']:3d}  "
                f"wall={wall_ms:8.2f}ms  "
                f"mem={mem_mb:6.2f}MB  "
                f"res={entry['final_residual']:.2e}  "
                f"{lam_str}{warn_str}"
            )
        print()

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "tol": tol,
                "n_runs": n_runs,
                "test_cases": TEST_CASES,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"  JSON     : {json_path}")

    # Write Markdown
    write_markdown(md_path, results, TEST_CASES, tol, n_runs)
    print(f"  Markdown : {md_path}")


def write_markdown(path, results, cases, tol, n_runs):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Benchmark results — {today}",
        "",
        f"- **Tolerance**: `{tol}`",
        f"- **Runs per (case, algorithm)**: `{n_runs}` (median wall-clock reported)",
        "- **Algorithms**: Multi / HONI exact / HONI inexact / NNI spsolve",
        "- **Note**: All five cases are Q7-style M-tensors; Q1 / Q3 ill-conditioned builders are unavailable in this codebase, so fragility regimes (halving / shift-invert / Rayleigh-quotient noise floor) are not exercised here.",
        "",
        "## Test cases",
        "",
        "| name | n | m | seed | note |",
        "|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(f"| `{c['name']}` | {c['n']} | {c['m']} | {c['seed']} | {c['note']} |")
    lines += [
        "",
        "## Results",
        "",
        "| case | algorithm | nit | inner nit | wall (ms) | peak mem (MB) | final res | final λ | halving warn |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r["failure"] is not None:
            lines.append(
                f"| `{r['case']}` | {r['algorithm']} | **FAILED** | — | — | — | — | — | "
                f"{r['halving_warning_count_median']} |"
            )
            continue
        wall_ms = r["wall_clock_median_s"] * 1000
        inner = r["inner_iteration_count"]
        inner_str = str(inner) if inner is not None else "—"
        lam_str = f"{r['final_lambda']:.6f}" if r["final_lambda"] is not None else "—"
        warn = r["halving_warning_count_median"]
        warn_str = f"⚠ {warn}" if warn > 0 else "0"
        nit_str = str(r["iteration_count"]) if r["iteration_count"] is not None else "—"
        lines.append(
            f"| `{r['case']}` | {r['algorithm']} | {nit_str} | "
            f"{inner_str} | {wall_ms:.2f} | {r['peak_memory_median_mb']:.2f} | "
            f"{r['final_residual']:.2e} | {lam_str} | {warn_str} |"
        )
    lines += [
        "",
        "## Per-run wall-clock detail (sanity check for outliers)",
        "",
        "| case | algorithm | runs (ms) |",
        "|---|---|---|",
    ]
    for r in results:
        runs = r["wall_clock_all_runs_s"]
        runs_ms = ", ".join(f"{t * 1000:.2f}" for t in runs) if runs else "—"
        suffix = " (run before failure)" if r["failure"] is not None else ""
        lines.append(f"| `{r['case']}` | {r['algorithm']} | {runs_ms}{suffix} |")
    lines.append("")

    failures = [r for r in results if r["failure"] is not None]
    if failures:
        lines += [
            "## Failures",
            "",
            "| case | algorithm | exception | message |",
            "|---|---|---|---|",
        ]
        for r in failures:
            f = r["failure"]
            msg = f["message"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{r['case']}` | {r['algorithm']} | `{f['exception_type']}` | {msg} |")
        lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
