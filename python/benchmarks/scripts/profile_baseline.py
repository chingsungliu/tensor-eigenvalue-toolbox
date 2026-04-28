"""Run Q7 baseline with profiling enabled and dump per-category breakdown.

Phase 1 Sub-step 1.2 deliverable. For each of Multi / HONI exact / HONI
inexact / NNI spsolve runs the algorithm on Q7_baseline (n=20, m=3,
seed=42) with the timing harness enabled, then produces:

  python/benchmarks/results/performance_baseline_breakdown.json
  python/benchmarks/results/performance_baseline_breakdown.md

The breakdown is *inclusive* — when categories nest (e.g. Multi
``tensor_contract`` inside HONI ``inner_multi_call``) both totals see
the same wall-clock. The Markdown table flags this where relevant.
"""
from __future__ import annotations

import json
import sys
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


N, M, SEED, TOL, MAXIT = 20, 3, 42, 1e-10, 200
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def _profile_one(name, runner):
    reset_profiler()
    enable_profiling()
    runner()
    disable_profiling()
    return get_report()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AA, vec = build_q7_tensor(n=N, m=M, rng_seed=SEED)

    print(f"Profiling baseline — Q7 (n={N}, m={M}, seed={SEED}, tol={TOL})")
    print()

    runs = {
        "Multi": lambda: multi(AA, vec, M, TOL),
        "HONI_exact": lambda: honi(
            AA, M, TOL, linear_solver="exact", maxit=MAXIT, initial_vector=vec,
        ),
        "HONI_inexact": lambda: honi(
            AA, M, TOL, linear_solver="inexact", maxit=MAXIT, initial_vector=vec,
        ),
        "NNI_spsolve": lambda: nni(
            AA, M, TOL, linear_solver="spsolve", maxit=MAXIT,
            initial_vector=vec, halving=False,
        ),
        "NNI_ha": lambda: nni(
            AA, M, TOL, linear_solver="spsolve", maxit=MAXIT,
            initial_vector=vec, halving=True,
        ),
    }

    breakdown = {}
    for algo, fn in runs.items():
        rep = _profile_one(algo, fn)
        breakdown[algo] = rep
        # console summary: top-5 by total_ms
        timed = {k: v for k, v in rep.items() if not k.startswith("flag.")}
        flagged = {k: v for k, v in rep.items() if k.startswith("flag.")}
        sorted_cats = sorted(
            timed.items(),
            key=lambda kv: kv[1]["total_ms"],
            reverse=True,
        )
        print(f"  {algo}")
        for cat, stats in sorted_cats[:5]:
            print(
                f"    {cat:<28s} count={stats['count']:>5d}  "
                f"total={stats['total_ms']:8.2f}ms  "
                f"mean={stats['mean_ns'] / 1000:7.1f}μs"
            )
        for cat, stats in flagged.items():
            print(f"    {cat:<28s} count={stats['count']}  ⚠ flag")
        print()

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"n": N, "m": M, "seed": SEED, "tol": TOL, "maxit": MAXIT},
        "breakdown": breakdown,
    }
    json_path = OUT_DIR / "performance_baseline_breakdown.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON     : {json_path}")

    md_path = OUT_DIR / "performance_baseline_breakdown.md"
    write_markdown(md_path, payload)
    print(f"  Markdown : {md_path}")


def write_markdown(path, payload):
    cfg = payload["config"]
    lines = [
        "# Performance baseline breakdown — auto report",
        "",
        f"- **Generated**: {payload['timestamp']}",
        f"- **Case**: Q7 baseline (n={cfg['n']}, m={cfg['m']}, "
        f"seed={cfg['seed']}, tol={cfg['tol']})",
        "- **Categories** are *inclusive*: when timing blocks nest, "
        "the parent total includes children. Subtraction is left to "
        "the reader.",
        "- **F1 detection flags** appear under `flag.*` rows.",
        "",
    ]
    for algo, rep in payload["breakdown"].items():
        timed = {k: v for k, v in rep.items() if not k.startswith("flag.")}
        flagged = {k: v for k, v in rep.items() if k.startswith("flag.")}
        sorted_cats = sorted(
            timed.items(),
            key=lambda kv: kv[1]["total_ms"],
            reverse=True,
        )
        lines.append(f"## {algo}")
        lines.append("")
        if not sorted_cats and not flagged:
            lines.append("_(no events recorded)_")
            lines.append("")
            continue
        lines.append("| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for cat, stats in sorted_cats:
            lines.append(
                f"| `{cat}` | {stats['count']} | "
                f"{stats['total_ms']:.3f} | "
                f"{stats['mean_ns'] / 1000:.2f} | "
                f"{stats['median_ns'] / 1000:.2f} | "
                f"{stats['min_ns'] / 1000:.2f} | "
                f"{stats['max_ns'] / 1000:.2f} |"
            )
        for cat, stats in flagged.items():
            lines.append(
                f"| `{cat}` | **flag count = {stats['count']}** | — | — | — | — | — |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
