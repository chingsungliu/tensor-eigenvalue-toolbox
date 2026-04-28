"""Optional timing harness for Layer 3 algorithms.

Disabled by default. When ``enable_profiling()`` has not been called the
context manager returned by ``Profiler.time(category)`` is a singleton
``contextlib.nullcontext`` — no work is done, so wrapping algorithms in
``with prof.time(...):`` blocks adds only the minimum cost of a method
call and ``__enter__`` / ``__exit__`` invocation.

Usage::

    from benchmarks.profiling import (
        enable_profiling, disable_profiling, get_report, reset_profiler,
    )

    reset_profiler()
    enable_profiling()
    # ... run algorithm calls ...
    report = get_report()
    disable_profiling()

The reported numbers are *inclusive* — when categories nest (e.g.
``honi.inner_multi_call`` contains ``multi.linear_solve`` calls) the
parent total includes the children. Reports are intended for
side-by-side breakdown analysis, not for strict additive accounting.
"""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Dict, List

import numpy as np


class Profiler:
    """Per-category timing + boolean-event recorder.

    Public API used by ``tensor_utils``:
    - ``time(category)`` — context manager. No-op when ``enabled=False``.
    - ``flag(name)`` — record a single boolean event under
      ``flag.<name>``. No-op when ``enabled=False``.

    Public API used by benchmark drivers:
    - ``enabled`` (attribute) — toggle via ``enable_profiling()`` /
      ``disable_profiling()``.
    - ``report()`` — summary dict per category.
    - ``reset()`` — clear all events.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.events: Dict[str, List[int]] = {}
        # Singleton no-op context manager re-used while disabled to avoid
        # generator-frame creation per call.
        self._null = nullcontext()

    @contextmanager
    def _timer(self, category: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - t0
            self.events.setdefault(category, []).append(elapsed)

    def time(self, category: str):
        if not self.enabled:
            return self._null
        return self._timer(category)

    def flag(self, name: str) -> None:
        if not self.enabled:
            return
        self.events.setdefault(f"flag.{name}", []).append(1)

    def report(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for category, samples in self.events.items():
            if category.startswith("flag."):
                out[category] = {"count": int(sum(samples))}
                continue
            arr = np.asarray(samples, dtype=np.int64)
            out[category] = {
                "count": int(arr.size),
                "total_ns": int(arr.sum()),
                "total_ms": float(arr.sum() / 1e6),
                "mean_ns": float(arr.mean()),
                "median_ns": float(np.median(arr)),
                "min_ns": int(arr.min()),
                "max_ns": int(arr.max()),
            }
        return out

    def reset(self) -> None:
        self.events.clear()


# Module-level singleton accessed by tensor_utils.py via get_profiler().
_GLOBAL_PROFILER = Profiler()


def get_profiler() -> Profiler:
    return _GLOBAL_PROFILER


def enable_profiling() -> None:
    _GLOBAL_PROFILER.enabled = True


def disable_profiling() -> None:
    _GLOBAL_PROFILER.enabled = False


def get_report() -> Dict[str, Dict[str, float]]:
    return _GLOBAL_PROFILER.report()


def reset_profiler() -> None:
    _GLOBAL_PROFILER.reset()
