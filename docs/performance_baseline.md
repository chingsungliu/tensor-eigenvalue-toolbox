# Performance baseline — tensor-eigenvalue toolbox

**Phase**: 1 (Sub-steps 1.1, 1.1.5, 1.2, 1.3)
**Status**: baseline complete; Phase 2 optimization candidates listed in §5
**Generated**: 2026-04-28
**Raw data**:
- `python/benchmarks/results/benchmark_results_20260428.json` (Sub-step 1.1)
- `python/benchmarks/results/f1_analysis/f1_trajectory.json` (Sub-step 1.1.5)
- `python/benchmarks/results/performance_baseline_breakdown.json` (Sub-step 1.2)
- `python/benchmarks/results/phase1_supplemental.json` (Sub-step 1.3)

---

## §1 Setup

### Environment

| Item | Value |
|---|---|
| Python | 3.9.6 (`python/.venv/`) |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| OS | macOS 15.6.1 (Darwin 24.6.0) |
| Architecture | arm64 |
| CPU | Apple M1 Ultra, 20 physical cores |
| RAM | 128 GiB |
| MATLAB reference | base MATLAB (no Image Processing Toolbox) |

### Test cases

All five cases are produced by
`streamlit_app.problems.tensor_eigenvalue.defaults.build_q7_tensor`,
which constructs a positive M-tensor by combining a diagonal
`d ~ U[1, 11]` with a sparse perturbation (density 0.02, magnitude 0.01)
and returns a positive initial vector `x0 = |rand(n)| + 0.1`. Phase 1
did not have access to dedicated ill-conditioned builders (the missing
`build_q1_tensor` / `build_q3_tensor` from the MATLAB pipeline), so the
five cases vary `n` and the random seed only.

| Case | n | m | seed | Role |
|---|---:|---:|---:|---|
| `Q7_baseline` | 20 | 3 | 42 | Well-conditioned default; matches the Streamlit demo Q7 |
| `Q7_large` | 50 | 3 | 42 | Scale-up — sensitivity to `n` |
| `Q7_small` | 10 | 3 | 42 | Scale-down — overhead-vs-computation ratio |
| `Q7_seed_alt1` | 20 | 3 | 7 | Same dimensions, different random tensor |
| `Q7_seed_alt2` | 20 | 3 | 137 | Same dimensions, different random tensor |

### Algorithms

| Name | Function call | Branch / kwargs |
|---|---|---|
| Multi | `multi(AA, b, m, tol)` | n/a |
| HONI exact | `honi(AA, m, tol, linear_solver="exact", maxit=200, initial_vector=x0)` | exact branch |
| HONI inexact | `honi(..., linear_solver="inexact", ...)` | inexact branch |
| NNI canonical | `nni(AA, m, tol, linear_solver="spsolve", maxit=200, initial_vector=x0, halving=False)` | NNI.m mirror |
| NNI_ha | `nni(..., halving=True)` | NNI_ha.m mirror |

Tolerance was fixed at `tol = 1e-10` and outer iteration cap at
`maxit = 200` for all runs.

### Metrics

| Metric | Definition |
|---|---|
| `iter_count` (`nit`) | Outer iteration count, Python 0-based (MATLAB `nit_mat = nit + 1`) |
| `inner_iter_count` | Sum of inner Multi Newton iterations per HONI outer iter |
| `wall_clock_median_s` | Median over `n_runs=3` measurements with `time.perf_counter` |
| `peak_memory_mb` | `tracemalloc` peak per run, then median |
| `final_residual` | HONI / NNI: `outer_res_history[-1]`; Multi: `‖A · u^(m-1) − b^(m-1)‖` post-hoc |
| `final_lambda` | HONI: `lam`; NNI: `lam_U`; Multi: n/a |
| `halving_warning_count` | Count of "Can't find a suitible step length" prints from inner Multi (captured via `redirect_stdout`) |
| `flag.honi.inner_trap` | Per-outer-iter flag fired when inner Multi hits `chit_py >= 99` (cap = `_BUF − 1`) |
| `flag.honi.lambda_nonmonotone` | Per-outer-iter flag fired after `nit ≥ 3` if `|Δλ_U| / λ_U_prev > 1%` |

### Profiling instrumentation

`python/benchmarks/profiling.py` provides `Profiler` with two APIs used
by `tensor_utils.py`:
- `time(category)` — context manager. Disabled by default; returns a
  cached `nullcontext` when `enabled = False`. Bit-identical numerics
  enabled vs. disabled (verified by
  `benchmarks/scripts/test_profiling_parity.py`).
- `flag(name)` — boolean event recorder. Disabled by default.

12 `with` blocks were added across `multi`, `honi`, and `nni`:

| Algorithm | Categories |
|---|---|
| Multi | `multi.tensor_contract`, `multi.linear_solve`, `multi.halving_search`, `multi.residual_check` |
| HONI | `honi.outer_iter`, `honi.inner_multi_call`, `honi.tensor_contract`, `honi.rayleigh_extract`, `honi.lambda_update` |
| NNI | `nni.tensor_contract`, `nni.linear_solve`, `nni.rayleigh_quotient`, `nni.halving_inner`, `nni.bracket_update` |

**Inclusive timing semantics**: nested categories double-count (e.g.
`honi.inner_multi_call` includes all `multi.*` recorded during that
call). The report subtracts where useful and notes when it does.

---

## §2 Results — baseline benchmark

Source: `benchmark_results_20260428.json` (Sub-step 1.1) joined with
`phase1_supplemental.json` (Sub-step 1.3 D1 flag counts). Wall-clock
reported is the median of three runs with `tracemalloc` active and
stdout redirected.

### §2.1 Convergence and timing

| Case | Algorithm | nit | inner nit | wall (ms) | peak mem (MB) | final res | final λ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Q7_baseline` | Multi | 5 | — | 20.4 | 0.05 | 5.89e-12 | — | OK |
| `Q7_baseline` | HONI_exact | 4 | 50 | 129.3 | 0.07 | 4.60e-20 | 10.756234 | OK |
| `Q7_baseline` | HONI_inexact | 4 | 45 | 104.3 | 0.07 | 9.91e-16 | 10.756234 | OK |
| `Q7_baseline` | NNI_spsolve | 16 | — | 42.0 | 0.05 | 4.48e-12 | 10.756234 | OK |
| `Q7_large` | Multi | 5 | — | 11.6 | 0.19 | 2.48e-13 | — | OK |
| `Q7_large` | HONI_exact | 6 | 233 | 898.7 | 0.96 | 1.87e-14 | **9.333980** | ⚠ silent wrong λ |
| `Q7_large` | HONI_inexact | 77 | 7229 | 26 405.2 | 0.96 | 2.31e-15 | 10.756272 | OK (slow) |
| `Q7_large` | NNI_spsolve | 13 | — | 35.4 | 0.24 | 1.65e-15 | 10.756272 | OK |
| `Q7_small` | Multi | 5 | — | 11.2 | 0.02 | 2.56e-16 | — | OK |
| `Q7_small` | HONI_exact | 6 | 351 | 1240.6 | 0.03 | 2.25e-12 | **8.756218** | ⚠ silent wrong λ |
| `Q7_small` | HONI_inexact | — | — | — | — | — | — | ⛔ AssertionError |
| `Q7_small` | NNI_spsolve | — | — | — | — | — | — | ⛔ AssertionError |
| `Q7_seed_alt1` | Multi | 5 | — | 10.9 | 0.04 | 6.89e-14 | — | OK |
| `Q7_seed_alt1` | HONI_exact | 4 | 64 | 168.5 | 0.07 | 2.19e-19 | 10.955004 | OK |
| `Q7_seed_alt1` | HONI_inexact | 4 | 57 | 146.7 | 0.07 | 6.49e-16 | 10.955004 | OK |
| `Q7_seed_alt1` | NNI_spsolve | 15 | — | 36.3 | 0.05 | 4.61e-13 | 10.955004 | OK |
| `Q7_seed_alt2` | Multi | 5 | — | 10.0 | 0.04 | 1.04e-13 | — | OK |
| `Q7_seed_alt2` | HONI_exact | 4 | 90 | 242.7 | 0.07 | 6.00e-17 | 10.989793 | OK |
| `Q7_seed_alt2` | HONI_inexact | 4 | 134 | 405.2 | 0.07 | 2.18e-14 | 10.989793 | OK |
| `Q7_seed_alt2` | NNI_spsolve | 23 | — | 65.6 | 0.05 | 4.18e-11 | 10.989793 | OK |

Two correctness incidents on `Q7_large` and `Q7_small` for HONI_exact
silently report a wrong eigenvalue (residual at machine epsilon, but
λ off by ~13% and ~19% respectively). Two failures on `Q7_small` for
HONI_inexact and NNI_spsolve hit the in-algorithm `assert` that the
residual buffer's pre-fill upper bound holds — this fires when
`lambda_L < 0` empirically, indicating the random AA does not satisfy
the clean M-tensor assumption at this scale.

### §2.2 F1 detection flags (HONI exact + inexact across all cases)

| Case | Branch | final λ | halving warns | `flag.inner_trap` | `flag.lambda_nonmonotone` |
|---|---|---:|---:|---:|---:|
| `Q7_baseline` | exact | 10.756234 | 0 | 0 | 0 |
| `Q7_baseline` | inexact | 10.756234 | 0 | 0 | 0 |
| `Q7_large` | exact | **9.333980** | 99 | **2** | **2** |
| `Q7_large` | inexact | 10.756272 | 0 | **71** | 0 |
| `Q7_small` | exact | **8.756218** | 294 | **3** | **2** |
| `Q7_small` | inexact | (FAIL) | 99 | **1** | **6** |
| `Q7_seed_alt1` | exact | 10.955004 | 0 | 0 | 0 |
| `Q7_seed_alt1` | inexact | 10.955004 | 0 | 0 | 0 |
| `Q7_seed_alt2` | exact | 10.989793 | 0 | 0 | 0 |
| `Q7_seed_alt2` | inexact | 10.989793 | 2 | **1** | 0 |

The two flag definitions are doing exactly the work they were designed
to. Every silent-wrong-λ row has both flags positive. Every
correct-λ row has zero flags **except** `Q7_large` HONI_inexact (71
inner traps but a correct final λ — inexact's adaptive inner tolerance
weathers the inner Multi traps without contaminating the eigenvalue
update) and `Q7_seed_alt2` HONI_inexact (1 inner trap, correct λ).
This is the operational basis for the detection rule offered in §5: a
positive `flag.honi.lambda_nonmonotone` together with a positive
`flag.honi.inner_trap` is the precise signature of a silent failure.

### §2.3 Scaling with `n` (HONI exact and NNI spsolve)

The three Q7 size points (n = 10, 20, 50, all seed = 42) give a coarse
scaling read.

| n | HONI_exact wall (ms) | HONI_exact final λ | NNI_spsolve wall (ms) | NNI_spsolve final λ |
|---:|---:|---:|---:|---:|
| 10 | 1240.6 | 8.756218 (wrong) | (FAIL) | (FAIL) |
| 20 | 129.3 | 10.756234 (correct) | 42.0 | 10.756234 |
| 50 | 898.7 | 9.333980 (wrong) | 35.4 | 10.756272 |

NNI_spsolve wall-clock is essentially flat across n = 20 → 50 (42 → 35
ms — within timing noise). HONI_exact's wall is dominated by inner
Multi traps when `n` is small or large; the U-shape (1241 ms / 129 ms /
899 ms) shows that the well-conditioned middle is a narrow regime.

---

## §3 Time breakdown — profiling

Source: `phase1_supplemental.json` D2. Times are inclusive — when a
parent block contains a child block both totals see the same
wall-clock. The "exclusive" estimate listed for each parent subtracts
the child totals reported in the same row.

### §3.1 Q7_baseline (n = 20, seed = 42)

Single profiled run per algorithm; categories sorted by total.

#### Multi
| Category | count | total (ms) |
|---|---:|---:|
| `multi.linear_solve` | 5 | **3.94** |
| `multi.tensor_contract` | 6 | 0.10 |
| `multi.residual_check` | 6 | 0.02 |
| `multi.halving_search` | 5 | 0.02 |

`linear_solve` (sparse Jacobian + `spsolve`) is essentially the entire
algorithm. The other three categories together account for ~3% of
wall.

#### HONI exact
| Category | count | total (ms) | Note |
|---|---:|---:|---|
| `honi.outer_iter` | 4 | **57.95** | parent (full outer body) |
| `honi.inner_multi_call` | 4 | 57.81 | child of outer_iter |
| `multi.linear_solve` | 35 | 35.20 | inside Multi inside HONI |
| `multi.halving_search` | 35 | 17.18 | |
| `multi.tensor_contract` | 309 | 5.04 | |
| `multi.residual_check` | 309 | 0.71 | |
| `honi.tensor_contract` | 1 | 0.05 | init only (exact branch has none in loop) |
| `honi.rayleigh_extract` | 4 | 0.02 | |
| `honi.lambda_update` | 4 | 0.01 | |

Inner Multi accounts for **99.7%** of HONI's wall (`inner_multi_call`
57.81 / `outer_iter` 57.95). Inside that, `multi.linear_solve` is
60.9% and `multi.halving_search` is 29.7%.

#### HONI inexact
| Category | count | total (ms) | Note |
|---|---:|---:|---|
| `honi.outer_iter` | 4 | **45.04** | parent |
| `honi.inner_multi_call` | 4 | 44.84 | child |
| `multi.linear_solve` | 27 | 26.46 | |
| `multi.tensor_contract` | 247 | 13.72 | |
| `multi.halving_search` | 27 | 4.04 | |
| `multi.residual_check` | 247 | 0.69 | |
| `honi.rayleigh_extract` | 4 | 0.10 | inexact uses tpv each iter |
| `honi.tensor_contract` | 5 | 0.10 | init + 4 in-loop |
| `honi.lambda_update` | 4 | 0.005 | |

On Q7_baseline the inexact branch is **faster** than exact (45 ms vs
58 ms) — HONI_inexact recovered the converged eigenvector in 4 outer
iters with milder halving. (See §3.2 for what changes at n = 50.)

#### NNI canonical (spsolve, halving=False)
| Category | count | total (ms) |
|---|---:|---:|
| `nni.linear_solve` | 16 | **17.50** |
| `nni.tensor_contract` | 17 | 0.49 |
| `nni.rayleigh_quotient` | 16 | 0.10 |
| `nni.bracket_update` | 16 | 0.01 |

`linear_solve` is essentially the entire NNI cost. 16 outer iters at
~1.1 ms per linear solve.

#### NNI_ha (spsolve, halving=True)
| Category | count | total (ms) |
|---|---:|---:|
| `nni.linear_solve` | 24 | **24.18** |
| `nni.tensor_contract` | 43 | 1.10 |
| `nni.halving_inner` | 24 | 0.55 |
| `nni.rayleigh_quotient` | 42 | 0.27 |
| `nni.bracket_update` | 24 | 0.01 |

NNI_ha takes 24 outer iters vs. NNI canonical's 16 (halving extends
trajectory) and is correspondingly slower at ~25 ms vs. ~18 ms here.
The Sub-step 1.2 finding F-1.2.3 ("NNI_ha 2× faster than canonical")
was a measurement artefact — see §4 Finding #2 for the multi-case
confirmation.

### §3.2 Q7_large (n = 50, seed = 42) — bottleneck shift

| Algorithm | wall (ms) | dominant category | secondary |
|---|---:|---|---|
| Multi | 7.0 | `multi.linear_solve` (~95%) | `multi.tensor_contract` |
| HONI exact | 397.8 | `multi.linear_solve` (~38%) | `multi.halving_search` (~50%) ⚠ |
| HONI inexact | **12 196** | `multi.halving_search` (~75%) | `multi.tensor_contract` (~20%) |
| NNI spsolve | 15.1 | `nni.linear_solve` (~99%) | — |
| NNI_ha | 23.9 | `nni.linear_solve` (~95%) | `nni.tensor_contract` |

The crucial shift between Q7_baseline (n = 20) and Q7_large (n = 50):

- **HONI inexact balloons from 45 ms to 12.2 seconds** — a 270×
  slowdown for 2.5× more dimensions. Profiling pins this on
  `multi.halving_search`: the dynamic inner-tolerance schedule pushes
  the inner Multi solver into trap-and-halve territory in 71 of 77
  outer iterations (see `flag.inner_trap = 71` in §2.2).
- **HONI exact becomes 7× slower** (130 ms → 898 ms in this run; the
  398 ms profiled-run number above is faster than the 1.1 baseline due
  to single-run vs. median-of-three) and silently locks at the wrong
  eigenvalue (§4 Finding #1).
- **Multi standalone barely budges** (4 ms → 7 ms) — the multilinear
  solver scales linearly with `n` because the unfolded A has
  `O(n · n^(m-1))` non-zeros and the sparse Jacobian solve is
  bandwidth-bound.
- **NNI spsolve actually gets *slightly faster* at n = 50** (18 ms →
  15 ms; 16 outer iters → 13 outer iters). Different random tensors
  converge in different iter counts; for this seed the larger AA
  yields a faster single-step contraction. NNI_ha follows the same
  pattern.

---

## §4 Research findings

### Finding #1 — HONI_exact silent wrong-eigenvalue lock at n ≈ 50

Documented in detail at:
- Research note: `docs/papers/honi_exact_silent_failure_n50.md`
- Memory: `memory/feedback_honi_exact_silent_lock.md`
- Repro: `python/benchmarks/scripts/repro_f1_honi_exact_silent_failure.py`
- Trace: `python/benchmarks/results/f1_analysis/f1_trajectory.json`

In a 20-trial multi-restart sweep on Q7_large, NNI converges to the
correct λ ≈ 10.756 in 20/20 trials while HONI_exact converges in 5/20
and silently locks onto 15 distinct non-spectral fixed points in the
remaining 15 trials. The mechanism is a composition of two existing
fragility modes (Multi halving + HONI shift-invert near-singularity):
incremental λ update overshoots, Multi traps, Rayleigh extraction on
the trapped y produces an O(1) λ jump, and the algorithm declares
convergence at a non-eigenpair fixed point of the iteration map.

The detection rule from §2.2 — both `flag.honi.inner_trap > 0` AND
`flag.honi.lambda_nonmonotone > 0` — is a clean discriminator: it
fires on the two known wrong-λ rows (Q7_large exact and Q7_small
exact) and stays silent on every correct-λ row. It does not fire on
`Q7_large` HONI_inexact (71 inner traps without nonmono) or
`Q7_seed_alt2` HONI_inexact (1 inner trap), reflecting the empirical
fact that those runs converge correctly.

### Finding #2 — NNI_ha vs NNI canonical (Sub-step 1.2 F-1.2.3 demoted)

Sub-step 1.2 reported a single-run observation that NNI_ha was ~2×
faster than NNI canonical on Q7_baseline (25 ms vs. 53 ms). Sub-step
1.3 D3 reproduced this comparison across all five test cases with
three runs each, profiling **disabled** for clean wall-clock:

| Case | NNI canonical median (ms) | NNI_ha median (ms) | Verdict |
|---|---:|---:|---|
| `Q7_baseline` | 16.89 | 25.74 | canonical 1.52× faster |
| `Q7_large` | 14.97 | 21.17 | canonical 1.41× faster |
| `Q7_small` | (FAIL) | 201.65 | canonical assertion-fails; halving works |
| `Q7_seed_alt1` | 17.19 | 248.75 | canonical **14.47× faster** |
| `Q7_seed_alt2` | 23.73 | 219.15 | canonical **9.24× faster** |

Verdict: F-1.2.3 was a measurement artefact, not algorithmic.
Profiling adds ~5–10% wall-clock overhead, but more importantly the
1.2 single-run timing of NNI canonical was inflated by a one-off
SciPy SuperLU warm-up cost (Sub-step 1.2 finding F-1.2.4 had already
flagged the first `nni.linear_solve` taking 19 ms vs. a 1 ms median).
With the warm-up amortised over three runs, NNI canonical is
**generally faster** than NNI_ha across the test cases — sometimes by
an order of magnitude (`Q7_seed_alt1`, `Q7_seed_alt2`).

The interesting residual is `Q7_small`: NNI canonical fires the
in-algorithm assertion (`lambda_L < 0`) and cannot complete, while
NNI_ha's halving keeps the iteration in a region where `lambda_L`
stays non-negative and converges in ~200 ms. So the two variants are
not strictly ordered: canonical is faster when it works, but halving
extends the regime where the algorithm produces *any* answer. This
trade-off is a candidate Phase 2 discussion: either tighten the input
preconditions on canonical, or use halving as a fallback when
canonical's assertion fires.

**Methodology note**: the F-1.2.3 reversal is itself a finding.
Single-run wall-clock measurements are unreliable at the millisecond
scale because of warm-up effects (scipy SuperLU first-call factory
init was the specific culprit, ~19× slower than steady-state).
Sub-step 1.3 D3 used 3-run median and reversed the conclusion. Going
forward all performance comparisons in Phase 2+ will use 3-run median
minimum and exclude cold-start outliers.

---

## §5 Phase 2 optimization candidates

Five candidates emerge from §3 + §4. They are listed in no particular
priority order; the trade-offs are what matter, not a single ranking.
Ratings are author judgement, not measured.

### Candidate A — HONI silent-failure auto-detection + NNI fallback
- **Bottleneck**: not speed but **correctness** — HONI_exact returns a
  wrong eigenvalue 75% of the time on n ≈ 50 random M-tensors (Sub-step
  1.1.5 §6).
- **Mechanism**: wrap `honi(linear_solver="exact", ...)` so that on
  return, if both F1 detection flags are positive, automatically rerun
  with NNI canonical and either return the NNI result or raise.
- **Expected gain**: 0% on speed; **75% → 100%** on correctness for
  affected regime. NNI cross-check costs ~35 ms on Q7_large.
- **Implementation cost**: small. New thin wrapper at the HONI call
  site; the flags themselves are already in place.
- **Risk**: low. NNI is the validated reference for the eigenvalue
  (Sub-step 1.1.5 §6 spectrum sweep was 20/20).

### Candidate B — Reduce Multi halving cap inside HONI inexact
- **Bottleneck**: §3.2 — `multi.halving_search` is 75% of HONI inexact
  wall on Q7_large. The current `_BUF − 1 = 99` cap lets the inner
  Newton spin for tens of milliseconds per outer iter even when it
  cannot make progress.
- **Mechanism**: lower the inner cap to ~25 specifically when the
  outer caller is HONI_inexact and the inner residual has not improved
  in K consecutive halvings. Either parameterise Multi's `_BUF` or
  short-circuit from inside HONI.
- **Expected gain**: 30–60% wall-clock on HONI_inexact at n = 50; less
  on smaller n where halving is rare.
- **Implementation cost**: medium. Parity with MATLAB will need to be
  re-validated under the new cap, since MATLAB's inner loop runs the
  full 100. The existing parity tests use Q5-style cases that do not
  exercise the cap, so a targeted parity re-baseline may be needed.
- **Risk**: medium. Cap-shrinking trades inner accuracy for speed;
  the wrong cap could cause HONI_inexact to fail rather than just slow
  down.

### Candidate C — Inner-Multi `linear_solve` acceleration
- **Bottleneck**: Multi's `linear_solve` is 99% of standalone Multi
  cost and 60% of HONI exact's inner Multi cost (§3.1). It is the
  single biggest item in any HONI run.
- **Mechanism**: candidates include reusing the symbolic LU
  factorisation across consecutive Newton iterations (the Jacobian
  sparsity pattern is constant), trying iterative solvers (Bi-CGSTAB
  or GMRES with a Jacobi preconditioner), or batching multiple shifted
  systems.
- **Expected gain**: 30–50% if symbolic reuse works; harder to predict
  for iterative solvers.
- **Implementation cost**: high. Symbolic reuse needs a different
  scipy API path (`splu` rather than `spsolve`) and careful state
  management. Iterative solvers need preconditioner design and a new
  parity strategy (cannot expect bit-identical to MATLAB direct
  solver).
- **Risk**: medium-high. Numerical regressions are the main concern —
  parity tests must not silently relax to accommodate iterative solver
  noise.

### Candidate D — Cold-start mitigation for the first `spsolve`
- **Bottleneck**: Sub-step 1.2 F-1.2.4 — the first call to
  `scipy.sparse.linalg.spsolve` in a process is ~19× slower than the
  median (factory cost / SuiteSparse warm-up).
- **Mechanism**: at module import, call `spsolve` once on a tiny dummy
  sparse system to amortise the cost.
- **Expected gain**: trims ~15 ms from the first NNI / HONI run per
  process; makes single-run benchmark numbers more stable.
- **Implementation cost**: trivial (3 lines).
- **Risk**: negligible. Side-effect-free import-time computation.

### Candidate E — Replace canonical NNI's hard `assert` with a graceful fallback
- **Bottleneck**: §2 — Q7_small NNI_spsolve and HONI_inexact both die
  on the in-algorithm `assert` for the residual buffer's pre-fill upper
  bound. Conceptually this asserts the input is a clean M-tensor; for
  random Q7-style inputs at small `n` this is not always true.
- **Mechanism**: replace the `assert` with a structured warning + a
  fallback to NNI_ha (which §4 Finding #2 shows succeeds on Q7_small
  in 200 ms).
- **Expected gain**: 0% speed; converts current hard failures into
  correct-but-slower answers.
- **Implementation cost**: small. The trick is preserving the original
  diagnostic value of the `assert` for genuinely bad inputs (do we
  fall back, or do we surface? user judgement).
- **Risk**: low. Structurally similar to Candidate A.

### Phase 2 execution order

Phase 2 will execute in tiers:

- **Tier 1 (correctness)**: A + E executed first, restoring
  trustworthy output before any performance optimization.
- **Tier 2 (free lunch)**: D will be folded in alongside Tier 1.
- **Tier 3 (performance)**: C and B will follow after Tier 1 detection
  is stable. C has highest leverage but highest parity risk; B
  interacts with silent-failure detection signals and must come after
  the detection mechanism is hardened.

Rationale: correctness must precede performance optimization. The
detection-based safety net (A) is also a prerequisite for B, since
B's halving cap shrink may reduce the very signals A relies on.

---

## §6 Caveats and limitations

- **Test cases are all Q7-style**. No Q1 (halving fragility) or Q3
  (Rayleigh noise floor) ill-conditioned builders exist in this
  codebase. Phase 2 work that depends on stress-testing those regimes
  needs new builders first.
- **`tracemalloc` plus `redirect_stdout` adds overhead** to Sub-step
  1.1's wall-clock numbers (estimated 10–20% from prior tracemalloc
  measurements). Sub-step 1.3 D3 was run with both disabled to get a
  cleaner number for the NNI_ha vs canonical comparison.
- **Profiling enabled adds ~5–10% wall-clock overhead** on top of the
  no-op `with` block cost. The Sub-step 1.2 breakdown numbers and the
  Sub-step 1.3 D2 breakdown numbers are *with* this overhead;
  Sub-step 1.3 D3 numbers are *without* it. Do not compare across
  these directly.
- **F1 detection thresholds are empirical**. The "1% relative jump
  after iter ≥ 3" rule for `flag.honi.lambda_nonmonotone` and the
  "`chit_py >= 99`" rule for `flag.honi.inner_trap` come from the
  specific F1 trajectory in `f1_trajectory.json`. They have not been
  derived from a theoretical condition; the rule may need adjustment
  for other algorithm families or other test-case structures.
- **Single-run breakdown reports** (Sub-step 1.2 and 1.3 D2) can be
  unstable for the smallest categories. The first `linear_solve` cost
  and other warm-up artefacts can dominate single-run totals. The
  median-of-three numbers in §2 are more reliable.
- **All wall-clock numbers are on a single machine** (Apple M1 Ultra).
  Linear solve performance on x86 with MKL is likely meaningfully
  different; relative orderings should hold but absolute numbers
  should not be cited cross-platform without re-measurement.
- **Q7_small failures may not be Q7's fault**. The in-algorithm
  asserts indicate the random Q7 construction at n = 10 occasionally
  produces a tensor that violates the algorithm's `lambda_L >= 0`
  assumption. A more disciplined Q7 builder that rejects bad seeds
  would simplify the picture.

---

## §7 Phase 2 outcomes (Sub-steps 2.1, 2.2, 2.3)

Phase 2 Tier 1 + 2 prioritised correctness over performance. Three
sub-steps shipped: cold-start mitigation (Tier 2 D), graceful fallback
for the NNI canonical assertion (Tier 1 E), and silent-failure
auto-detection for HONI exact (Tier 1 A).

### §7.1 Sub-step 2.1 — Cold-start spsolve mitigation (Tier 2 D)

| | NNI Q7_baseline `linear_solve` |
|---|---|
| BEFORE | median 1010.9 μs / max 4905.1 μs / **ratio 4.9×** |
| AFTER | median 947.0 μs / max 1141.2 μs / **ratio 1.2×** |

The first call to `scipy.sparse.linalg.spsolve` triggers a SuperLU
factory initialisation that runs ~5–19× slower than the steady state
(Phase 1 §3 D2 finding F-1.2.4). Sub-step 2.1 calls `spsolve` and
`gmres` on a 1×1 dummy system at `tensor_utils` import time so the
first run inside an algorithm sees the warmed-up code path. Cost is a
one-time ~15 ms at import; runtime overhead is zero. This eliminates
the root cause of the F-1.2.3 measurement artefact (single-run NNI
canonical timing was 53 ms in the 1.2 profiled run because the first
linear solve carried the warm-up; in 1.3 D3 with 3-run median + warm
state the canonical median was 17 ms — a clean 3× difference that had
nothing to do with the algorithm).

### §7.2 Sub-step 2.2 — NNI canonical graceful fallback (Tier 1 E)

| | Q7_small (n = 10) NNI canonical |
|---|---|
| BEFORE | ⛔ `AssertionError: res upper-bound violation` |
| AFTER | ✓ λ = 10.756224 + `RuntimeWarning` (~875 ms via fallback) |

The post-loop assertion `np.all(res[:nit + 1] <= 1.0 + 1e-10)` fires
when `lambda_L` goes negative during canonical iteration, indicating
the input `AA` is not a clean M-tensor at this scale. Sub-step 2.2
turns the bare `assert` into a conditional check: if the violation
fires while `halving=False`, warn and recursively call
`nni(..., halving=True, initial_vector=initial_vector)`. The recursion
guard is structural — `halving=True` cannot re-enter the canonical
fallback branch, so even if the halving variant also violates the
upper bound the original `AssertionError` simply surfaces. Healthy
cases never enter the fallback path and remain bit-identical to
Phase 1.

The implementation is conditional + recursive `return` rather than a
`try`/`except`, which avoids using exceptions for control flow and
prevents accidentally catching unrelated `AssertionError`s.

### §7.3 Sub-step 2.3 — HONI_exact silent-failure auto-detection (Tier 1 A)

| | Multi-restart spectrum on Q7_large (20 random initial vectors) |
|---|---|
| BEFORE | 5/20 correct (λ ≈ 10.756); 15/20 silent fail at 15 distinct non-spectral fixed points (9.27–9.75) |
| AFTER | **20/20 correct** (15 trials fall back to NNI on detection) |

The Phase 1 instrumentation flags (`flag.honi.inner_trap` and
`flag.honi.lambda_nonmonotone`) discriminated the 10 baseline entries
perfectly: both fire on silent-failure rows, neither fires on
correct-output rows. Sub-step 2.3 repurposes those flag conditions as
an active fallback trigger — when both fire on the exact branch, warn
and re-solve via NNI canonical. NNI's spectrum sweep was 20/20
correct on Q7_large in Phase 1 §1.1.5, so it is the natural recovery
target.

The fallback is implemented through `_honi_fallback_to_nni`, a private
helper at the top of `tensor_utils.py` that calls `nni(...,
halving=False, ...)` and reshapes the return tuple into HONI's
6-tuple shape (or 7-tuple when `record_history=True`). The third
position differs (HONI: `total_inner_nit`; NNI: `lambda_L`); the
helper returns `0` there because NNI is single-layer and has no inner
Newton solver. HONI-only history fields are filled with zeros / NaN
of the correct shape so callers expecting HONI's history layout still
receive valid arrays.

The inexact branch is intentionally **not** auto-fallback'd. Phase 1
§2.2 showed that on Q7_large, HONI inexact fires `flag.inner_trap`
71 times across its 77 outer iterations yet still converges to the
correct λ. The `flag.lambda_nonmonotone` does not fire there because
inexact's adaptive inner-tolerance schedule prevents the O(1) λ_U
jumps. Auto-falling back inexact would be a false-positive triggered
by inner_trap alone (which fires on its own elsewhere, e.g.
`Q7_seed_alt2 HONI_inexact` 1 trap).

A side effect worth noting: silent failure also wastes wall-clock.
Q7_large HONI exact previously ran 6 outer iters in 899 ms producing
the wrong answer; the fallback path runs NNI in 13 outer iters in
440 ms producing the right answer. Catching the silent failure made
the routine **51% faster as well as correct**.

Stacked fallbacks compose cleanly. Q7_small HONI exact triggers the
2.3 detection (HONI silent → NNI canonical), and the NNI canonical
call then triggers the 2.2 condition (NNI canonical → NNI_ha). The
user sees two stacked `RuntimeWarning`s in order, and the final
return is from NNI_ha. There is no infinite recursion because NNI
never re-enters HONI and the 2.2 fallback only fires when
`halving=False`.

### §7.4 Phase 2 final benchmark snapshot

20 entries on the same 5 cases × 4 algorithms grid as Phase 1 §2.1.
Phase 2 final wall-clock and λ from the post-2.3 run (3-run median).

| Case | Algorithm | nit | wall (ms) | final λ | Status |
|---|---|---:|---:|---:|---|
| `Q7_baseline` | Multi | 5 | 10.4 | — | ✓ |
| `Q7_baseline` | HONI_exact | 4 | 117.8 | 10.756234 | ✓ |
| `Q7_baseline` | HONI_inexact | 4 | 106.9 | 10.756234 | ✓ |
| `Q7_baseline` | NNI_spsolve | 16 | 39.1 | 10.756234 | ✓ |
| `Q7_large` | Multi | 5 | 11.8 | — | ✓ |
| `Q7_large` | HONI_exact | 13 | **439.8** | **10.756272** | ✓ via fallback (was 899 ms / 9.333980 ⚠) |
| `Q7_large` | HONI_inexact | 77 | 23 950.7 | 10.756272 | ✓ (slow, no fallback by design) |
| `Q7_large` | NNI_spsolve | 13 | 35.1 | 10.756272 | ✓ |
| `Q7_small` | Multi | 5 | 10.3 | — | ✓ |
| `Q7_small` | HONI_exact | 199 | **1706.9** | **10.756224** | ✓ via stacked fallback (was 1241 ms / 8.756218 ⚠) |
| `Q7_small` | HONI_inexact | — | — | — | ⛔ Multi-internal assert (out of Tier 1 scope) |
| `Q7_small` | NNI_spsolve | 199 | **904.6** | **10.756224** | ✓ via 2.2 fallback (was crash) |
| `Q7_seed_alt1` | Multi | 5 | 10.3 | — | ✓ |
| `Q7_seed_alt1` | HONI_exact | 4 | 150.7 | 10.955004 | ✓ |
| `Q7_seed_alt1` | HONI_inexact | 4 | 135.7 | 10.955004 | ✓ |
| `Q7_seed_alt1` | NNI_spsolve | 15 | 35.9 | 10.955004 | ✓ |
| `Q7_seed_alt2` | Multi | 5 | 10.9 | — | ✓ |
| `Q7_seed_alt2` | HONI_exact | 4 | 222.2 | 10.989793 | ✓ |
| `Q7_seed_alt2` | HONI_inexact | 4 | 376.1 | 10.989793 | ✓ |
| `Q7_seed_alt2` | NNI_spsolve | 23 | 56.0 | 10.989793 | ✓ |

19 of 20 entries are now correct (Phase 1 had 14 correct, 2 silent
wrong, 4 hard failure / silent — depending on count). The only
remaining ⛔ is `Q7_small HONI_inexact`, which fires the **Multi**
internal assertion (line 598, `max res > na+nb`) — a different
mechanism from NNI's line 1334 assertion that 2.2 covered. That
fallback is out of Tier 1 scope and is listed below.

### §7.5 What was not done (Tier 3 + leftover)

- **Candidate C — Inner-Multi `linear_solve` acceleration**: highest
  leverage but highest parity risk. Deferred until the Tier 1 A
  detection mechanism has had an observation period in production
  use; if the detection rule turns out to false-positive on some
  regime, fixing C without that signal would risk reverting to silent
  wrong answers.
- **Candidate B — Multi halving cap shrinking inside HONI inexact**:
  the cap shrink would directly reduce the `chit_py >= 99` signal
  that 2.3 relies on; running it before Tier 1 A is observationally
  established would be premature.
Phase 2 closes here. Tier 3 candidates remain on the table for a
later phase once the Phase 2 safety net has been observed in real
use.

### §7.6 Sub-step 2.6 — Demo UI warning capture (Day 9)

The Sub-step 2.2 / 2.3 fallback notices originally surfaced only on
stderr / Streamlit Cloud's log stream — the demo UI showed the
recovered (correct) result with no indication that a fallback had
fired. Sub-step 2.6 makes those notices visible to demo visitors.

`algorithms.py` gains two helpers:

- `_capture_algorithm_warnings()` — context manager that wraps each
  algorithm call in `warnings.catch_warnings(record=True)` and yields
  the captured list to the caller. Stashed in `st.session_state`
  under a `"warnings"` key.
- `_render_warnings(warning_records)` — displays each captured
  warning as `st.warning`. Identical messages are deduplicated and
  shown once with a "(this warning fired N times)" suffix; this is
  what keeps the UI clean when Sub-step 2.7's `Q7_small HONI_inexact`
  path emits ~189 Multi residual warnings collapsed into a single
  notice with the count.

All six renderers (`render_multi`, `render_honi`, `render_nni`,
`render_hni_vs_nni`, `render_eigenvalue_compare`,
`render_multilinear_compare`) follow the same pattern: wrap the
algorithm call(s) in `with _capture_algorithm_warnings() as caught`,
stash `list(caught)` in session_state, and call
`_render_warnings(result.get("warnings"))` near the top of the
output column. Healthy cases (zero warnings) render nothing — the UI
stays clean.

### §7.7 Sub-step 2.7 — Multi internal assert + HONI inexact graceful fallback (Day 9)

The last `⛔` row in §7.4 was `Q7_small HONI_inexact`, which raised
`AssertionError` from inside Multi (`tensor_utils.py:598`) before
HONI's outer loop had a chance to run. Sub-step 2.7 takes this from
crash to recovered output.

**(c) Multi `graceful` keyword-only kwarg**. The bare assert at
line 598 becomes:

```python
upper_bound_violated = bool(np.any(res[:nit + 1] > na + nb + 1e-10))
if upper_bound_violated and graceful:
    warnings.warn(...)
else:
    assert ...   # original parity-safe assert kept for graceful=False
```

Default `graceful=False` keeps standalone Multi and the parity tests
bit-identical to Phase 1. HONI passes `graceful=True` at both inner
multi() call sites (`record_inner_history` True / False branches).

**(d.1) HONI inexact graceful fallback**. With Multi no longer
crashing, Q7_small HONI_inexact ran its outer loop to maxit=200 and
then tripped HONI's own residual upper-bound assert (line 1056,
`res > 1.0`). Sub-step 2.7 mirrors Sub-step 2.3 for the inexact
branch: when the violation fires while `linear_solver == "inexact"`,
warn and fall back to NNI canonical via the existing
`_honi_fallback_to_nni` helper. The exact branch keeps its bare
assert because Sub-step 2.3's detection-based fallback already
covers it; reaching the assert on exact would mean something
genuinely unexpected occurred and surfacing the error is correct.

**Stacked fallback chain on Q7_small**: HONI inexact → NNI canonical
→ NNI_ha. All three layers fire warnings (HONI inexact's, NNI
canonical's from Sub-step 2.2, plus 189 Multi graceful warnings from
the inner Multi calls). The 189 Multi warnings come in three
distinct text variants (`max res = 2.130e+10`, `2.636e+10`,
`2.637e+10` — different residual scales per Multi call); after
Sub-step 2.6's dedup the user sees five notices total, not 191. The
final result is `λ ≈ 10.756224`, matching the explicit
`halving=True` baseline from Phase 1 §1.3 D3.

**Effect on the §7.4 benchmark snapshot**: the previously `⛔`
`Q7_small HONI_inexact` row is now `✓ nit=199 / 2344 ms /
λ=10.756224`. All 20 entries in the 5-case × 4-algorithm grid
return a usable answer.
