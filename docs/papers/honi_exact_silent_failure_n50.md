# HONI_exact silent wrong-eigenvalue failure on Q7-style M-tensors at n = 50

**Status**: research note (draft, 2026-04-28)
**Config**: n = 50, m = 3, build_q7_tensor seed = 42, tol = 1e-10, maxit = 200
**Trigger**: Phase 1 Sub-step 1.1 baseline benchmark (finding F1)
**Repro**: `python/benchmarks/scripts/repro_f1_honi_exact_silent_failure.py`
**Raw data**: `python/benchmarks/results/f1_analysis/f1_trajectory.json`

## §1 Phenomenon

On the Q7_large benchmark case (n = 50, m = 3, seed = 42), HONI with
`linear_solver="exact"` returns

  - λ = **9.333980**
  - residual = 1.87 × 10⁻¹⁴
  - outer iterations = 6
  - inner iterations (Σ Multi calls) = 233
  - halving warnings printed by Multi = 99

Two algorithms run on the **same tensor and same initial vector**
both converge to a different eigenvalue:

  - HONI inexact : λ = 10.756272
  - NNI spsolve : λ = 10.756272
  - NNI halving : λ = 10.756272

HONI_exact's residual sits at machine epsilon, so by every internal
signal the routine is "converged" — but the eigenvalue is wrong by
about 1.42 in absolute terms (≈ 13% relative). The failure is silent:
no exception, no convergence warning, only the inner Multi noise.

## §2 Reproduction

```python
from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from tensor_utils import honi
AA, x0 = build_q7_tensor(n=50, m=3, rng_seed=42)
lam, x, *_ = honi(AA, m=3, tol=1e-10,
                  linear_solver="exact", maxit=200,
                  initial_vector=x0)
# lam ≈ 9.3339801587  (silent wrong answer)
```

The full trajectory under `record_history=True` is reproduced below;
each row is one outer iteration, with the inner Multi statistics from
`record_inner_history=True` aligned alongside.

| outer | λ_U at start | residual | hal_per_outer | inner_nit |
|---:|---:|---:|---:|---:|
| 0 | 11.572213 | 8.365 × 10⁻¹ | 0 | 0 |
| 1 | 10.902251 | 7.975 × 10⁻¹ | 0 | 6 |
| 2 | 10.767959 | 7.035 × 10⁻¹ | 0 | 12 |
| 3 | 10.756739 | 2.458 × 10⁻¹ | 3 | 19 |
| 4 | 10.756279 | 6.161 × 10⁻³ | 34 | 33 |
| 5 | 10.333980 | 3.372 × 10⁻² | **1697** | 133 |
| 6 | 9.333980 | 1.870 × 10⁻¹⁴ | **2970** | 233 |

Two structural features stand out:

  1. **Two large discrete λ jumps**: 10.756279 → 10.333980 → 9.333980,
      each almost exactly 0.42 and 1.00. The trajectory leaves the
      neighborhood of the true eigenvalue and lands on what looks like
      a quantized fixed point.
  2. **Halving counts explode**: outer iterations 5 and 6 each trigger
      around 2 000 halving steps inside Multi (vs. ≤ 34 in earlier
      iterations). The inner Newton has clearly stopped making progress.

## §3 Comparison with HONI_inexact, NNI canonical, and NNI_ha

The same `(AA, x0)` is fed to three other algorithms.

| algorithm | final λ | residual | outer nit | halving warnings |
|---|---:|---:|---:|---:|
| HONI_exact | 9.333980 | 1.87 × 10⁻¹⁴ | 6 | 99 |
| HONI_inexact | 10.756272 | 2.31 × 10⁻¹⁵ | 77 | 0 |
| NNI canonical (halving=False) | 10.756272 | 1.65 × 10⁻¹⁵ | 13 | 0 |
| NNI_ha (halving=True) | 10.756272 | 2.13 × 10⁻¹³ | 16 | 0 |

Three observations:

  - All three "control" algorithms agree on λ = 10.756272 to 10⁻⁶
    relative, identifying it as the true largest H-eigenvalue.
  - None of the three triggers a single Multi halving warning. The
    inner Newton trap is specific to HONI_exact's overshoot trajectory.
  - HONI_inexact takes 77 outer iterations to settle (vs. 6 for
    HONI_exact). The inexact branch's adaptive inner tolerance pays off
    with reliability at the cost of speed: ~26 s wall-clock on this
    case (Sub-step 1.1).

## §4 Root-cause hypothesis

The failure is the **combined manifestation of two known fragility
modes** — Multi halving (memory: `feedback_multi_halving_fragility`)
and HONI shift-invert near-singularity (memory:
`feedback_honi_multi_fragility_propagation`) — operating in series
through HONI's incremental λ update.

Mechanism, step by step, against the trajectory in §2:

**(a) Outer iter 4 — the overshoot trigger**

HONI_exact updates λ_U incrementally:

```
λ_U_new = λ_U − min(temp)^(m−1)        # honi.py exact branch
temp    = (A · y^(m−1)) / y^(m−1)      # element-wise, y from inner Multi
```

At outer iter 4, λ_U has reached 10.756279 — already 7 × 10⁻⁶ above
the true eigenvalue 10.756272. The shifted matrix `(λ_U·I − A)` is now
near-singular by precisely the shift-invert fragility margin. Multi's
inner solve survives (inner residual 1.1 × 10⁻¹³) but only after 9
halvings.

**(b) Outer iter 5 — Multi inner trap**

With λ_U just above the true eigenvalue, the shifted system is
ill-enough that Multi's inner Newton diverges and halving cannot
recover. Inner residual is stuck at 0.994 (≈ machine ‖x^{m-1}‖) for the
full 99-iteration cap. The vector `y` produced is **not** an
approximate solution to `(λ_U·I − A) y^(m-1) = x^(m-1)`; it is the
last-iter state of a trapped Newton descent.

**(c) Outer iter 5 — Rayleigh extraction lies**

The downstream `temp = (A · y^(m−1)) / y^(m−1)` computed on the trapped
`y` produces a `min(temp)` that is large in magnitude (not near zero
as it would be near a real eigenvector). Squaring gives a positive
correction of order 0.42, so

```
λ_U_new ≈ 10.756279 − 0.4222 ≈ 10.333980
```

This is the first 0.42 jump. The new λ_U is now far from the true
eigenvalue, so the shifted system is no longer near-singular — but the
fixed point Multi finds for it bears no relation to the original
eigenproblem.

**(d) Outer iter 6 — fixed-point lock**

A second iteration of the same dynamic produces the second jump
(another 1.00) to λ_U = 9.333980. At this λ Multi finds a vector y
satisfying the shifted system to numerical precision *for the iterate
itself*, but not for the original eigenproblem. The Rayleigh-quotient
spread `|max(temp) − min(temp)| / λ_U` collapses to 1.87 × 10⁻¹⁴,
because `temp` is approximately constant on this fixed point — and
HONI declares convergence.

The condition number of `(λ_U·I − A)·diag(...)` at the locked λ is
≈ 4.6 × 10⁴ (computed in §3 of the trace), so the system is
moderately ill-conditioned but not numerically singular: an internally
consistent solve exists, it just is not the one we want.

## §5 Why the residual signal does not warn

The diagnostic at the heart of the silent failure is HONI_exact's
stopping criterion:

```
res = |max(temp) − min(temp)| / λ_U
```

This is the spread of the Rayleigh quotient and is a perfectly valid
*local* convergence indicator: at a true eigenpair `(λ, x)` we have
`A·x^(m−1) = λ·x^(m−1)` element-wise, so `temp` is the constant
vector `λ·1` and the spread is zero. Crucially, however, the spread
is also zero at any **fixed point of the iteration map**, not just
at true eigenpairs. Multi's trapped solution at outer iter 6
satisfies a shifted-system fixed-point condition that flattens
`temp` to within 10⁻¹⁴ of constant, even though it is not a real
H-eigenpair. The residual cannot tell the two cases apart without
external corroboration.

Compare the same residual definition applied with HONI_inexact:
inexact's adaptive inner tolerance never lets the inner solve trap,
so the trajectory stays in the basin of the true eigenpair and the
residual is honest.

## §6 The 9.333980 number is not in the spectrum

A multi-restart sweep is reported in §4 of the trace JSON: 20 random
initial vectors are fed to NNI canonical and HONI_exact, both on the
identical `AA`.

  - **NNI canonical** lands on λ ≈ 10.756 in **20/20** trials.
    To three decimal places, every bucket is the same.
  - **HONI_exact** lands on λ ≈ 10.756 in **5/20** trials. The other
    15 trials produce 15 distinct values scattered between 9.27 and
    9.75: `9.269, 9.283, 9.315, 9.409, 9.452, 9.501, 9.552, 9.559,
    9.593, 9.654, 9.657, 9.697, 9.704, 9.714, 9.751`.

The wide spread is the deciding evidence. If 9.33 (or any of the
others) were a genuine H-eigenvalue of `AA`, multi-restart from random
positive `x_0` should have lit up one or two clear buckets, not 15
distinct numbers. What we see instead is the empirical signature of a
**numerical artifact**: each random `x_0` traps HONI_exact at a
different fixed point of the iteration map, and the spread reflects
the geometry of the Multi inner solver's trap basin, not the
spectrum of `A`.

The practical consequence is severe: on this size and structure of
tensor, HONI_exact returns the **wrong eigenvalue 75% of the time**.
This makes the routine unsafe to use without external
cross-validation when n ≈ 50.

## §7 Recommendations

The minimum mitigation has two parts. The first is detection. The
second is recovery.

**Detection signals**, ordered by reliability:

  1. **Multi halving warning count per outer iter ≥ ~100** is a strong
     correlate of the trap. The clean trajectories above never exceed
     a few dozen halvings; the failed iter 5/6 each cross 1 600.
  2. **Non-monotone λ_U trajectory with O(1) jumps** between
     consecutive outer iterations is a strong signal. Monotone
     decrease (or near-monotone with corrections at the 10⁻⁶ level)
     is the healthy pattern; a 0.42 jump after 4 well-behaved
     iterations is not.
  3. **Inner residual stuck above ~0.1 for two consecutive outer
     iterations** while the algorithm continues to "advance" λ.

Any one of these triggering during a run should mark the result as
suspect and force fallback to a different algorithm or initial vector.

**Recovery options**, ordered by cost:

  - **Cross-validate against NNI**. NNI canonical is fast (35 ms on
    Q7_large) and reliable (20/20 in the spectrum sweep). For any
    HONI_exact result on n ≳ 50, run NNI on the same input and abort
    if the eigenvalues disagree to more than the user's tolerance.
  - **Multi-start HONI_exact** (e.g. 3–5 random positive `x_0`) and
    accept the result only if a majority agree. At 25% success rate
    a five-restart majority vote is reliable but multiplies cost by
    five.
  - **Cap halvings per outer iter** explicitly inside HONI's call
    site to Multi. If the cap fires, raise rather than silently
    accept the resulting `y`. This is an upstream code change and
    deserves its own design pass — it is *not* applied here.

For the present toolbox the cheapest fix is the first: NNI is already
available, already validated, and already integrated in the demo's
HONI vs NNI comparison tile, which makes the cross-check natural.

## §8 Connection to existing fragility memory

The failure analyzed here is *not a new fragility regime*. It is the
combination, magnified at n ≈ 50, of two existing modes:

  - `memory/feedback_multi_halving_fragility.md` documents the line
    search's lack of a healthy sweet spot when `m ≥ 3` and `AA` is
    random. That is what produces the inner trap of §4(b).
  - `memory/feedback_honi_multi_fragility_propagation.md` documents
    the shift-invert near-singularity that arises as λ_U approaches
    the true eigenvalue. That is what overshoots in §4(a) and seeds
    the trap.

The novel content of this note is the **detection pattern** and the
**statistically dominant failure mode at this scale**:

  - HONI_exact silently locks onto a non-spectral fixed point in 75%
    of random-restart trials at n = 50.
  - The residual signal does not distinguish a real eigenpair from a
    Multi-trap fixed point (§5).
  - The combination of "halving warning count > 100" and
    "non-monotone λ_U with O(1) jumps" is a high-precision detector.

These three observations are what should be promoted to a memory
entry. The mechanism itself remains under the existing two memories.

## §9 Memory recommendation

Add `memory/feedback_honi_exact_silent_lock.md` documenting:

  - what triggers it (n ≈ 50 + halving cap exceeded + λ_U overshoot),
  - the detection pattern (halving count, λ trajectory monotonicity),
  - the recommended cross-validation (NNI),
  - the explicit relation to the two existing fragility memories.

Do **not** treat this as a fourth fragility regime; the mechanism
is already covered. The memory is a detection-and-mitigation entry
for a regime where two existing fragilities compose into a silent
wrong answer at a particular problem scale.
