# Benchmark results — 2026-04-28

- **Tolerance**: `1e-10`
- **Runs per (case, algorithm)**: `3` (median wall-clock reported)
- **Algorithms**: Multi / HONI exact / HONI inexact / NNI spsolve
- **Note**: All five cases are Q7-style M-tensors; Q1 / Q3 ill-conditioned builders are unavailable in this codebase, so fragility regimes (halving / shift-invert / Rayleigh-quotient noise floor) are not exercised here.

## Test cases

| name | n | m | seed | note |
|---|---|---|---|---|
| `Q7_baseline` | 20 | 3 | 42 | Well-conditioned default; matches Streamlit demo Q7 |
| `Q7_large` | 50 | 3 | 42 | Scale up — measures how cost grows with n |
| `Q7_small` | 10 | 3 | 42 | Scale down — overhead vs. computation ratio |
| `Q7_seed_alt1` | 20 | 3 | 7 | Same dimensions, different random tensor (seed variability) |
| `Q7_seed_alt2` | 20 | 3 | 137 | Same dimensions, different random tensor (seed variability) |

## Results

| case | algorithm | nit | inner nit | wall (ms) | peak mem (MB) | final res | final λ | halving warn |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Q7_baseline` | Multi | 5 | — | 10.35 | 0.05 | 5.89e-12 | — | 0 |
| `Q7_baseline` | HONI_exact | 4 | 50 | 117.79 | 0.07 | 4.60e-20 | 10.756234 | 0 |
| `Q7_baseline` | HONI_inexact | 4 | 45 | 106.94 | 0.07 | 9.91e-16 | 10.756234 | 0 |
| `Q7_baseline` | NNI_spsolve | 16 | — | 39.10 | 0.05 | 4.48e-12 | 10.756234 | 0 |
| `Q7_large` | Multi | 5 | — | 11.75 | 0.19 | 2.48e-13 | — | 0 |
| `Q7_large` | HONI_exact | 13 | 0 | 439.84 | 0.96 | 1.65e-15 | 10.756272 | 0 |
| `Q7_large` | HONI_inexact | 77 | 7229 | 23950.67 | 0.96 | 2.31e-15 | 10.756272 | 0 |
| `Q7_large` | NNI_spsolve | 13 | — | 35.09 | 0.24 | 1.65e-15 | 10.756272 | 0 |
| `Q7_small` | Multi | 5 | — | 10.34 | 0.02 | 2.56e-16 | — | 0 |
| `Q7_small` | HONI_exact | 199 | 0 | 1706.90 | 0.05 | 7.36e-01 | 10.756224 | ⚠ 195 |
| `Q7_small` | HONI_inexact | **FAILED** | — | — | — | — | — | 99 |
| `Q7_small` | NNI_spsolve | 199 | — | 904.60 | 0.03 | 7.36e-01 | 10.756224 | 0 |
| `Q7_seed_alt1` | Multi | 5 | — | 10.27 | 0.04 | 6.89e-14 | — | 0 |
| `Q7_seed_alt1` | HONI_exact | 4 | 64 | 150.67 | 0.07 | 2.19e-19 | 10.955004 | 0 |
| `Q7_seed_alt1` | HONI_inexact | 4 | 57 | 135.71 | 0.07 | 6.49e-16 | 10.955004 | 0 |
| `Q7_seed_alt1` | NNI_spsolve | 15 | — | 35.94 | 0.05 | 4.61e-13 | 10.955004 | 0 |
| `Q7_seed_alt2` | Multi | 5 | — | 10.90 | 0.04 | 1.04e-13 | — | 0 |
| `Q7_seed_alt2` | HONI_exact | 4 | 90 | 222.16 | 0.07 | 6.00e-17 | 10.989793 | 0 |
| `Q7_seed_alt2` | HONI_inexact | 4 | 134 | 376.11 | 0.07 | 2.18e-14 | 10.989793 | ⚠ 2 |
| `Q7_seed_alt2` | NNI_spsolve | 23 | — | 55.97 | 0.05 | 4.18e-11 | 10.989793 | 0 |

## Per-run wall-clock detail (sanity check for outliers)

| case | algorithm | runs (ms) |
|---|---|---|
| `Q7_baseline` | Multi | 10.80, 10.35, 10.29 |
| `Q7_baseline` | HONI_exact | 149.68, 116.15, 117.79 |
| `Q7_baseline` | HONI_inexact | 107.21, 105.97, 106.94 |
| `Q7_baseline` | NNI_spsolve | 39.18, 38.08, 39.10 |
| `Q7_large` | Multi | 11.71, 12.50, 11.75 |
| `Q7_large` | HONI_exact | 439.30, 441.00, 439.84 |
| `Q7_large` | HONI_inexact | 23950.67, 24130.73, 23893.63 |
| `Q7_large` | NNI_spsolve | 36.28, 33.56, 35.09 |
| `Q7_small` | Multi | 10.60, 10.34, 10.12 |
| `Q7_small` | HONI_exact | 1713.21, 1706.90, 1701.98 |
| `Q7_small` | HONI_inexact | 735.87 (run before failure) |
| `Q7_small` | NNI_spsolve | 904.39, 904.60, 908.47 |
| `Q7_seed_alt1` | Multi | 10.24, 10.41, 10.27 |
| `Q7_seed_alt1` | HONI_exact | 150.67, 153.42, 150.38 |
| `Q7_seed_alt1` | HONI_inexact | 135.06, 138.64, 135.71 |
| `Q7_seed_alt1` | NNI_spsolve | 35.21, 35.94, 36.23 |
| `Q7_seed_alt2` | Multi | 10.77, 10.90, 10.92 |
| `Q7_seed_alt2` | HONI_exact | 226.21, 221.53, 222.16 |
| `Q7_seed_alt2` | HONI_inexact | 377.11, 376.11, 373.58 |
| `Q7_seed_alt2` | NNI_spsolve | 55.97, 57.02, 55.46 |

## Failures

| case | algorithm | exception | message |
|---|---|---|---|
| `Q7_small` | HONI_inexact | `AssertionError` | res upper-bound violation: max res[:2] = 2.636455e+10, na+nb = 9.818684e+00 |
