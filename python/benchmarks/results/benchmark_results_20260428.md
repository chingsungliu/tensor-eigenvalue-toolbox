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
| `Q7_baseline` | Multi | 5 | — | 20.44 | 0.05 | 5.89e-12 | — | 0 |
| `Q7_baseline` | HONI_exact | 4 | 50 | 129.28 | 0.07 | 4.60e-20 | 10.756234 | 0 |
| `Q7_baseline` | HONI_inexact | 4 | 45 | 104.33 | 0.07 | 9.91e-16 | 10.756234 | 0 |
| `Q7_baseline` | NNI_spsolve | 16 | — | 41.99 | 0.05 | 4.48e-12 | 10.756234 | 0 |
| `Q7_large` | Multi | 5 | — | 11.55 | 0.19 | 2.48e-13 | — | 0 |
| `Q7_large` | HONI_exact | 6 | 233 | 898.68 | 0.96 | 1.87e-14 | 9.333980 | ⚠ 99 |
| `Q7_large` | HONI_inexact | 77 | 7229 | 26405.21 | 0.96 | 2.31e-15 | 10.756272 | 0 |
| `Q7_large` | NNI_spsolve | 13 | — | 35.35 | 0.24 | 1.65e-15 | 10.756272 | 0 |
| `Q7_small` | Multi | 5 | — | 11.15 | 0.02 | 2.56e-16 | — | 0 |
| `Q7_small` | HONI_exact | 6 | 351 | 1240.58 | 0.03 | 2.25e-12 | 8.756218 | ⚠ 294 |
| `Q7_small` | HONI_inexact | **FAILED** | — | — | — | — | — | 99 |
| `Q7_small` | NNI_spsolve | **FAILED** | — | — | — | — | — | 0 |
| `Q7_seed_alt1` | Multi | 5 | — | 10.89 | 0.04 | 6.89e-14 | — | 0 |
| `Q7_seed_alt1` | HONI_exact | 4 | 64 | 168.51 | 0.07 | 2.19e-19 | 10.955004 | 0 |
| `Q7_seed_alt1` | HONI_inexact | 4 | 57 | 146.66 | 0.07 | 6.49e-16 | 10.955004 | 0 |
| `Q7_seed_alt1` | NNI_spsolve | 15 | — | 36.27 | 0.05 | 4.61e-13 | 10.955004 | 0 |
| `Q7_seed_alt2` | Multi | 5 | — | 9.99 | 0.04 | 1.04e-13 | — | 0 |
| `Q7_seed_alt2` | HONI_exact | 4 | 90 | 242.71 | 0.07 | 6.00e-17 | 10.989793 | 0 |
| `Q7_seed_alt2` | HONI_inexact | 4 | 134 | 405.20 | 0.07 | 2.18e-14 | 10.989793 | ⚠ 2 |
| `Q7_seed_alt2` | NNI_spsolve | 23 | — | 65.62 | 0.05 | 4.18e-11 | 10.989793 | 0 |

## Per-run wall-clock detail (sanity check for outliers)

| case | algorithm | runs (ms) |
|---|---|---|
| `Q7_baseline` | Multi | 17.59, 45.57, 20.44 |
| `Q7_baseline` | HONI_exact | 147.02, 117.05, 129.28 |
| `Q7_baseline` | HONI_inexact | 107.71, 99.70, 104.33 |
| `Q7_baseline` | NNI_spsolve | 41.99, 34.64, 47.77 |
| `Q7_large` | Multi | 11.33, 11.55, 14.85 |
| `Q7_large` | HONI_exact | 906.56, 879.05, 898.68 |
| `Q7_large` | HONI_inexact | 26525.94, 26405.21, 26246.69 |
| `Q7_large` | NNI_spsolve | 33.54, 35.35, 40.10 |
| `Q7_small` | Multi | 10.77, 11.15, 12.94 |
| `Q7_small` | HONI_exact | 1226.53, 1240.58, 1241.79 |
| `Q7_small` | HONI_inexact | 791.60 (run before failure) |
| `Q7_small` | NNI_spsolve | 477.18 (run before failure) |
| `Q7_seed_alt1` | Multi | 12.57, 9.86, 10.89 |
| `Q7_seed_alt1` | HONI_exact | 168.51, 155.16, 171.96 |
| `Q7_seed_alt1` | HONI_inexact | 148.44, 146.66, 142.80 |
| `Q7_seed_alt1` | NNI_spsolve | 36.27, 39.71, 34.84 |
| `Q7_seed_alt2` | Multi | 10.47, 9.99, 9.88 |
| `Q7_seed_alt2` | HONI_exact | 242.71, 239.18, 246.87 |
| `Q7_seed_alt2` | HONI_inexact | 410.81, 396.61, 405.20 |
| `Q7_seed_alt2` | NNI_spsolve | 67.87, 65.62, 57.45 |

## Failures

| case | algorithm | exception | message |
|---|---|---|---|
| `Q7_small` | HONI_inexact | `AssertionError` | res upper-bound violation: max res[:2] = 2.636455e+10, na+nb = 9.818684e+00 |
| `Q7_small` | NNI_spsolve | `AssertionError` | res upper-bound violation: max res[:200] = 2.023117e+05, pre-fill was 1.0 (should hold for clean M-tensor with lambda_L >= 0) |
