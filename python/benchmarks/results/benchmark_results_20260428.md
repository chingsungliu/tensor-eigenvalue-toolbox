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
| `Q7_baseline` | Multi | 5 | — | 10.40 | 0.05 | 5.89e-12 | — | 0 |
| `Q7_baseline` | HONI_exact | 4 | 50 | 114.27 | 0.07 | 4.60e-20 | 10.756234 | 0 |
| `Q7_baseline` | HONI_inexact | 4 | 45 | 103.78 | 0.07 | 9.91e-16 | 10.756234 | 0 |
| `Q7_baseline` | NNI_spsolve | 16 | — | 36.22 | 0.05 | 4.48e-12 | 10.756234 | 0 |
| `Q7_large` | Multi | 5 | — | 11.59 | 0.19 | 2.48e-13 | — | 0 |
| `Q7_large` | HONI_exact | 13 | 0 | 422.88 | 0.96 | 1.65e-15 | 10.756272 | 0 |
| `Q7_large` | HONI_inexact | 77 | 7229 | 23635.88 | 0.96 | 2.31e-15 | 10.756272 | 0 |
| `Q7_large` | NNI_spsolve | 13 | — | 33.15 | 0.24 | 1.65e-15 | 10.756272 | 0 |
| `Q7_small` | Multi | 5 | — | 9.83 | 0.02 | 2.56e-16 | — | 0 |
| `Q7_small` | HONI_exact | 199 | 0 | 1686.12 | 0.05 | 7.36e-01 | 10.756224 | ⚠ 195 |
| `Q7_small` | HONI_inexact | 199 | 0 | 2344.02 | 0.05 | 7.36e-01 | 10.756224 | ⚠ 287 |
| `Q7_small` | NNI_spsolve | 199 | — | 887.16 | 0.03 | 7.36e-01 | 10.756224 | 0 |
| `Q7_seed_alt1` | Multi | 5 | — | 10.52 | 0.04 | 6.89e-14 | — | 0 |
| `Q7_seed_alt1` | HONI_exact | 4 | 64 | 145.47 | 0.07 | 2.19e-19 | 10.955004 | 0 |
| `Q7_seed_alt1` | HONI_inexact | 4 | 57 | 130.92 | 0.07 | 6.49e-16 | 10.955004 | 0 |
| `Q7_seed_alt1` | NNI_spsolve | 15 | — | 34.15 | 0.05 | 4.61e-13 | 10.955004 | 0 |
| `Q7_seed_alt2` | Multi | 5 | — | 10.32 | 0.04 | 1.04e-13 | — | 0 |
| `Q7_seed_alt2` | HONI_exact | 4 | 90 | 213.86 | 0.07 | 6.00e-17 | 10.989793 | 0 |
| `Q7_seed_alt2` | HONI_inexact | 4 | 134 | 361.77 | 0.07 | 2.18e-14 | 10.989793 | ⚠ 2 |
| `Q7_seed_alt2` | NNI_spsolve | 23 | — | 53.24 | 0.05 | 4.18e-11 | 10.989793 | 0 |

## Per-run wall-clock detail (sanity check for outliers)

| case | algorithm | runs (ms) |
|---|---|---|
| `Q7_baseline` | Multi | 14.08, 10.40, 10.40 |
| `Q7_baseline` | HONI_exact | 161.12, 114.08, 114.27 |
| `Q7_baseline` | HONI_inexact | 103.84, 103.78, 102.03 |
| `Q7_baseline` | NNI_spsolve | 37.23, 35.99, 36.22 |
| `Q7_large` | Multi | 11.59, 11.58, 11.64 |
| `Q7_large` | HONI_exact | 422.90, 422.88, 420.23 |
| `Q7_large` | HONI_inexact | 23644.20, 23635.88, 23614.76 |
| `Q7_large` | NNI_spsolve | 33.36, 33.15, 32.34 |
| `Q7_small` | Multi | 9.58, 9.88, 9.83 |
| `Q7_small` | HONI_exact | 1687.54, 1678.55, 1686.12 |
| `Q7_small` | HONI_inexact | 2365.28, 2342.58, 2344.02 |
| `Q7_small` | NNI_spsolve | 879.53, 888.39, 887.16 |
| `Q7_seed_alt1` | Multi | 10.52, 10.27, 10.52 |
| `Q7_seed_alt1` | HONI_exact | 145.52, 144.67, 145.47 |
| `Q7_seed_alt1` | HONI_inexact | 130.92, 130.76, 132.29 |
| `Q7_seed_alt1` | NNI_spsolve | 34.15, 34.15, 34.29 |
| `Q7_seed_alt2` | Multi | 10.32, 10.33, 10.28 |
| `Q7_seed_alt2` | HONI_exact | 213.86, 215.90, 213.59 |
| `Q7_seed_alt2` | HONI_inexact | 361.77, 361.56, 361.94 |
| `Q7_seed_alt2` | NNI_spsolve | 52.87, 53.30, 53.24 |
