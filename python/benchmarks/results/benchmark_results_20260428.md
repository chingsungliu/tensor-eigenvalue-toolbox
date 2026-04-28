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
| `Q7_baseline` | Multi | 5 | — | 2.74 | 0.03 | 5.89e-12 | — | 0 |
| `Q7_baseline` | HONI_exact | 4 | 50 | 59.99 | 0.07 | 4.60e-20 | 10.756234 | 0 |
| `Q7_baseline` | HONI_inexact | 4 | 45 | 40.59 | 0.07 | 6.61e-16 | 10.756234 | 0 |
| `Q7_baseline` | NNI_spsolve | 16 | — | 12.18 | 0.04 | 4.48e-12 | 10.756234 | 0 |
| `Q7_baseline` | NNI_gmres | 16 | — | 8986.55 | 0.04 | 4.48e-12 | 10.756234 | 0 |
| `Q7_large` | Multi | 5 | — | 4.06 | 0.32 | 2.48e-13 | — | 0 |
| `Q7_large` | HONI_exact | 13 | 0 | 209.73 | 0.96 | 1.16e-15 | 10.756272 | 0 |
| `Q7_large` | HONI_inexact | 77 | 7229 | 12840.92 | 0.96 | 2.31e-15 | 10.756272 | 0 |
| `Q7_large` | NNI_spsolve | 13 | — | 13.61 | 0.38 | 1.16e-15 | 10.756272 | 0 |
| `Q7_large` | NNI_gmres | 16 | — | 18159.10 | 0.35 | 3.83e-12 | 10.756272 | 0 |
| `Q7_small` | Multi | 5 | — | 2.58 | 0.01 | 2.56e-16 | — | 0 |
| `Q7_small` | HONI_exact | 199 | 0 | 729.32 | 0.04 | 7.29e-01 | 10.756224 | ⚠ 195 |
| `Q7_small` | HONI_inexact | 199 | 0 | 1076.62 | 0.04 | 7.29e-01 | 10.756224 | ⚠ 287 |
| `Q7_small` | NNI_spsolve | 199 | — | 302.95 | 0.03 | 7.29e-01 | 10.756224 | 0 |
| `Q7_small` | NNI_gmres | 199 | — | 472912.45 | 0.03 | 7.33e-01 | 10.756224 | 0 |
| `Q7_seed_alt1` | Multi | 5 | — | 2.80 | 0.03 | 6.89e-14 | — | 0 |
| `Q7_seed_alt1` | HONI_exact | 4 | 64 | 53.23 | 0.07 | 2.19e-19 | 10.955004 | 0 |
| `Q7_seed_alt1` | HONI_inexact | 4 | 57 | 49.45 | 0.07 | 6.49e-16 | 10.955004 | 0 |
| `Q7_seed_alt1` | NNI_spsolve | 15 | — | 11.46 | 0.04 | 4.61e-13 | 10.955004 | 0 |
| `Q7_seed_alt1` | NNI_gmres | 15 | — | 2452.04 | 0.04 | 4.61e-13 | 10.955004 | 0 |
| `Q7_seed_alt2` | Multi | 5 | — | 2.62 | 0.03 | 1.04e-13 | — | 0 |
| `Q7_seed_alt2` | HONI_exact | 4 | 141 | 139.46 | 0.07 | 6.11e-17 | 10.989793 | 0 |
| `Q7_seed_alt2` | HONI_inexact | 4 | 134 | 163.26 | 0.07 | 1.94e-14 | 10.989793 | ⚠ 1 |
| `Q7_seed_alt2` | NNI_spsolve | 23 | — | 17.68 | 0.04 | 4.18e-11 | 10.989793 | 0 |
| `Q7_seed_alt2` | NNI_gmres | 199 | — | 315.66 | 0.04 | 1.00e+00 | 64878637642771665347599568008244311838532493872079213915270762567231272741163983529907710265837931565416448.000000 | 0 |

## Per-run wall-clock detail (sanity check for outliers)

| case | algorithm | runs (ms) |
|---|---|---|
| `Q7_baseline` | Multi | 3.10, 2.74, 2.66 |
| `Q7_baseline` | HONI_exact | 64.99, 59.99, 56.14 |
| `Q7_baseline` | HONI_inexact | 42.76, 40.59, 39.55 |
| `Q7_baseline` | NNI_spsolve | 12.54, 11.69, 12.18 |
| `Q7_baseline` | NNI_gmres | 8967.70, 9038.55, 8986.55 |
| `Q7_large` | Multi | 4.31, 4.06, 4.04 |
| `Q7_large` | HONI_exact | 209.66, 211.22, 209.73 |
| `Q7_large` | HONI_inexact | 12848.31, 12792.66, 12840.92 |
| `Q7_large` | NNI_spsolve | 13.97, 13.61, 13.58 |
| `Q7_large` | NNI_gmres | 18304.33, 18146.76, 18159.10 |
| `Q7_small` | Multi | 2.65, 2.58, 2.39 |
| `Q7_small` | HONI_exact | 730.43, 723.70, 729.32 |
| `Q7_small` | HONI_inexact | 1076.06, 1082.05, 1076.62 |
| `Q7_small` | NNI_spsolve | 302.71, 305.45, 302.95 |
| `Q7_small` | NNI_gmres | 472912.45, 472453.38, 473491.10 |
| `Q7_seed_alt1` | Multi | 3.22, 2.80, 2.59 |
| `Q7_seed_alt1` | HONI_exact | 54.47, 52.82, 53.23 |
| `Q7_seed_alt1` | HONI_inexact | 49.81, 49.45, 49.40 |
| `Q7_seed_alt1` | NNI_spsolve | 12.11, 11.44, 11.46 |
| `Q7_seed_alt1` | NNI_gmres | 2463.50, 2447.16, 2452.04 |
| `Q7_seed_alt2` | Multi | 2.80, 2.62, 2.60 |
| `Q7_seed_alt2` | HONI_exact | 141.21, 139.43, 139.46 |
| `Q7_seed_alt2` | HONI_inexact | 162.80, 163.26, 164.20 |
| `Q7_seed_alt2` | NNI_spsolve | 17.77, 17.68, 17.63 |
| `Q7_seed_alt2` | NNI_gmres | 320.28, 315.66, 313.02 |
