# Performance baseline breakdown — auto report

- **Generated**: 2026-04-28T09:04:43
- **Case**: Q7 baseline (n=20, m=3, seed=42, tol=1e-10)
- **Categories** are *inclusive*: when timing blocks nest, the parent total includes children. Subtraction is left to the reader.
- **F1 detection flags** appear under `flag.*` rows.

## Multi

| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |
|---|---:|---:|---:|---:|---:|---:|
| `multi.linear_solve` | 5 | 4.354 | 870.78 | 832.25 | 812.42 | 1037.71 |
| `multi.tensor_contract` | 6 | 0.163 | 27.20 | 23.63 | 22.42 | 44.00 |
| `multi.residual_check` | 6 | 0.022 | 3.73 | 3.58 | 3.46 | 4.21 |
| `multi.halving_search` | 5 | 0.021 | 4.22 | 4.17 | 3.67 | 5.04 |

## HONI_exact

| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |
|---|---:|---:|---:|---:|---:|---:|
| `honi.outer_iter` | 4 | 48.557 | 12139.35 | 13497.73 | 5444.54 | 16117.42 |
| `honi.inner_multi_call` | 4 | 48.277 | 12069.32 | 13429.54 | 5368.58 | 16049.62 |
| `multi.linear_solve` | 46 | 36.968 | 803.66 | 803.75 | 775.29 | 860.88 |
| `multi.halving_search` | 46 | 8.896 | 193.38 | 127.96 | 3.50 | 644.00 |
| `multi.tensor_contract` | 415 | 6.666 | 16.06 | 15.33 | 14.25 | 25.67 |
| `multi.residual_check` | 415 | 1.010 | 2.43 | 2.29 | 2.04 | 3.58 |
| `honi.tensor_contract` | 1 | 0.024 | 23.71 | 23.71 | 23.71 | 23.71 |
| `honi.rayleigh_extract` | 4 | 0.018 | 4.62 | 4.62 | 4.46 | 4.79 |
| `honi.lambda_update` | 4 | 0.005 | 1.26 | 1.08 | 1.00 | 1.88 |

## HONI_inexact

| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |
|---|---:|---:|---:|---:|---:|---:|
| `honi.outer_iter` | 4 | 122.837 | 30709.32 | 11567.92 | 2803.46 | 96898.00 |
| `honi.inner_multi_call` | 4 | 122.458 | 30614.54 | 11475.50 | 2709.50 | 96797.67 |
| `multi.halving_search` | 41 | 80.295 | 1958.43 | 146.71 | 3.50 | 68118.71 |
| `multi.tensor_contract` | 394 | 42.793 | 108.61 | 14.83 | 14.04 | 33861.12 |
| `multi.linear_solve` | 41 | 39.497 | 963.34 | 795.04 | 763.17 | 3791.17 |
| `multi.residual_check` | 394 | 1.361 | 3.46 | 2.33 | 2.08 | 64.46 |
| `honi.tensor_contract` | 5 | 0.119 | 23.74 | 23.79 | 21.71 | 25.92 |
| `honi.rayleigh_extract` | 4 | 0.104 | 25.91 | 25.44 | 25.00 | 27.75 |
| `honi.lambda_update` | 4 | 0.003 | 0.85 | 0.75 | 0.67 | 1.25 |

## NNI_spsolve

| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |
|---|---:|---:|---:|---:|---:|---:|
| `nni.linear_solve` | 16 | 53.193 | 3324.57 | 983.75 | 876.50 | 19259.21 |
| `nni.tensor_contract` | 17 | 0.762 | 44.80 | 25.04 | 21.96 | 212.83 |
| `nni.rayleigh_quotient` | 16 | 0.163 | 10.21 | 6.40 | 5.38 | 33.25 |
| `nni.bracket_update` | 16 | 0.012 | 0.75 | 0.58 | 0.46 | 2.00 |

## NNI_ha

| category | count | total (ms) | mean (μs) | median (μs) | min (μs) | max (μs) |
|---|---:|---:|---:|---:|---:|---:|
| `nni.linear_solve` | 24 | 25.196 | 1049.85 | 932.96 | 874.12 | 1904.00 |
| `nni.tensor_contract` | 43 | 1.040 | 24.18 | 22.21 | 17.92 | 51.79 |
| `nni.halving_inner` | 24 | 0.601 | 25.06 | 32.17 | 0.33 | 36.25 |
| `nni.rayleigh_quotient` | 42 | 0.256 | 6.09 | 5.42 | 4.54 | 12.62 |
| `nni.bracket_update` | 24 | 0.011 | 0.48 | 0.46 | 0.33 | 0.75 |
