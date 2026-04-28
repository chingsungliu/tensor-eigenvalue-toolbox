# F1 root-cause analysis — auto summary

- **Generated**: 2026-04-28T08:48:20
- **Config**: n=50, m=3, seed=42, tol=1e-10, maxit=200

## §1 HONI_exact reproduction

- final λ           : `9.3339801587`
- final residual    : `1.870e-14`
- outer iterations  : `6`
- inner iterations  : `233`
- halving warnings  : `99`

λ_U trajectory (every outer iter):

| iter | λ_U | residual | halvings | inner nit |
|---:|---:|---:|---:|---:|
| 0 | 11.572213 | 8.365e-01 | 0 | 0 |
| 1 | 10.902251 | 7.975e-01 | 0 | 6 |
| 2 | 10.767959 | 7.035e-01 | 0 | 12 |
| 3 | 10.756739 | 2.458e-01 | 3 | 19 |
| 4 | 10.756279 | 6.161e-03 | 34 | 33 |
| 5 | 10.333980 | 3.372e-02 | 1697 | 133 |
| 6 | 9.333980 | 1.870e-14 | 2970 | 233 |

## §2 Cross-check on same input

| algorithm | final λ | residual | outer nit | halving warns |
|---|---:|---:|---:|---:|
| HONI_inexact | 10.7562720687 | 2.312e-15 | 77 | 0 |
| NNI_spsolve (canonical) | 10.7562720687 | 1.651e-15 | 13 | 0 |
| NNI_ha (halving=True) | 10.7562720687 | 2.129e-13 | 16 | 0 |

## §3 Inner trace (Multi inside HONI_exact)

- first outer iter with halving > 0: **3**
- shifted system condition at locked λ: ||M||=1.425e+00, ||M^-1||=3.240e+04, cond=4.619e+04

Per-outer Multi inner diagnostics:

| outer | λ_U | inner max it | inner res min | inner res max | hal | min θ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.902251 | 5 | 9.212e-11 | 9.321e-01 | 0 | 1.000e+00 |
| 2 | 10.767959 | 5 | 1.753e-11 | 3.937e-01 | 0 | 1.000e+00 |
| 3 | 10.756739 | 6 | 1.387e-13 | 3.986e-01 | 2 | 1.111e-01 |
| 4 | 10.756279 | 13 | 1.099e-13 | 7.088e-01 | 9 | 1.372e-03 |
| 5 | 10.333980 | 99 | 9.943e-01 | 9.943e-01 | 99 | 2.581e-09 |
| 6 | 9.333980 | 99 | 1.417e+00 | 1.417e+00 | 99 | 4.857e-15 |

## §4 Multi-restart spectrum exploration

- `20` random initial vectors
- HONI_exact λ buckets : `{'10.756': 5, '9.315': 1, '9.593': 1, '9.657': 1, '9.409': 1, '9.559': 1, '9.452': 1, '9.697': 1, '9.704': 1, '9.751': 1, '9.654': 1, '9.283': 1, '9.501': 1, '9.269': 1, '9.552': 1, '9.714': 1}`
- NNI_spsolve λ buckets: `{'10.756': 20}`
