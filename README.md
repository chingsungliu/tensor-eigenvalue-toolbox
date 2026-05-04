# tensor-eigenvalue-toolbox

Python port of MATLAB tensor-eigenvalue research code, with a live
Streamlit demo that reproduces the five numerical examples from
**Liu, Guo, Lin** _Newton-Noda iteration for finding the Perron pair
of a weakly irreducible nonnegative tensor_, **Numer. Math. 137(1),
63–90 (2017)** — DOI [10.1007/s00211-017-0869-7](https://doi.org/10.1007/s00211-017-0869-7).

**Live demo**: <https://csliu-toolbox.streamlit.app>

---

## Quick start

### 1 · Just try the demo (no install)

Open <https://csliu-toolbox.streamlit.app> → sidebar **Step 2 演算法**
→ pick **Paper Examples (Liu 2017)** → dropdown one of the five
§7 examples → **Run NNI**. Each example shows the toolbox result
side-by-side with the paper-§7 expected iter / λ.

### 2 · Run locally

```bash
git clone https://github.com/chingsungliu/tensor-eigenvalue-toolbox
cd tensor-eigenvalue-toolbox/python
python3 -m venv .venv
.venv/bin/pip install -r ../requirements.txt -r ../requirements_ui.txt
.venv/bin/streamlit run streamlit_app/demo_v0.py
```

The demo opens at `http://localhost:8501`. Tested with Python 3.9,
numpy 2.0.2, scipy 1.13.1, streamlit 1.50.

### 3 · Reproduce paper §7 examples from the command line

```bash
cd python
.venv/bin/python test_paper_example1.py   # wind-power MTD (m=4, n=4)
.venv/bin/python test_paper_example2.py   # signless Laplacian, paper Table 1 (9 cases)
.venv/bin/python test_paper_example3.py   # halving demo, m=4, n=20
.venv/bin/python test_paper_example4.py   # weakly irreducible non-primitive (m=3, n=3)
.venv/bin/python test_paper_example5.py   # Z-tensor smallest eigenpair, paper Table 2 (9 cases)
```

Each test asserts iter / λ within paper Table tolerances and prints
PASS on success.

---

## What's implemented

### Algorithms

Three core algorithms, fully ported from the MATLAB sources in
`source_code/Tensor Eigenvalue Problem/2020_HNI_*` and parity-tested
to machine precision (see `python/test_*_parity.py`):

- **NNI** — Newton-Noda Iteration (single-layer Newton + Rayleigh-quotient
  bracket `λ_L ≤ λ_★ ≤ λ_U`). Computes the largest H-eigenvalue and the
  corresponding nonnegative eigenvector of a positive M-tensor. Two
  variants: canonical `θ = 1` and `halving` line search. Day 10/11
  perf work: **2.6–3.2× speedup** vs the initial Day 9 port (Q7 baseline
  39 ms → 12 ms).
- **HONI** — Higher-Order Newton Iteration (two-level shift-invert).
  Outer Newton on `λ_U`, inner multilinear solve via Multi. Day 10/11
  perf: **2.0× / 2.6× speedup** for `exact` / `inexact` branches.
- **Multi** — Multilinear Newton solver for `A · u^(m-1) = b`. Used as
  HONI's inner solver; exposed standalone in the demo. Day 10/11 perf:
  **3.8× speedup**.

All three accept a sparse mode-1 unfolding `(n, n^(m-1))` or a dense
m-D tensor; the polymorphic dispatch handles both. `linear_solver`
options: `spsolve` (MATLAB-parity, direct LU) and `gmres` (Python
fallback for large sparse cases where LU runs out of memory).

### Paper §7 reproduction (Phase B / C, Day 13–17)

| Example | Paper / topic | Toolbox iter | Paper iter | Status |
|---|---|---:|---:|---|
| §7 Ex 1 | Wind-power MTD (m=4, n=4) | 5 | 5 | ✅ matches paper, λ = 15.911456053987 |
| §7 Ex 2 | Signless Laplacian Table 1 (9 cases) | 8–13 | 8–13 | ✅ 9/9 within ±1 iter |
| §7 Ex 3 | Halving demo, m=4 n=20 (canonical) | 18 | ≤20 | ✅ within paper bound |
| §7 Ex 3 halving | NNI-hav variant | 199 NOT-CONVERGED | 74 (paper) / 200 NOT-CONVERGED (paper MATLAB) | ⚠ matches paper MATLAB; 74-iter paper figure unreproducible from source |
| §7 Ex 4 | Weakly irreducible (m=3, n=3) | 6 (Python RNG) / 10 (MATLAB) | Figure 3 only | ✅ Perron pair λ=1, x=(1/√3)·[1,1,1] exact |
| §7 Ex 5 | Z-tensor Table 2 (9 cases) | varies, all 9 PASS | 5–7 | ✅ 9/9 within ±2 iter (RNG difference in V trap potential) |

The Ex 3 halving discrepancy is documented in
`docs/papers/liu2017_alignment_audit.md` §7 — Day 15 verified via
GNU Octave that the paper's own MATLAB `NNI_hav.m` does not reach
the 74-iter target either; the toolbox matches MATLAB behaviour.

---

## Repository layout

```
tensor-eigenvalue-toolbox/
├── README.md                     ← you are here
├── PROGRESS.md                   per-session notes (Day 1–7)
├── journal.txt                   per-session notes (Day 8+)
├── requirements.txt              numpy, scipy
├── requirements_ui.txt           streamlit, plotly  (demo extras)
│
├── python/                       Python implementation
│   ├── tensor_utils.py           Layer 1/2/3: 5 tensor utils + multi/honi/nni
│   ├── streamlit_app/            Demo (`demo_v0.py` + `problems/`)
│   │   ├── demo_v0.py            Two-level menu (problem → algorithm)
│   │   └── problems/
│   │       └── tensor_eigenvalue/
│   │           ├── algorithms.py        7 renderers + ALGORITHM_GROUP
│   │           ├── defaults.py          Q7 builder
│   │           ├── hypergraph_utils.py  Cooper-Dutle + cyclic edge sets
│   │           ├── paper_examples.py    5 Liu 2017 §7 builders
│   │           └── uploads.py           .mat / .npz parser
│   ├── test_paper_example*.py    5 paper §7 reproduction tests
│   ├── test_*_parity.py          9 parity tests (vs MATLAB references)
│   ├── test_demo_smoke.py        UI integration smoke (AppTest-based)
│   └── test_tensor_utils.py      Python-only sanity tests
│
├── matlab_ref/                   Canonical MATLAB sources (parity references)
│   ├── hni/                      HONI / Multi
│   ├── nni/                      NNI canonical + NNI_ha
│   └── poc_iteration/            per-iteration parity POC
│
├── docs/
│   ├── algorithms_status.md      1-minute status of the 3 core algorithms
│   ├── audits/                   Phase E audit doc (this PR's parent)
│   └── papers/                   Liu 2017 alignment audit + research notes
│
├── matlab_exercise/              Day 1 worked example: gaussian_blur port
│   ├── gaussian_blur.m           Reference MATLAB implementation
│   └── generate_reference.m      Reference .mat generator
│
└── scripts/
    └── check_deploy.py           Streamlit Cloud deploy verifier
```

The `matlab_exercise/` worked example (gaussian_blur port) is preserved
for historical reference — it established the parity-test pattern that
the tensor work later inherited. See the relevant section in
`PROGRESS.md` Day 1–2.

---

## Development log

This repo grew from a single-algorithm porting exercise (gaussian_blur,
Day 1) into a paper-reproduction toolbox over ~17 working sessions:

- **Day 1–2** — gaussian_blur port, parity-test harness, `matlab_compat`
  flag pattern.
- **Day 3–7** — Layer 1/2 tensor utilities (5 functions), Multi /
  HONI / NNI Layer 3 ports, per-iteration parity framework, Streamlit
  demo v0.
- **Day 8–12** — Performance optimization (sp_Jaco_Ax 3.8× speedup
  cascading to NNI 2.6–3.2×, HONI 2.0–2.6×), m≥3 scope enforcement
  with friendly errors pointing users to scipy alternatives.
- **Day 13–16** — Phase A NNI alignment audit + Phase B paper §7
  reproduction (5 examples), Octave verification of paper-vs-MATLAB
  discrepancy on Ex 3.
- **Day 17** — Phase C demo UI integration of paper examples, Phase E
  audit (24 issues across 5 dimensions, see `docs/audits/`) + Stage 1/2
  remediation (sidebar grouping, Q7 explainer, README rewrite, UI smoke).

Per-session notes:

- `PROGRESS.md` — Day 1–7 (initial sessions, often paragraph-style).
- `journal.txt` — Day 8 onwards (terser, dated entries).

`docs/algorithms_status.md` is the recommended one-minute overview
of where the three core algorithms stand.

---

## Citation

Please cite both **the paper** that this toolbox reproduces and
**the toolbox itself** if it shows up in your work.

### The paper

> Liu, C.-S., Guo, C.-H., & Lin, W.-W. (2017). Newton-Noda iteration
> for finding the Perron pair of a weakly irreducible nonnegative
> tensor. _Numer. Math._ 137(1), 63–90.
> doi:[10.1007/s00211-017-0869-7](https://doi.org/10.1007/s00211-017-0869-7)

### The toolbox

> Liu, C.-S. (2026). _tensor-eigenvalue-toolbox: Python port and
> Streamlit demo of Liu / Guo / Lin (2017) NNI._
> commit `<your-commit-sha>`, accessed YYYY-MM-DD.
> Source: <https://github.com/chingsungliu/tensor-eigenvalue-toolbox>.

To pin a specific commit when citing:

```bash
git rev-parse --short HEAD     # → 7-char SHA, e.g. "3617b5e"
```

A formal `CITATION.cff` and a Zenodo DOI are deferred to v1.0
(see Phase E audit §3.5 in `docs/audits/`).

---

## Known limitations

- `m = 1` (scalar) and `m = 2` (matrix) are out of scope by design.
  Calling `multi / honi / nni` with `m < 3` raises `ValueError` with a
  pointer to `scipy.sparse.linalg.eigs` / `numpy.linalg.eig` for the
  matrix case.
- Paper §7 Example 3's NNI-hav 74-iter figure is unreproducible from
  the paper's own MATLAB source (not a port bug — see
  `docs/papers/liu2017_alignment_audit.md` §7).
- Paper §7 Examples 4 / 5 use random initial vectors / trap potentials;
  Python `np.random.default_rng` is not bit-compatible with MATLAB
  `rand()` so iter counts differ by ±1–2 from the paper's
  printed values.

For the full audit of code / UX / docs / test gaps see
`docs/audits/phase_e_audit_2026-05-05.md`.

---

## Troubleshooting

### Demo on Streamlit Cloud shows ImportError after a push

Streamlit Cloud's auto-rebuild webhook occasionally serves the previous
container's source for one of the changed files (cache mismatch).
Symptoms: ImportError on a symbol that exists in the latest source
(e.g., `cannot import name 'ALGORITHM_GROUP'`). Fix:

1. Open <https://share.streamlit.io/>, find the `csliu-toolbox` app.
2. Click **Manage app → Reboot**.
3. Wait ~2 minutes, refresh the demo URL.

If reboot does not pick up the latest commit:

```bash
git commit --allow-empty -m "redeploy"
git push origin main
```

If the demo sidebar shows `Build: unknown` in the build SHA caption,
the Cloud container is not preserving `.git/`; see
`docs/audits/phase_e_audit_2026-05-05.md` §5.4 for build-SHA injection
options (deferred to v1.0).

---

## License

Not yet declared. Pending v1.0 milestone (Phase E audit §3.5).
For now, cite the paper and link this repo; ask the author before
redistributing.
