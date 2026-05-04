# Phase E Tensor Module Audit (Day 17, 2026-05-05)

Read-only inventory of the tensor eigenvalue module — code quality, UX,
documentation, test coverage, and onboarding. **No code was modified
during this audit**; the deliverable is this document plus the §6
priority matrix that the user can act on selectively.

---

## Scope

| Area | Files |
|---|---|
| Module code | `python/streamlit_app/problems/tensor_eigenvalue/{algorithms,defaults,hypergraph_utils,paper_examples,uploads}.py` |
| Module support | `python/streamlit_app/_internal/{snippets,utility_renderers}.py` |
| Algorithm core (user surface only) | `python/tensor_utils.py` (signatures + validation, not internals) |
| Tests — paper reproduction | `python/test_paper_example{1..5}.py` |
| Tests — parity | `python/test_{tenpow,tpv,sp_tendiag,ten2mat,sp_Jaco_Ax,multi,honi,nni,nni_ha}_parity.py` |
| Tests — sanity | `python/test_tensor_utils.py`, `python/test_sp_Jaco_Ax_unit.py` |
| Demo entry | `python/streamlit_app/demo_v0.py` |
| Documentation | `README.md`, `docs/papers/liu2017_alignment_audit.md`, `docs/algorithms_status.md`, `docs/papers/rayleigh_quotient_noise_floor_en.md` |
| Build | `requirements.txt`, `requirements_ui.txt` |

Out of scope: BEC (NLS) module (per task spec), `matlab_ref/`, `source_code/`,
`.streamlit/`, MATLAB-port internals of `tensor_utils.py`.

---

## §1 Code quality / consistency

### §1.1 [HIGH] `render_paper_examples` breaks the established renderer pattern

**Location**: `python/streamlit_app/problems/tensor_eigenvalue/algorithms.py`
lines 196–366 (`render_multi`), 368–515 (`render_honi`), 518–693 (`render_nni`),
696–924 (`render_hni_vs_nni`), 927–1180 (`render_eigenvalue_compare`),
1182–1350 (`render_multilinear_compare`) all share an identical template:
**2-column layout (`col_in`, `col_out`)** + `st.form` + `session_state["{key}_result"]`
persistence + `_render_data_source_block` (Q7 / Upload) + `with st.expander("📄
對應的 MATLAB 原始碼")` snippet block at the bottom.

`render_paper_examples` (lines ~1490–1730, added Day 17 Phase C) uses
**vertical layout, no `st.form`, no `st.session_state` persistence,
no Q7/Upload data source, no MATLAB snippet expander**.

**Impact**: A user clicking through the 7 algorithm tabs experiences a
visual / interaction discontinuity at the 7th. State is lost on any
widget interaction (results vanish; rerun required). Day 17 Phase C
explicitly chose the simpler design as a paper-showcase tile, but
the inconsistency reads as a half-finished tab.

**Suggestion**:
- Option A (preserve simpler pattern, document decision): add a
  module-level comment at the top of `render_paper_examples` stating
  why the layout differs (e.g. "intentional — paper examples have
  fixed config, not a configurable lab").
- Option B (refactor to match): wrap dropdown + config in `st.form`,
  persist results via `st.session_state["paper_example_result"]`. Adds
  ~30 lines but eliminates inconsistency.

**Fix cost**: Option A ~5 min, Option B ~45 min.

---

### §1.2 [MEDIUM] `_EX2_CASE_TOL` is duplicated between test and renderer

**Location**:
- `python/test_paper_example2.py` line ~70 — `CASE_TOL = {(3,20): 1e-13, ...}`
- `python/streamlit_app/problems/tensor_eigenvalue/algorithms.py` line ~1357 —
  `_EX2_CASE_TOL = {(3,20): 1e-13, ...}`

Same 9-entry dict with identical comments about which cases need a
tighter tol and why. If the user ever revises one (e.g. relaxes
`(4, 100)` to a different value after a noise-floor re-investigation),
the other will silently disagree.

**Impact**: Latent bug — Phase B B2 reproduction in the test could
pass while the demo gives a different iter count for the same case.

**Suggestion**: Move `CASE_TOL` to `paper_examples.py` (module-level
constant) and import it from both sites. Or add a paper-Example-2
helper `default_tol_for_case((m, n)) -> float` next to
`build_liu2017_example2`.

**Fix cost**: ~15 min (move constant + update 2 import sites).

---

### §1.3 [MEDIUM] `render_paper_examples` lacks try/except around `nni()` call

**Location**: `algorithms.py` lines ~1605–1660 (the `with st.spinner: ... nni(...)` block).

Other 6 renderers wrap algorithm calls in `try / except Exception as e:`
and surface failures via `st.session_state["{key}_result"] = {"error": ...}`
+ `st.error(f"... failed — {result['error']}")`.

`render_paper_examples` does not. If a paper example's NNI call raises
(e.g. user uploads a corrupted .mat in a future variant; or an
upstream library regression), the user sees a Streamlit traceback
instead of a friendly error.

**Impact**: Demo-cloud user sees a stack trace; not actionable.

**Suggestion**: Wrap the NNI call in `try / except Exception as e: st.error(...); return`.

**Fix cost**: ~10 min.

---

### §1.4 [LOW] `algorithms.py` module docstring is stale

**Location**: `algorithms.py` lines 1–11.

> "Exposes four Streamlit renderers via the ``ALGORITHMS`` dict:
>   - ``Multi``, ``HONI``, ``NNI``, ``HONI vs NNI comparison``"

Reality: 7 renderers (the docstring missed Day 6's two `*_compare`
multi-run renderers and Day 17 Phase C's `Paper Examples`).

**Impact**: Anyone reading the file top-to-bottom is told the wrong
count and misses the 3 newer renderers.

**Suggestion**: Update bullet list to 7 entries.

**Fix cost**: ~5 min.

---

### §1.5 [LOW] `nni()` default `initial_vector` uses unseeded RNG

**Location**: `python/tensor_utils.py` lines 1421–1423.

```python
if initial_vector is None:
    rng = np.random.default_rng()
    initial_vector = rng.random(n)
```

No seed — every call without an explicit `initial_vector` gives a
different x0. The demo always passes `x0` explicitly so this is not
hit through the UI, but a user calling `nni()` directly from a script
(or REPL) gets non-reproducible results without realising why.

**Impact**: Surprise non-determinism for direct users.

**Suggestion**: Either accept a `random_state` kwarg (passing
`None` → unseeded, `42` → reproducible), or document the unseeded
behaviour explicitly in the docstring near the parameter description.
Minimum: docstring note.

**Fix cost**: ~5 min (docstring) / ~20 min (kwarg).

---

## §2 UX / demo friendliness

### §2.1 [HIGH] No "where to start" guidance for the 7-tab algorithm dropdown

**Location**: Sidebar in `python/streamlit_app/demo_v0.py` lines 94–99 —
shows all 7 algorithms in a flat radio list. No annotation, no recommended
entry point.

A first-time visitor lands on **Multi** (first alphabetical, the inner
solver of HONI — least intuitive for someone wanting "tensor eigenvalue").
A natural reading order would be Paper Examples → NNI → HONI → comparisons.

**Impact**: New visitors ricochet through the tabs trying to figure
out what they're looking at. Especially poor if the visitor came from
the paper / a citation.

**Suggestion**:
- Group the 7 entries with subheadings: "**Algorithms**" (Multi / HONI / NNI),
  "**Comparisons**" (HONI vs NNI / Eigenvalue compare / Multilinear compare),
  "**Paper reproduction**" (Paper Examples).
- Or reorder so Paper Examples is first (visitor coming from paper
  immediately sees what they're looking for).
- Add a one-sentence caption above the radio: "**New here?** Try
  *Paper Examples* to see the toolbox reproducing Liu 2017 §7 figures."

**Fix cost**: ~15 min.

---

### §2.2 [HIGH] Q7 default tensor has no in-UI explanation

**Location**: `python/streamlit_app/problems/tensor_eigenvalue/defaults.py`
lines 10–40 + every renderer's data-source radio shows `"Q7 default
(n=20, m=3)"` as the default option label.

A new user clicks Run NNI and sees `λ ≈ 9.5`, residual converged. They
have no idea what tensor was solved (Q7 is internal jargon — references
"Q7 case" from `feedback_nni_rayleigh_quotient_noise_floor.md` memory,
not visible to demo users).

**Impact**: Demo runs but the result is meaningless without context.
Cloud-deploy URL visitors don't read source memos.

**Suggestion**: Add a `?` help icon (Streamlit's `help=` kwarg) on the
data-source radio explaining: "Q7 is a synthetic positive M-tensor —
diagonal U[1, 11] plus 2%-density 0.01-magnitude perturbation,
positive initial vector. Designed to exercise the noise-floor
fragility documented in the research note." (or similar non-jargon
wording).

**Fix cost**: ~15 min.

---

### §2.3 [MEDIUM] Halving toggle uses paper notation without explanation

**Location**: `algorithms.py` `render_paper_examples` Example 3 toggle
(lines ~1567–1577) shows two radio options:

- `"canonical (θ_k=1)"` — paper notation, no explanation.
- `"NNI-hav (halving)"` — research jargon, no explanation.

A non-expert user (e.g. a student just learning about NNI) doesn't
know what θ means or what halving does.

**Impact**: User picks one at random, gets confused by why one
takes 18 iter and the other 200.

**Suggestion**: Use friendlier labels:
- `"Canonical Newton step (θ=1, faster)"`
- `"With halving line search (more robust on ill-conditioned cases)"`

Or add a `help=` tooltip on the radio.

**Fix cost**: ~5 min.

---

### §2.4 [MEDIUM] `render_paper_examples` Example 3 + 5 expected-iter UI is awkward for non-int values

**Location**: `algorithms.py` `render_paper_examples` lines ~1675–1686.

For Examples 3 (halving) and 4, `expected_nit` is a string like
`"~74 (paper text) / 200 NOT CONVERGED (paper MATLAB)"` or `"~10
(MATLAB Octave reference, seed=42)"`. The "Paper comparison" block
falls back to a single line of plain markdown with no match marker
(✅/🟡/❌). A user sees iter=199 and "Paper expectation: ~74 / 200
NOT CONVERGED; toolbox got 199" — no immediate visual signal of "this
is the expected outcome, not a bug".

**Impact**: User misreads NOT CONVERGED as failure when in fact the
audit shows MATLAB itself doesn't converge (paper Figure 2 is
unreproducible from the MATLAB source).

**Suggestion**: For string-valued `expected_nit`, add an explicit
`st.info("ℹ️ This is the expected outcome — see audit §7")` or
similar visual signal that the result is documented.

**Fix cost**: ~10 min.

---

### §2.5 [LOW] Eigenvector display threshold is hardcoded to `n ≤ 20`

**Location**: `algorithms.py` `render_paper_examples` lines ~1718–1730
(`if len(x_display) <= 20: bar; else: summary`).

For Example 5 cases (n ∈ {20, 50, 100}), `n=20` shows a bar (good)
but `n=50` and `n=100` collapse to a 6-line summary. A user comparing
across cases loses the visual structure of the eigenvector. (Other
renderers always show `_plot_bar_vector` regardless of n; Plotly
handles 100-bar charts fine.)

**Impact**: Consistency loss — user wonders why two of the 9 cases
show no plot.

**Suggestion**: Drop the threshold; always plot. Or raise the
threshold to ~200.

**Fix cost**: ~3 min.

---

### §2.6 [LOW] Mixed Chinese/English UI text

**Location**: Throughout `algorithms.py` — captions like `"**讀圖要點**：…"`
in Chinese, but metric labels (`"final λ_U"`, `"spread"`) and tab
headers (`"HONI"`, `"NNI"`, `"Comparison"`) in English.

`demo_v0.py` sidebar uses `"### Step 1 — 問題類別"` (Chinese);
`render_paper_examples` markdown is mostly English.

**Impact**: A non-Chinese reader hits the `讀圖要點` captions and
loses ~30% of the explanatory content. A Chinese reader sees the
mix as inconsistent.

**Suggestion**: This is by design (user is a Traditional-Chinese
speaker, demo is bilingual). Document the convention in a
`CONTRIBUTING.md` or `.claude/skills/` note. Low priority — the
captions are explanatory, not control flow.

**Fix cost**: ~10 min (write a "bilingual convention" note) or skip.

---

## §3 Documentation gaps

### §3.1 [HIGH] `README.md` is severely stale

**Location**: `README.md` (199 lines).

The README is from Day 1–2 era and describes only `gaussian_blur`:

- **Lines 19–26** ("這是什麼"): "目前還不是正式的演算法庫，裡面只有
  一個示範用的 `gaussian_blur` 實作". Reality: the active scope is
  the **tensor eigenvalue module** (5 paper examples reproduced,
  Streamlit demo with 7 algorithm tabs).
- **Lines 30–43** (directory tree): Shows only `gaussian_blur.m` /
  `gaussian_blur.py` / `test_parity.py`. Misses `streamlit_app/`,
  `tensor_utils.py`, all 14 tensor-related test files, `docs/papers/`,
  `matlab_ref/`.
- **Lines 47–73** (環境設定): `pip install numpy scipy` only — misses
  `requirements_ui.txt` (streamlit, plotly).
- **Lines 91–119** (新增 port 流程): describes a single-algorithm port
  workflow. Doesn't mention the 5-paper-example reproduction track or
  the demo integration.
- **Live demo URL** at top is correct (line 3) but isolated from the
  rest of the README narrative.
- Lines 122–139 ("三大陷阱"): only 3 of the actual 5 (the iter / sparse
  trap from Day 2 is missing — see CLAUDE.md §4.5).

**Impact**: GitHub-cloning newcomers / paper-citation visitors get
the wrong picture of what this repo is. The Live demo URL hint at
top is the only signal that there's more.

**Suggestion**: Major rewrite — preserve the gaussian_blur design
discussion as a "**Worked example: gaussian_blur port**" sub-section,
elevate tensor eigenvalue / paper reproduction / Streamlit demo to
the primary narrative. Estimate ~2 hours for a thoughtful rewrite
that preserves the historical content.

**Fix cost**: ~2 h.

---

### §3.2 [MEDIUM] `requirements_ui.txt` has duplicated lines

**Location**: `requirements_ui.txt`:

```
streamlit
plotly
streamlit
plotly
```

(Each package listed twice.) Pip handles this gracefully but it's a
code-smell that suggests a copy-paste accident.

Neither requirements file pins versions. The Phase B / C work that
exercised the module was tested against numpy 2.0.2, scipy 1.13.1,
streamlit (latest at Day 5–17). Future installs may drift.

**Impact**: Latent reproducibility issue + cosmetic.

**Suggestion**:
- Dedupe lines.
- Pin versions: `numpy==2.0.2`, `scipy==1.13.1`, `streamlit>=1.28`,
  `plotly>=5.0` (or whatever range was tested).
- Optionally split into `requirements.txt` (core, demanded by `tensor_utils.py`)
  and `requirements_ui.txt` (extras for the demo).

**Fix cost**: ~10 min.

---

### §3.3 [MEDIUM] No top-level guide to the test suite

**Location**: There is no top-level docs/tests.md or section in README
that says "to run the tests, do X".

`tests/` (or its absence — tests are flat under `python/test_*.py`)
isn't pytest-discoverable in the standard way: each test file uses
`if __name__ == "__main__": main()` and is run as a script. There's
no `pytest.ini` or `conftest.py`.

**Impact**: A reviewer / collaborator has to inspect the file structure
and figure out the convention. Phase B reproduction tests need explicit
`.venv/bin/python test_paper_example3.py` invocation; nothing collects
them all.

**Suggestion**: Add a `python/run_all_tests.sh` (or `Makefile` target)
that loops over the 14 in-scope test files. Or add `if __name__ ==
"__main__"` plus pytest-style entry points. Or write a one-line
pytest command in the README:

```bash
.venv/bin/pytest python/test_paper_example*.py python/test_*_parity.py
```

(Tests are pytest-compatible if you run them as scripts; just add a
small `__main__` block to each that calls `pytest.main([__file__])`.)

**Fix cost**: ~30 min (write a `run_tests.sh` + document it in README).

---

### §3.4 [LOW] `docs/papers/liu2017_alignment_audit.md` lacks a TOC

**Location**: `docs/papers/liu2017_alignment_audit.md` (306 lines, 7 sections).

Sections §1 through §7 are well-structured but require scrolling. §6
(Day 14 update) and §7 (Day 15 update) were appended after the original
§1–§5 and grew the doc materially. No table of contents or section
links at the top.

**Impact**: Reader has to scroll to find specific sections (e.g.
"where is the §7 paper-vs-MATLAB three-way comparison?"). Not a
correctness issue.

**Suggestion**: Add a short TOC after the front-matter:

```markdown
**Contents**: [§1 Background](#1-background) ·
[§2 7-point alignment](#2-7-point-alignment-audit) ·
[§3 Deviation classification](#3-deviation-classification) ·
[§4 No code changes needed](#4-no-algorithm-code-changes-needed) ·
[§5 Conclusion](#5-conclusion) ·
[§6 Stopping criterion update (Day 14)](#6-update--stopping-criterion-day-14-phase-b-b2-confer) ·
[§7 Paper-vs-MATLAB three-way (Day 15)](#7-update--paper-text-vs-matlab-code-numerical-discrepancy-day-15-octave-verification)
```

**Fix cost**: ~5 min.

---

### §3.5 [LOW] No CITATION.cff or LICENSE file

**Location**: Repo root has neither `LICENSE` nor `CITATION.cff`.

A researcher wanting to cite this toolbox in a paper has nothing
formal to point to. A user wanting to use the code is unsure of the
licensing terms.

**Impact**: External reuse / citation friction.

**Suggestion**:
- Add `LICENSE` (MIT / Apache 2.0 / GPL — user choice).
- Add `CITATION.cff` with author info + repo URL + (optionally) a Zenodo DOI.

**Fix cost**: ~20 min for LICENSE + CITATION.cff drafts.

---

## §4 Test coverage gaps

### §4.1 [HIGH] No automated UI / integration test in the repo

**Location**: There is no `python/test_demo_ui.py` (or similar) that
exercises `streamlit_app/demo_v0.py`. The Phase C smoke test
(`_paper_examples_smoke.py`) was a throwaway file, deleted after the
verification.

**Impact**: Future renderer additions / refactors won't be caught
until someone manually clicks through every tab. The Day 17 Phase C
verification proved that **`streamlit.testing.v1.AppTest` works
cleanly** in this repo's environment — the harness is ready but
unused.

**Suggestion**: Resurrect a stripped-down `python/test_demo_smoke.py`
that:

1. Loads `demo_v0.py` via `AppTest.from_file(...)`.
2. For each of the 7 algorithm tabs, switches to it and asserts no
   exception.
3. For Paper Examples, drives the 5-example dropdown and asserts each
   example's Run produces the expected `nit`.

The existing smoke logic can be lifted from the deleted file (Day 17
Phase C). ~120 lines.

**Fix cost**: ~45 min (rewrite + tune for stability).

---

### §4.2 [MEDIUM] No test for `uploads.py` parser

**Location**: `python/streamlit_app/problems/tensor_eigenvalue/uploads.py`
(235 lines) has clean error messages, edge-case handling
(reserved keys, auto-detect, m inference), and tests like 80% of
the `_render_data_source_block` UX. No corresponding `test_uploads.py`.

**Impact**: An accidental regression in the upload parser (e.g.,
key-detection logic, m inference) wouldn't be caught until a user
uploads a real file in the demo.

**Suggestion**: Add `python/test_uploads.py` with these cases:

- `.mat` with reserved key `AA` (sparse) — round-trips.
- `.npz` with reserved key `AA_tensor` (dense m-D) — auto-unfolds.
- `.npz` from `scipy.sparse.save_npz` — picked up via `format` key.
- File with neither reserved key — auto-detect picks first eligible.
- Malformed file (raise ValueError with helpful message).
- Wrong x0 shape (raises with informative message).

~80 lines. All testable via `load_tensor_bytes(raw, filename)` (the
testable variant explicitly designed for this — no Streamlit deps).

**Fix cost**: ~45 min.

---

### §4.3 [MEDIUM] No edge-case rejection tests for `multi/honi/nni`

**Location**: `python/test_tensor_utils.py` has `test_multi_basic`,
`test_honi_basic`, `test_nni_basic` — these test the happy path but
none assert that `m=2`, `m=1`, `tol<=0`, `maxit<2` raise the friendly
errors documented in `tensor_utils.py` (lines 568–581 multi, 862–881
honi, 1353–1373 nni).

**Impact**: A regression that turns a `ValueError` into a silent crash
won't be caught.

**Suggestion**: Add ~5 lines to each `test_xxx_basic` asserting the
expected `ValueError` is raised with `pytest.raises` (or
`try/except` if pytest isn't standard here).

**Fix cost**: ~20 min.

---

### §4.4 [LOW] GMRES branch coverage is thin

**Location**: `nni()` supports `linear_solver="gmres"` (Python-only,
not parity-tested against MATLAB by design). The renderer
`render_nni` exposes it, but there's no automated test that GMRES
actually returns approximately the same eigenpair as `spsolve`.

**Impact**: The GMRES path could silently degrade (e.g. wrong
preconditioner setup) without tripping any current test.

**Suggestion**: Add a sanity test in `test_tensor_utils.py::test_nni_gmres_matches_spsolve`
on the Q7 default — assert `|λ_gmres − λ_spsolve| < 1e-6`,
`‖x_gmres − x_spsolve‖_2 < 1e-4`. Loose tolerance because GMRES is
iterative and not bit-equivalent to the direct solver.

**Fix cost**: ~20 min.

---

## §5 Onboarding / discoverability

### §5.1 [HIGH] No `git clone → demo running` instructions

**Location**: README §"環境設定" (lines 47–73) gives instructions for
the **gaussian_blur** workflow, not the tensor eigenvalue / Streamlit
flow that is the active scope.

A new visitor cloning the repo has no instruction sequence for:

```bash
cd ~/Projects/my-toolbox/python
python3 -m venv .venv
.venv/bin/pip install -r ../requirements.txt -r ../requirements_ui.txt
.venv/bin/streamlit run streamlit_app/demo_v0.py
```

(or whatever the correct sequence is — note the requirements files
live in repo root, not inside `python/`.)

**Impact**: Visitor either gives up, or fumbles around trying.

**Suggestion**: Add a `## Quick start` section near the top of README:

```markdown
## Quick start

Run the demo locally:

\`\`\`bash
git clone https://github.com/chingsungliu/tensor-eigenvalue-toolbox
cd tensor-eigenvalue-toolbox/python
python3 -m venv .venv
.venv/bin/pip install -r ../requirements.txt -r ../requirements_ui.txt
.venv/bin/streamlit run streamlit_app/demo_v0.py
\`\`\`

Or just visit the live demo at <https://csliu-toolbox.streamlit.app>.
```

**Fix cost**: ~15 min (write + verify the commands work).

---

### §5.2 [MEDIUM] "How do I reproduce paper Example N?" has no top-level pointer

**Location**: Phase B's 5 paper-reproduction tests live as
`python/test_paper_example{1..5}.py`. They are runnable directly
(`.venv/bin/python test_paper_example3.py`) but there's no README /
docs hint that they exist.

A reader of Liu 2017 visiting this repo to verify paper Example 3
has two options:

- **Run a script** — needs to discover `python/test_paper_example*.py` exists.
- **Use the demo** — needs to navigate sidebar → Tensor Eigenvalue → 7th
  tab "Paper Examples" → dropdown → Example 3.

Neither is signposted from the entry doc.

**Suggestion**: Add a `## Reproducing paper Liu 2017 §7` section
to the README:

```markdown
## Reproducing paper Liu 2017 §7

Five examples from Liu, Guo, Lin (Numer. Math. 2017) §7 are
reproduced in this toolbox. Three ways to inspect:

1. **Live demo**: <https://csliu-toolbox.streamlit.app> →
   Tensor Eigenvalue Problem → Paper Examples (Liu 2017) tab.
2. **Run reproduction tests**:
   \`.venv/bin/python python/test_paper_example1.py\` etc.
3. **Read the audit**:
   \`docs/papers/liu2017_alignment_audit.md\`.
```

**Fix cost**: ~15 min.

---

### §5.3 [LOW] requirements files convention is undocumented

**Location**: Repo root has `requirements.txt` and `requirements_ui.txt`
side by side. No comment / README explains:

- `requirements.txt` = core (numpy, scipy) — what `tensor_utils.py` needs to import.
- `requirements_ui.txt` = extras for the demo (streamlit, plotly).

A user pip-installing only `requirements.txt` then trying to run
`streamlit run …` would hit a missing-streamlit error.

**Impact**: Mild install confusion.

**Suggestion**: Either combine into a single `requirements.txt`
(simpler), or add comments at the top of each file:

```
# requirements_ui.txt
# Extras for the Streamlit demo. Combine with requirements.txt:
#   pip install -r requirements.txt -r requirements_ui.txt
streamlit
plotly
```

**Fix cost**: ~5 min.

---

### §5.4 [LOW] Demo "Build SHA" footer pattern is undocumented for visitors

**Location**: `demo_v0.py` lines 46–64 + 112 — sidebar shows
`Build: <sha>` caption that's reset by Streamlit Cloud. The
mechanism + fallback is documented in `README.md` lines 154–199 but
that section is for **deployers**, not visitors.

**Impact**: A visitor seeing `Build: 89e09a3` doesn't know what to
do with it. Not actionable, but mildly mysterious.

**Suggestion**: Wrap the caption in a tooltip via `st.caption(...,
help="Short SHA of the deployed commit. Compare against the latest
commit at github.com/chingsungliu/tensor-eigenvalue-toolbox to
verify the demo is up to date.")`.

**Fix cost**: ~5 min.

---

## §6 Priority matrix

| Severity | Low fix cost (≤30 min) | High fix cost (≥30 min) |
|---|---|---|
| **HIGH** | §1.1 (renderer pattern, 5–45 min depending on option) · §2.1 (where-to-start, 15 min) · §2.2 (Q7 explanation, 15 min) · §5.1 (Quick start, 15 min) | §3.1 (README rewrite, ~2 h) · §4.1 (UI smoke test, 45 min) |
| **MEDIUM** | §1.2 (CASE_TOL DRY, 15 min) · §1.3 (try/except, 10 min) · §2.3 (halving labels, 5 min) · §2.4 (string expected_nit UX, 10 min) · §3.2 (requirements dedupe, 10 min) · §5.2 (reproduce-paper pointer, 15 min) | §3.3 (test guide, 30 min) · §4.2 (uploads test, 45 min) · §4.3 (edge-case rejection tests, 20 min — borderline) |
| **LOW** | §1.4 (module docstring, 5 min) · §1.5 (RNG seed doc, 5 min) · §2.5 (eigenvector display, 3 min) · §2.6 (bilingual convention note, 10 min) · §3.4 (audit doc TOC, 5 min) · §3.5 (LICENSE / CITATION drafts, 20 min) · §4.4 (GMRES sanity, 20 min) · §5.3 (requirements comments, 5 min) · §5.4 (build SHA tooltip, 5 min) | (none in this cell) |

**Quick-win batch** (all HIGH severity, ≤30 min each): §1.1 (option A
docstring), §2.1, §2.2, §5.1 — together ~50 min, addresses the most
visible UX issues and the renderer-inconsistency footnote.

**Worth scheduling**: §3.1 (README rewrite) and §4.1 (UI smoke test) —
both ~45–120 min, both pay back across many future sessions
(README is the public face, UI smoke test prevents Phase C-style
silent regressions).

**Skip / known limitation**: None in this audit. All issues are
addressable; severity gradient reflects priority, not skip status.

---

## §7 Recommendations

### §7.1 First-pass focus: README + onboarding (combined ~3h)

The README is the most visible artifact and the most stale. Pair the
rewrite (§3.1) with §5.1 (Quick start) and §5.2 (paper reproduction
pointer) as a single coherent edit. Side-effect: §5.3 (requirements
comments) becomes natural to fix in the same pass.

After this, a `git clone` visitor lands on coherent text that explains
what the repo does, links to the live demo, and shows how to
reproduce a paper example in three ways.

### §7.2 Second-pass: UI consistency batch (~1.5h)

Group the renderer-related issues:

- §1.1 (renderer pattern alignment — pick option A or B).
- §2.1 (sidebar grouping / "new here?" caption).
- §2.2 (Q7 help tooltip).
- §2.3 (halving label friendliness).
- §2.4 (Example 3 / 4 expected-iter UX).

These form a single "demo polish" PR. Together they materially
improve the first-time-visitor experience.

### §7.3 Third-pass: Test infrastructure (~1.5h)

- §4.1 (UI smoke test).
- §4.2 (uploads test).
- §3.3 (`run_all_tests.sh` or pytest entry point).

A consolidated test pass means future renderer / parser regressions
get caught in CI before reaching the live demo.

### §7.4 Don't fix yet: §3.5 (LICENSE / CITATION)

These are external-facing artifacts. Worth doing eventually but the
upstream user choice (which license? Zenodo DOI now or after first
release?) is the blocker, not implementation effort. Defer until the
toolbox has a "v1.0" milestone the user wants to attach citations to.

### §7.5 Process meta: keep this audit doc as a baseline

Phase E sets a baseline. Future sessions can:

- Update issue status (resolved / deferred / superseded) by editing
  the matrix in §6.
- Add §8 / §9 for Phase F / G audits as the toolbox grows.
- Use the cost estimates as a sanity check when scoping new work
  ("we're spending 4h on §1.1 — was the original estimate 5–45 min wrong?").

This audit doc is part of the historical record (like
`liu2017_alignment_audit.md`), not a one-shot deliverable.

---

## Appendix A — Issue counts by dimension

| Dimension | HIGH | MEDIUM | LOW | Total |
|---|---:|---:|---:|---:|
| §1 Code quality | 1 | 2 | 2 | 5 |
| §2 UX / demo | 2 | 2 | 2 | 6 |
| §3 Documentation | 1 | 2 | 2 | 5 |
| §4 Test coverage | 1 | 2 | 1 | 4 |
| §5 Onboarding | 1 | 1 | 2 | 4 |
| **Total** | **6** | **9** | **9** | **24** |

## Appendix B — Files touched by this audit

Read-only. The single artifact added is this document
(`docs/audits/phase_e_audit_2026-05-05.md`). Working tree has
exactly one new file after Phase E.
