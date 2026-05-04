"""UI integration smoke for the Streamlit demo (Phase E §4.1).

Drives ``streamlit_app/demo_v0.py`` via :class:`streamlit.testing.v1.AppTest`
and asserts the demo renders cleanly across:

1. **Initial render** — ``demo_v0.py`` loads without exception.
2. **Sidebar grouping** (Phase E Stage 1, §2.1) — the
   "新手建議從 Paper Examples (Liu 2017) 開始" caption is present and the
   algorithm radio carries 7 ``captions=`` strings, each formatted as
   ``"<group> · <role>"``.
3. **Tab navigation** — switching to each of the 7 algorithm entries
   re-renders without exception.
4. **Q7 explainer expander** (Phase E Stage 1, §2.2) — present on the
   6 Q7-using renderers (Multi / HONI / NNI + 3 ``*_compare`` tabs) and
   correctly absent on Paper Examples (which doesn't use the Q7 path).
5. **Paper Examples dispatch** (Phase C, Day 17) — drives all 5 example
   builders via the dropdown + Run NNI button, asserts each example's
   ``nit`` matches the expected value from the standalone
   ``test_paper_example*.py`` tests.

Run::

    .venv/bin/python test_demo_smoke.py

Note on the AppTest "missing ScriptRunContext" warning: AppTest runs the
Streamlit script in bare mode (no real session); the warning is
benign and emitted by Streamlit's ``runtime.scriptrunner_utils`` at
import time. Suppress with ``2>/dev/null`` or filter via ``warnings``
if running interactively.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from streamlit.testing.v1 import AppTest


SCRIPT = str(_PYTHON_DIR / "streamlit_app" / "demo_v0.py")
Q7_EXPANDER_LABEL = "ℹ️ About the default Q7 tensor"

# Expected nit per Paper Examples dropdown — derived from the
# standalone test_paper_example*.py outputs at this commit. Off-by-one
# from paper Table values is documented in each test's docstring (RNG
# differences in V trap potential / initial vector).
EXPECTED_NIT_PER_EXAMPLE = {
    "Example 1: Wind power MTD (m=4, n=4)": 5,
    "Example 2: Signless Laplacian (Table 1)": 8,            # default case (3, 20)
    "Example 3: Halving demo (m=4, n=20)": 18,               # canonical θ=1
    "Example 4: Weakly irreducible non-primitive (m=3, n=3)": 6,   # seed=42
    "Example 5: Z-tensor smallest eigenpair (Table 2)": 4,   # default case (3, 20), seed=42
}

Q7_USING_RENDERERS = [
    "Multi",
    "HONI",
    "NNI",
    "HONI vs NNI comparison",
    "Eigenvalue solver comparison (HONI / NNI, multi-run)",
    "Multilinear solver comparison (Multi, multi-run)",
]
PAPER_EXAMPLES_KEY = "Paper Examples (Liu 2017)"


def _run(at: AppTest, *, label: str = "") -> AppTest:
    at.run(timeout=60)
    if at.exception:
        excs = "\n".join(str(e.value) for e in at.exception)
        raise AssertionError(
            f"AppTest raised during {label or 'run'}:\n{excs}"
        )
    return at


def _find_algorithm_radio(at: AppTest):
    """Return the algorithm radio (the one carrying Paper Examples)."""
    for r in at.sidebar.radio:
        if PAPER_EXAMPLES_KEY in r.options:
            return r
    raise AssertionError(
        f"Algorithm radio not found; sidebar radios = "
        f"{[(r.label, list(r.options)) for r in at.sidebar.radio]}"
    )


def test_initial_render():
    at = AppTest.from_file(SCRIPT)
    _run(at, label="initial render")
    print("PASS  initial render")


def test_sidebar_grouping():
    """Phase E §2.1: caption + 7 grouped captions on the algorithm radio."""
    at = AppTest.from_file(SCRIPT)
    _run(at, label="sidebar grouping")

    sidebar_caption_text = "\n".join(c.value for c in at.sidebar.caption)
    assert "新手建議" in sidebar_caption_text, (
        "missing '新手建議從 Paper Examples ...' caption in sidebar; "
        f"sidebar captions seen = {[c.value for c in at.sidebar.caption]}"
    )

    alg_radio = _find_algorithm_radio(at)
    captions = list(getattr(alg_radio, "captions", []) or [])
    assert len(captions) == 7, (
        f"expected 7 captions on algorithm radio, got {len(captions)}: {captions}"
    )
    assert all("·" in c for c in captions), (
        "every algorithm caption should carry the '<group> · <role>' "
        f"separator, got {captions}"
    )

    print("PASS  sidebar grouping (caption + 7 algorithm captions)")


def test_seven_tabs_navigate():
    """Phase E Stage 1 sanity: each of the 7 tabs re-renders without exception."""
    at = AppTest.from_file(SCRIPT)
    _run(at, label="tab navigation initial")

    options = list(_find_algorithm_radio(at).options)
    assert len(options) == 7, f"expected 7 algorithm options, got {len(options)}"

    for key in options:
        # Re-grab the radio each iteration: AppTest references can drift
        # across reruns.
        _find_algorithm_radio(at).set_value(key)
        _run(at, label=f"switch to {key!r}")
    print("PASS  7 tabs navigate without exception")


def test_q7_expander_visibility():
    """Phase E §2.2: Q7 expander on 6 Q7-using renderers, absent on Paper Examples."""
    at = AppTest.from_file(SCRIPT)
    _run(at, label="q7 expander initial")

    for key in Q7_USING_RENDERERS:
        _find_algorithm_radio(at).set_value(key)
        _run(at, label=f"q7 expander on {key!r}")
        labels = [e.label for e in at.expander]
        assert Q7_EXPANDER_LABEL in labels, (
            f"missing Q7 explainer expander on tab {key!r}; expanders "
            f"present = {labels}"
        )

    _find_algorithm_radio(at).set_value(PAPER_EXAMPLES_KEY)
    _run(at, label="paper examples q7 absence")
    labels = [e.label for e in at.expander]
    assert Q7_EXPANDER_LABEL not in labels, (
        f"Paper Examples tab should not carry the Q7 expander but did; "
        f"expanders present = {labels}"
    )
    print(
        "PASS  Q7 expander present on 6 Q7-using renderers, "
        "absent on Paper Examples"
    )


def test_paper_examples_dispatch():
    """Phase C (Day 17): drive all 5 paper examples via dropdown + Run."""
    at = AppTest.from_file(SCRIPT)
    _run(at, label="paper examples initial")

    _find_algorithm_radio(at).set_value(PAPER_EXAMPLES_KEY)
    _run(at, label="switch to Paper Examples tab")

    example_sel = next(
        sb for sb in at.selectbox if sb.label == "Select example"
    )
    options = list(example_sel.options)
    assert set(options) == set(EXPECTED_NIT_PER_EXAMPLE.keys()), (
        f"paper example dropdown options mismatch; got {options}, "
        f"expected {sorted(EXPECTED_NIT_PER_EXAMPLE.keys())}"
    )

    for label, expected_nit in EXPECTED_NIT_PER_EXAMPLE.items():
        # Re-grab widgets each iteration for the same drift reason as
        # _find_algorithm_radio above.
        sel = next(sb for sb in at.selectbox if sb.label == "Select example")
        sel.set_value(label)
        _run(at, label=f"select {label!r}")

        # Examples 4 / 5 expose a "Random seed" radio; pin to the Fixed
        # option so the assertion is deterministic.
        for radio in at.radio:
            if radio.label == "Random seed":
                fixed_opt = next(
                    o for o in radio.options if "Fixed" in str(o)
                )
                radio.set_value(fixed_opt)
                _run(at, label=f"pin seed for {label!r}")
                break

        run_btn = next(btn for btn in at.button if btn.label == "Run NNI")
        run_btn.click()
        _run(at, label=f"run NNI for {label!r}")

        success_msgs = [el.value for el in at.success]
        assert success_msgs, (
            f"Run NNI on {label!r} produced no st.success element; "
            f"expanders / errors present = "
            f"{[e.label for e in at.expander]} / "
            f"{[e.value for e in at.error]}"
        )
        match = re.search(r"in (\d+) iterations", success_msgs[-1])
        assert match, (
            f"unexpected success message for {label!r}: {success_msgs[-1]!r}"
        )
        got_nit = int(match.group(1))
        assert got_nit == expected_nit, (
            f"{label!r}: expected nit = {expected_nit}, got nit = {got_nit}. "
            f"If this fails, the standalone test_paper_example*.py "
            f"may also have shifted — check that first."
        )
    print(
        f"PASS  Paper Examples dispatch ({len(EXPECTED_NIT_PER_EXAMPLE)} "
        f"examples, all nit match standalone tests)"
    )


def main():
    print("test_demo_smoke — Streamlit demo UI smoke (AppTest)")
    print("=" * 72)
    test_initial_render()
    test_sidebar_grouping()
    test_seven_tabs_navigate()
    test_q7_expander_visibility()
    test_paper_examples_dispatch()
    print()
    print("All Phase E §4.1 demo smoke checks passed.")


if __name__ == "__main__":
    main()
