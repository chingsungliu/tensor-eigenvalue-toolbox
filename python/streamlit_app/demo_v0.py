"""Streamlit demo — problem-driven two-level navigation.

Sidebar flow:
  Step 1  Problem category  (Tensor Eigenvalue / Schrödinger / NCP)
  Step 2  Algorithm         (revealed only when Step 1 picks an active problem)

Layer 1 / 2 tensor utilities and gaussian_blur live in
``streamlit_app._internal.utility_renderers`` and are **not exposed in the UI
router** by design — they remain importable for internal review.

Run from the repo root (so .streamlit/config.toml is picked up):
    python/.venv/bin/streamlit run python/streamlit_app/demo_v0.py
"""
import sys
from pathlib import Path

# streamlit_app/ sits inside python/; make python/ importable so that
# `from tensor_utils import ...` (and `from streamlit_app... import ...`)
# work regardless of cwd.
_python_dir = Path(__file__).resolve().parent.parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

import streamlit as st

from streamlit_app.about import render_about
from streamlit_app.problems.tensor_eigenvalue.algorithms import (
    ALGORITHM_GROUP as TENSOR_EIGENVALUE_ALGORITHM_GROUP,
    ALGORITHMS as TENSOR_EIGENVALUE_ALGORITHMS,
)


def _inject_custom_css() -> None:
    """Inject assets/styles.css into the Streamlit page.

    Layered on top of `.streamlit/config.toml`: heading serif, code mono,
    thin gray borders for metric / form / expander tiles. Called once per
    script run from `main()`.
    """
    css_path = _ASSETS_DIR / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _read_build_sha() -> str:
    """Best-effort short SHA from the repo's ``.git/HEAD``.

    Works on local dev (cwd is the repo) and on Streamlit Cloud (which
    ``git clone``s the repo into the container, leaving ``.git/`` intact).
    Pure file read, no subprocess. Returns ``"unknown"`` on any failure
    so the sidebar footer never crashes the app — the caption is purely
    informational.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        head = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(maxsplit=1)[1]
            ref_path = repo_root / ".git" / ref
            return ref_path.read_text(encoding="utf-8").strip()[:7]
        return head[:7]  # detached HEAD
    except Exception:
        return "unknown"


# Two-level routing table.
# Value = dict[str, callable]  → active problem, expand to algorithm radio.
# Value = None                 → coming-soon placeholder.
PROBLEM_ALGORITHMS: dict = {
    "Tensor Eigenvalue Problem": TENSOR_EIGENVALUE_ALGORITHMS,
    "🚧 Nonlinear Schrödinger Equation (coming soon)": None,
    "🚧 Nonlinear Complementarity Problem (coming soon)": None,
}


def main() -> None:
    st.set_page_config(page_title="Algorithm Toolbox Demo", layout="wide")
    _inject_custom_css()

    with st.sidebar:
        st.title("🧰 Algorithm Toolbox")

        st.markdown("### Step 1 — 問題類別")
        problem = st.radio(
            "problem",
            options=list(PROBLEM_ALGORITHMS.keys()),
            label_visibility="collapsed",
        )

        algorithms = PROBLEM_ALGORITHMS[problem]
        algorithm_key: str | None = None
        if algorithms is not None:
            st.markdown("### Step 2 — 演算法")
            # Phase E §2.1: surface algorithm grouping (Solvers /
            # Comparisons / Paper reproduction) via the radio's
            # `captions=` parameter so first-time visitors see what
            # each entry is for. Tensor Eigenvalue only — other
            # problems (when added) can supply their own grouping
            # via PROBLEM_ALGORITHMS dispatch.
            if problem == "Tensor Eigenvalue Problem":
                st.caption(
                    "📍 **新手建議從 _Paper Examples (Liu 2017)_ 開始** — "
                    "看 5 個 paper §7 examples 在 toolbox 即時重現。"
                )
                option_keys = list(algorithms.keys())
                captions = [
                    f"{TENSOR_EIGENVALUE_ALGORITHM_GROUP[k][0]} · "
                    f"{TENSOR_EIGENVALUE_ALGORITHM_GROUP[k][1]}"
                    for k in option_keys
                ]
                algorithm_key = st.radio(
                    "algorithm",
                    options=option_keys,
                    captions=captions,
                    label_visibility="collapsed",
                )
            else:
                algorithm_key = st.radio(
                    "algorithm",
                    options=list(algorithms.keys()),
                    label_visibility="collapsed",
                )

        st.divider()
        if st.button(
            "ℹ️ About this toolbox",
            use_container_width=True,
            key="sidebar_about_btn",
        ):
            st.session_state["show_about"] = True

        # Build SHA footer — `scripts/check_deploy.py` prints the expected
        # SHA after `git push`; visitors can compare against this caption to
        # confirm Streamlit Cloud has redeployed the latest commit.
        st.caption(f"Build: `{_read_build_sha()}`")

    if st.session_state.get("show_about", False):
        render_about()
        return

    if algorithms is None:
        st.info(
            f"**{problem}** — 這類問題的演算法尚未 port，規劃中。"
            "目前可選的是 Tensor Eigenvalue Problem。"
        )
        return

    algorithms[algorithm_key]()


if __name__ == "__main__":
    main()
