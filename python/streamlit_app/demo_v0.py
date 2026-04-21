"""Streamlit Demo v0 — internal validation tool for the 5 ported tensor utilities.

See docs/superpowers/specs/2026-04-21-streamlit-demo-v0-design.md for design.

Run:
    cd python && .venv/bin/streamlit run streamlit_app/demo_v0.py
"""
import sys
from pathlib import Path

# streamlit_app/ sits inside python/; make python/ importable regardless of cwd
# so `from tensor_utils import ...` works no matter where `streamlit run` is invoked.
_python_dir = Path(__file__).resolve().parent.parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import numpy as np
import plotly.express as px
import streamlit as st
from scipy.sparse import issparse

from tensor_utils import (
    sp_Jaco_Ax,
    sp_tendiag,
    ten2mat,
    tenpow,
    tpv,
)

MATLAB_SNIPPETS_DIR = Path(__file__).resolve().parent / "matlab_snippets"


def read_snippet(name: str) -> str:
    """Read a MATLAB source snippet from matlab_snippets/<name>.m."""
    return (MATLAB_SNIPPETS_DIR / f"{name}.m").read_text(encoding="utf-8")


def _get_rng() -> np.random.Generator:
    """Build a fresh Generator from the sidebar-provided RNG seed."""
    seed = st.session_state.get("rng_seed", 42)
    return np.random.default_rng(seed)


def render_output_1d(arr: np.ndarray) -> None:
    """Display a 1-D numpy array: shape + preview + summary stats."""
    st.write(
        f"**shape**: `{arr.shape}`  **dtype**: `{arr.dtype}`  **length**: `{len(arr)}`"
    )

    preview = arr if len(arr) <= 10 else arr[:10]
    st.dataframe(preview, width="content")
    if len(arr) > 10:
        st.caption(f"(顯示前 10 個；完整長度 {len(arr)})")

    st.caption(
        f"summary — max: {float(arr.max()):.6f},  min: {float(arr.min()):.6f},  "
        f"mean: {float(arr.mean()):.6f},  std: {float(arr.std()):.6f}"
    )


def render_output_2d(arr) -> None:
    """Display a 2-D array (dense ndarray or scipy.sparse): shape + heatmap + summary."""
    if issparse(arr):
        dense = arr.toarray()
        st.write(
            f"**shape**: `{arr.shape}`  **dtype**: `{dense.dtype}`  "
            f"**nnz**: `{arr.nnz}` (stored as sparse)"
        )
    else:
        dense = np.asarray(arr)
        st.write(f"**shape**: `{dense.shape}`  **dtype**: `{dense.dtype}`")

    fig = px.imshow(
        dense,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels=dict(x="col", y="row", color="value"),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"summary — max: {float(dense.max()):.6f},  "
        f"min: {float(dense.min()):.6f},  "
        f"mean: {float(dense.mean()):.6f}"
    )


# ========================================================================
# Renderers — one per ported tensor utility
# ========================================================================


def render_tenpow() -> None:
    st.header("tenpow(x, p)")
    st.caption("Kronecker 次方：`x ⊗ x ⊗ ... ⊗ x`，共 p 個 x 相 kron。")

    n = st.slider("n (x length)", 1, 10, 5)
    p = st.slider("p (Kronecker power count)", 0, 4, 3)

    rng = _get_rng()
    x = rng.random(n)

    st.write(f"**input x** (shape `{x.shape}`):")
    st.dataframe(x, width="content")

    y = tenpow(x, p)

    st.subheader("Output")
    render_output_1d(y)

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("tenpow"), language="matlab")


def render_tpv() -> None:
    st.header("tpv(AA, x, m)")
    st.caption("Tensor-vector product：`AA · x^(m-1)`，是 `F(x) = Ax^(m-1)` 的計算。")

    n = st.slider("n (x length, AA rows)", 2, 6, 3)
    m = st.slider("m (tensor order)", 2, 4, 3)

    rng = _get_rng()
    AA = rng.random((n, n ** (m - 1)))
    x = rng.random(n)

    st.write(f"**input AA** shape `{AA.shape}`,  **input x** shape `{x.shape}`")

    y = tpv(AA, x, m)

    st.subheader("Output")
    render_output_1d(y)

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("tpv"), language="matlab")


def render_sp_tendiag() -> None:
    st.header("sp_tendiag(d, m)")
    st.caption("構造 m-order n-dim 對角張量的 mode-1 unfolding（sparse matrix）。")

    n = st.slider("n (vector length)", 2, 5, 3)
    m = st.slider("m (tensor order)", 2, 5, 3)

    rng = _get_rng()
    d = rng.random(n)

    st.write(f"**input d** (shape `{d.shape}`):")
    st.dataframe(d, width="content")

    D = sp_tendiag(d, m)

    st.subheader("Output")
    render_output_2d(D)

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("sp_tendiag"), language="matlab")


def render_ten2mat() -> None:
    st.header("ten2mat(A, k)")
    st.caption(
        "Mode-k unfolding — 把 m-order 張量 A 沿第 k 軸展成 2-D 矩陣（column-major reshape）。"
    )

    n = st.slider("n (per-mode size)", 2, 5, 3)
    m = st.slider("m (tensor order)", 2, 5, 3)
    k = st.slider("k (mode to unfold, 0-based)", 0, m - 1, 0)

    rng = _get_rng()
    A = rng.random((n,) * m)

    st.write(f"**input A** shape `{A.shape}` (m-order n-dim tensor)")

    B = ten2mat(A, k=k)

    st.subheader("Output")
    render_output_2d(B)

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("ten2mat"), language="matlab")


def render_sp_Jaco_Ax() -> None:
    st.header("sp_Jaco_Ax(AA, x, m)")
    st.caption("Jacobian of `F(x) = AA · x^(m-1)`，回傳 `(n, n)` sparse 矩陣。")
    st.caption(
        "⚠️ n×m 受限以避免 sparse kron 過大；n>4 或 m>4 會明顯延遲。"
    )

    n = st.slider("n (x length)", 2, 4, 3)
    m = st.slider("m (tensor order)", 2, 4, 3)

    rng = _get_rng()
    AA = rng.random((n, n ** (m - 1)))
    x = rng.random(n)

    st.write(f"**input AA** shape `{AA.shape}`,  **input x** shape `{x.shape}`")

    J = sp_Jaco_Ax(AA, x, m)

    st.subheader("Output")
    render_output_2d(J)

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("sp_Jaco_Ax"), language="matlab")


# ========================================================================
# Dispatch — this dict is the v0 forward-compat contract.
# Adding a new ported function = add one renderer + one line here.
# ========================================================================

RENDERERS = {
    "tenpow": render_tenpow,
    "tpv": render_tpv,
    "sp_tendiag": render_sp_tendiag,
    "ten2mat": render_ten2mat,
    "sp_Jaco_Ax": render_sp_Jaco_Ax,
}


def main() -> None:
    st.set_page_config(page_title="Tensor Utils Demo v0", layout="wide")

    # v0 internal-preview banner. First thing to remove when audience expands to B/C.
    st.warning(
        "⚠️ Internal preview (v0) — 內部驗證用、不對外發布。如遇錯誤請回報。"
    )

    with st.sidebar:
        st.title("📊 Tensor Utils Demo v0")
        choice = st.selectbox("Function", list(RENDERERS.keys()))

        st.divider()
        st.caption("Shared settings")
        st.number_input("RNG seed", value=42, min_value=0, step=1, key="rng_seed")

    RENDERERS[choice]()


if __name__ == "__main__":
    main()
