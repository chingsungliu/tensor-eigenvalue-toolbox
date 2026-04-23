"""Layer 1 / 2 tensor utilities + gaussian_blur — hidden from the main UI router.

These renderers remain importable for internal validation, but are not
exposed via ``PROBLEM_ALGORITHMS`` in ``demo_v0.py``. Keeping them here
preserves the original Streamlit demo surface for future internal review.
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from scipy.sparse import issparse

from gaussian_blur import gaussian_blur
from streamlit_app._internal.snippets import read_snippet
from tensor_utils import sp_Jaco_Ax, sp_tendiag, ten2mat, tenpow, tpv


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


def _make_sample_image(n: int = 128) -> np.ndarray:
    """128x128 灰階合成測試圖：有銳利邊緣 + 內部方塊 + 隨機黑點，模糊起來明顯。"""
    img = np.full((n, n), 0.9, dtype=float)
    img[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 0.1
    img[3 * n // 8 : 5 * n // 8, 3 * n // 8 : 5 * n // 8] = 0.9
    rng = np.random.default_rng(42)
    for _ in range(30):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        img[i, j] = 0.0
    return img


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


def render_gaussian_blur() -> None:
    st.header("gaussian_blur(A, sigma)")
    st.caption(
        "2-D 高斯模糊（用 `matlab_compat=True` 對應 MATLAB `conv2` 的零填充 + `ceil(3σ)` 截斷）。"
    )

    uploaded = st.file_uploader(
        "上傳灰階圖片（可選；不上傳就用內建範例）",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        img = np.asarray(Image.open(uploaded).convert("L"), dtype=float) / 255.0
    else:
        img = _make_sample_image()

    sigma = st.slider("sigma (模糊半徑)", 0.5, 5.0, 1.5, step=0.1)

    blurred = gaussian_blur(img, sigma, matlab_compat=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img, width="stretch", clamp=True)
        st.caption(
            f"shape: {img.shape}, dtype: {img.dtype}, "
            f"min/max: {float(img.min()):.3f} / {float(img.max()):.3f}"
        )
    with col2:
        st.subheader(f"Blurred (σ={sigma})")
        st.image(blurred, width="stretch", clamp=True)
        st.caption(
            f"shape: {blurred.shape}, dtype: {blurred.dtype}, "
            f"min/max: {float(blurred.min()):.3f} / {float(blurred.max()):.3f}"
        )

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("gaussian_blur"), language="matlab")


UTILITY_RENDERERS = {
    "tenpow": render_tenpow,
    "tpv": render_tpv,
    "sp_tendiag": render_sp_tendiag,
    "ten2mat": render_ten2mat,
    "sp_Jaco_Ax": render_sp_Jaco_Ax,
    "gaussian_blur": render_gaussian_blur,
}
