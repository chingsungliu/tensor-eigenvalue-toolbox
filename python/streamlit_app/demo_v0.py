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
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.sparse import csr_matrix, issparse

from gaussian_blur import gaussian_blur
from tensor_utils import (
    honi,
    multi,
    nni,
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
# Helpers for Layer 3 renderers (multi / honi / nni)
# ========================================================================


def _build_q7_tensor(n: int, m: int, rng_seed: int = 42):
    """Build a Q7-style sparse M-tensor + positive initial vector.

    Mirrors the parity-test setup shared by Multi Q5 / HONI Q4 / NNI Q7
    reference generators: diagonal `d ~ U[1, 11]` plus sparse perturbation
    (density 0.02, magnitude 0.01), `x0 = |rand(n)| + 0.1`. Keeping the
    Streamlit defaults aligned with the parity cases means the UI exercises
    the same regime documented in `memory/feedback_nni_rayleigh_quotient_noise_floor.md`.

    Returns
    -------
    AA : scipy.sparse.csr_matrix, shape (n, n**(m-1))
        Mode-1 unfolding of the m-order n-dim positive M-tensor.
    x0 : np.ndarray, shape (n,)
        Positive initial vector (``|rand(n)| + 0.1``, all entries > 0).
    """
    rng = np.random.default_rng(rng_seed)
    d = rng.uniform(1.0, 11.0, size=n)
    AA_diag = sp_tendiag(d, m)

    n_cols = n ** (m - 1)
    nnz = max(1, int(round(0.02 * n * n_cols)))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_cols, size=nnz)
    vals = 0.01 * rng.random(nnz)
    pert = csr_matrix((vals, (rows, cols)), shape=(n, n_cols))

    AA = (AA_diag + pert).tocsr()
    x0 = np.abs(rng.random(n)) + 0.1
    return AA, x0


def _plot_log_history(
    series: dict,
    title: str,
    x_label: str = "iteration",
    y_label: str = "value",
    log_y: bool = True,
) -> None:
    """Line chart for one or more per-iteration history arrays.

    `series` maps legend label → 1-D array. When `log_y=True` (default,
    suited to residuals), non-positive entries are masked out. When
    `log_y=False` (suited to slowly-varying quantities like λ_U), values
    are plotted linearly with no masking.
    """
    fig = go.Figure()
    for name, arr in series.items():
        arr = np.asarray(arr, dtype=float)
        y = np.where(arr > 0, arr, np.nan) if log_y else arr
        fig.add_trace(go.Scatter(
            x=list(range(len(arr))),
            y=y,
            mode="lines+markers",
            name=name,
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_type="log" if log_y else "linear",
    )
    st.plotly_chart(fig, width="stretch")


def _plot_bar_vector(
    v,
    title: str,
    x_label: str = "index",
    y_label: str = "value",
) -> None:
    """Bar chart of a 1-D vector (for final u / x eigenvector display)."""
    v = np.asarray(v).flatten()
    fig = go.Figure(go.Bar(x=list(range(len(v))), y=list(v)))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    st.plotly_chart(fig, width="stretch")


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


def _make_sample_image(n: int = 128) -> np.ndarray:
    """128x128 灰階合成測試圖：有銳利邊緣 + 內部方塊 + 隨機黑點，模糊起來明顯。"""
    img = np.full((n, n), 0.9, dtype=float)
    img[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 0.1        # 中央黑方塊
    img[3 * n // 8 : 5 * n // 8, 3 * n // 8 : 5 * n // 8] = 0.9  # 內部白方塊
    rng = np.random.default_rng(42)
    for _ in range(30):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        img[i, j] = 0.0
    return img


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


def render_multi() -> None:
    st.header("multi(AA, b, m, tol)")
    st.caption(
        "求解多線性系統 `A · u^(m-1) = b` 的正解。外層 Newton + 內層三等分（one-third）"
        "halving line search。HNI 線的 Newton solver 基礎層、HONI 會呼叫此 module 做內層。"
    )
    st.caption(
        "⚠️ 外層迭代上限硬編碼 `nit < 100`（MATLAB `Multi.m` 行為）— 無 `maxit` 參數。"
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Inputs")
        with st.form("multi_form"):
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1
            )
            tol = st.number_input("tol", value=1e-10, format="%.0e")
            submitted = st.form_submit_button("Run multi", type="primary")

        if submitted:
            AA, b = _build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            try:
                u, nit, hal, history = multi(
                    AA, b, int(m), float(tol), record_history=True
                )
                st.session_state["multi_result"] = {
                    "u": u, "nit": nit, "hal": hal, "history": history,
                    "n": int(n), "m": int(m), "tol": float(tol),
                }
            except Exception as e:
                st.session_state["multi_result"] = {"error": f"{type(e).__name__}: {e}"}

    with col_out:
        st.subheader("Output")
        result = st.session_state.get("multi_result")
        if result is None:
            st.info("Adjust inputs on the left, then press **Run multi**.")
        elif "error" in result:
            st.error(f"multi failed — {result['error']}")
        else:
            u = result["u"]
            nit = result["nit"]
            hal = result["hal"]
            history = result["history"]
            total_hal = int(hal[: nit + 1].sum())
            final_res = float(history["res_history"][-1])

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("outer iterations (nit)", nit)
            mc2.metric("final residual", f"{final_res:.3e}")
            mc3.metric("total halving steps", total_hal)

            _plot_log_history(
                {"residual": history["res_history"]},
                title="Residual convergence",
                y_label="‖A·u^(m-1) - b^(m-1)‖",
            )
            _plot_bar_vector(u, title="Final u (solution)", x_label="component")

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("multi"), language="matlab")


def render_honi() -> None:
    st.header("honi(A, m, tol, *, linear_solver, maxit, initial_vector)")
    st.caption(
        "求 m-order n-dim tensor 的最大 H-eigenvalue（外層 eigenvalue iteration + "
        "內層 Multi Newton、shift-invert 結構）。`exact` / `inexact` 兩分支數學上求解"
        "同一問題、迭代路徑不同。"
    )
    st.caption(
        "⚠️ `lambda_U → eigenvalue` 尾段 `(λ·I - A)` near-singular、內層 `y` 量級"
        "可爆到 O(10^6)。最終 λ / x 仍 bit-identical（見 `memory/feedback_honi_multi_"
        "fragility_propagation.md`）。"
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Inputs")
        with st.form("honi_form"):
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1
            )
            tol = st.number_input("tol", value=1e-10, format="%.0e")
            maxit = st.number_input(
                "maxit (outer)", min_value=10, max_value=1000, value=200, step=10
            )
            linear_solver = st.radio(
                "linear_solver",
                options=["exact", "inexact"],
                index=0,
                help="exact = inner_tol 寫死 1e-10 + lambda_U 增量更新；"
                     "inexact = inner_tol 動態 + lambda_U 重算",
            )
            submitted = st.form_submit_button("Run honi", type="primary")

        if submitted:
            AA, x0 = _build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            try:
                lam, x, outer_nit, innit, outer_res, lam_hist, history = honi(
                    AA, int(m), float(tol),
                    linear_solver=linear_solver,
                    maxit=int(maxit),
                    initial_vector=x0,
                    record_history=True,
                )
                st.session_state["honi_result"] = {
                    "lam": lam, "x": x, "outer_nit": outer_nit, "innit": innit,
                    "outer_res": outer_res, "lam_hist": lam_hist, "history": history,
                    "linear_solver": linear_solver,
                }
            except Exception as e:
                st.session_state["honi_result"] = {"error": f"{type(e).__name__}: {e}"}

    with col_out:
        st.subheader("Output")
        result = st.session_state.get("honi_result")
        if result is None:
            st.info("Adjust inputs on the left, then press **Run honi**.")
        elif "error" in result:
            st.error(f"honi failed — {result['error']}")
        else:
            lam = result["lam"]
            x = result["x"]
            outer_nit = result["outer_nit"]
            innit = result["innit"]
            outer_res = result["outer_res"]
            lam_hist = result["lam_hist"]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("outer nit", outer_nit)
            mc2.metric("total inner nit", innit)
            mc3.metric("final λ", f"{lam:.6f}")
            mc4.metric("final residual", f"{float(outer_res[-1]):.3e}")

            st.caption(f"solver branch: `{result['linear_solver']}`")

            _plot_log_history(
                {"outer residual": outer_res},
                title="Outer convergence",
                y_label="(λ_U − λ_L) / λ_U",
            )
            _plot_log_history(
                {"λ_U": lam_hist},
                title="λ_U evolution",
                y_label="λ_U",
                log_y=False,
            )
            _plot_bar_vector(x, title="Final x (eigenvector)", x_label="component")

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("honi"), language="matlab")


def _gmres_info_summary(info_history) -> str:
    """Describe gmres_info_history: total iter + count/locations of non-zero info.

    scipy `gmres` returns info: 0 = converged, >0 = not converged in maxiter,
    <0 = illegal input. Index 0 of the history is the initial state (no gmres
    call), so we scan from index 1.
    """
    if info_history is None:
        return "gmres info unavailable"
    info = np.asarray(info_history)
    n_iters = len(info) - 1
    warn_positions = (np.where(info[1:] != 0)[0] + 1).tolist()
    n_warn = len(warn_positions)
    if n_warn == 0:
        return f"gmres: {n_iters} outer iter, 0 warnings"
    return f"gmres: {n_iters} outer iter, {n_warn} warning(s) at iter {warn_positions}"


def render_nni() -> None:
    st.header("nni(A, m, tol, *, linear_solver, maxit, initial_vector)")
    st.caption(
        "**使用者主演算法** — Nonnegative Newton Iteration：求非負 M-tensor 的最大 "
        "H-eigenvalue + 非負 eigenvector。單層 Newton（對比 HONI 雙層 shift-invert）。"
        "Rayleigh-quotient 式 λ 上下界收斂。"
    )
    st.caption(
        "📎 研究筆記：`docs/papers/rayleigh_quotient_noise_floor_en.md` — "
        "描述 `min(x_i) → 0` 時的 noise floor 公式 `ε·n^(m-1)/min(x_i)^(m-1)`。"
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Inputs")
        with st.form("nni_form"):
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1
            )
            tol = st.number_input("tol", value=1e-10, format="%.0e")
            maxit = st.number_input(
                "maxit", min_value=10, max_value=1000, value=200, step=10
            )
            linear_solver = st.radio(
                "linear_solver",
                options=["spsolve", "gmres"],
                index=0,
                help="spsolve = MATLAB-parity 路徑（LU direct）；"
                     "gmres = Python-only 大 sparse 路徑（iterative、預設 maxit=1000 tol=1e-10 restart=20）",
            )
            submitted = st.form_submit_button("Run nni", type="primary")

        if submitted:
            AA, x0 = _build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            try:
                lam_U, x, nit, lam_L, res_hist, lam_U_hist, history = nni(
                    AA, int(m), float(tol),
                    linear_solver=linear_solver,
                    maxit=int(maxit),
                    initial_vector=x0,
                    record_history=True,
                )
                st.session_state["nni_result"] = {
                    "lam_U": lam_U, "lam_L": lam_L, "x": x, "nit": nit,
                    "res_hist": res_hist, "lam_U_hist": lam_U_hist,
                    "history": history, "linear_solver": linear_solver,
                }
            except Exception as e:
                st.session_state["nni_result"] = {"error": f"{type(e).__name__}: {e}"}

    with col_out:
        st.subheader("Output")
        result = st.session_state.get("nni_result")
        if result is None:
            st.info("Adjust inputs on the left, then press **Run nni**.")
        elif "error" in result:
            st.error(f"nni failed — {result['error']}")
        else:
            lam_U = result["lam_U"]
            lam_L = result["lam_L"]
            x = result["x"]
            nit = result["nit"]
            res_hist = result["res_hist"]
            lam_U_hist = result["lam_U_hist"]
            history = result["history"]
            spread = lam_U - lam_L

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("nit", nit)
            mc2.metric("final λ_U", f"{lam_U:.6f}")
            mc3.metric("final λ_L", f"{lam_L:.6f}")
            mc4.metric("spread (λ_U − λ_L)", f"{spread:.3e}")
            mc5.metric("final residual", f"{float(res_hist[-1]):.3e}")

            solver_caption = f"solver branch: `{result['linear_solver']}`"
            if result["linear_solver"] == "gmres":
                solver_caption += " — " + _gmres_info_summary(
                    history.get("gmres_info_history")
                )
            st.caption(solver_caption)

            _plot_log_history(
                {"residual": res_hist},
                title="Residual convergence",
                y_label="(λ_U − λ_L) / λ_U",
            )
            _plot_log_history(
                {"λ_U": history["lambda_U_history"], "λ_L": history["lambda_L_history"]},
                title="Eigenvalue bracket (λ_L ≤ λ_max ≤ λ_U)",
                y_label="λ",
                log_y=False,
            )
            _plot_bar_vector(x, title="Final x (eigenvector)", x_label="component")

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("nni"), language="matlab")


def render_hni_vs_nni() -> None:
    st.header("HNI vs NNI — 同一 AA 跑兩個演算法、比較收斂行為")
    st.caption(
        "兩演算法都求最大 H-eigenvalue，但結構不同：HONI 用**雙層 shift-invert**"
        "（外層更新 λ、內層 Multi Newton 解 `(λ·I − A)·y^(m−1) = x^(m−1)`）；"
        "NNI 用**單層 Newton**（每 iter 解 `(−M_shifted)·y = x^(m−1)`、Rayleigh-quotient 更新 λ）。"
    )
    st.caption(
        "📎 Port / fragility 背景：`docs/algorithms_status.md` §5 parity 總表 + "
        "三種 fragility 模式（halving / shift-invert / Rayleigh-quotient）。"
    )

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Shared Inputs")
        with st.form("cmp_form"):
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1
            )
            tol = st.number_input("tol", value=1e-10, format="%.0e")
            maxit = st.number_input(
                "maxit", min_value=10, max_value=1000, value=200, step=10
            )
            st.divider()
            honi_solver = st.radio(
                "HONI linear_solver",
                options=["exact", "inexact"], index=0,
            )
            nni_solver = st.radio(
                "NNI linear_solver",
                options=["spsolve", "gmres"], index=0,
            )
            submitted = st.form_submit_button("Run HONI + NNI", type="primary")

        if submitted:
            AA, x0 = _build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            try:
                h_lam, h_x, h_nit, h_inn, h_res, h_lam_hist, _ = honi(
                    AA, int(m), float(tol),
                    linear_solver=honi_solver, maxit=int(maxit),
                    initial_vector=x0, record_history=True,
                )
                n_lam_U, n_x, n_nit, n_lam_L, n_res, n_lam_U_hist, n_hist = nni(
                    AA, int(m), float(tol),
                    linear_solver=nni_solver, maxit=int(maxit),
                    initial_vector=x0, record_history=True,
                )
                st.session_state["cmp_result"] = {
                    "honi": {
                        "lam": h_lam, "x": h_x, "nit": h_nit, "inn": h_inn,
                        "res": h_res, "lam_hist": h_lam_hist,
                        "solver": honi_solver,
                    },
                    "nni": {
                        "lam_U": n_lam_U, "lam_L": n_lam_L, "x": n_x, "nit": n_nit,
                        "res": n_res, "lam_U_hist": n_lam_U_hist,
                        "lam_L_hist": n_hist["lambda_L_history"],
                        "gmres_info": n_hist.get("gmres_info_history"),
                        "solver": nni_solver,
                    },
                }
            except Exception as e:
                st.session_state["cmp_result"] = {"error": f"{type(e).__name__}: {e}"}

    with col_out:
        st.subheader("Output")
        result = st.session_state.get("cmp_result")
        if result is None:
            st.info(
                "Adjust shared inputs on the left, then press **Run HONI + NNI** "
                "to solve the same AA with both algorithms."
            )
        elif "error" in result:
            st.error(f"comparison failed — {result['error']}")
        else:
            h = result["honi"]
            nn = result["nni"]
            tab_h, tab_n, tab_cmp = st.tabs(["HONI", "NNI", "Comparison"])

            with tab_h:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("outer nit", h["nit"])
                mc2.metric("total inner nit", h["inn"])
                mc3.metric("final λ", f"{h['lam']:.6f}")
                mc4.metric("final residual", f"{float(h['res'][-1]):.3e}")
                st.caption(f"solver: `{h['solver']}`")
                _plot_log_history(
                    {"outer residual": h["res"]},
                    title="HONI outer convergence",
                    y_label="(λ_U − λ_L) / λ_U",
                )
                _plot_bar_vector(h["x"], title="HONI final x", x_label="component")

            with tab_n:
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("nit", nn["nit"])
                mc2.metric("final λ_U", f"{nn['lam_U']:.6f}")
                mc3.metric("final λ_L", f"{nn['lam_L']:.6f}")
                mc4.metric("spread", f"{nn['lam_U'] - nn['lam_L']:.3e}")
                mc5.metric("final residual", f"{float(nn['res'][-1]):.3e}")
                cap = f"solver: `{nn['solver']}`"
                if nn["solver"] == "gmres":
                    cap += " — " + _gmres_info_summary(nn["gmres_info"])
                st.caption(cap)
                _plot_log_history(
                    {"residual": nn["res"]},
                    title="NNI convergence",
                    y_label="(λ_U − λ_L) / λ_U",
                )
                _plot_log_history(
                    {"λ_U": nn["lam_U_hist"], "λ_L": nn["lam_L_hist"]},
                    title="NNI eigenvalue bracket",
                    y_label="λ",
                    log_y=False,
                )
                _plot_bar_vector(nn["x"], title="NNI final x", x_label="component")

            with tab_cmp:
                delta_lam = abs(h["lam"] - nn["lam_U"])
                x_diff = float(np.linalg.norm(h["x"] - nn["x"]))
                # Eigenvectors are unique up to sign; try the sign that minimizes diff.
                x_diff = min(x_diff, float(np.linalg.norm(h["x"] + nn["x"])))

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("HONI final λ", f"{h['lam']:.8f}")
                mc2.metric("NNI final λ_U", f"{nn['lam_U']:.8f}")
                mc3.metric("|Δλ|", f"{delta_lam:.3e}")

                mc4, mc5, mc6 = st.columns(3)
                mc4.metric("HONI iterations", f"{h['nit']} outer / {h['inn']} inner")
                mc5.metric("NNI iterations", nn["nit"])
                mc6.metric("‖x_HONI − x_NNI‖₂ (sign-aligned)", f"{x_diff:.3e}")

                _plot_log_history(
                    {"HONI outer res": h["res"], "NNI res": nn["res"]},
                    title="Residual convergence — HONI vs NNI (log y)",
                    y_label="residual",
                )

                st.caption(
                    "**讀圖要點**：HONI 外層 iter 數通常遠少於 NNI（雙層 shift-invert "
                    "每步更激進），但每外層 iter 含 Multi 的 inner iter，總工作量需對比 "
                    "`HONI total inner` vs `NNI nit`。兩曲線最終常落到同一 noise floor "
                    "（見研究筆記 §4.6）。"
                )

    with st.expander("📄 對應的 MATLAB 原始碼 — HONI.m + NNI.m"):
        left, right = st.columns(2)
        with left:
            st.markdown("**HONI.m**")
            st.code(read_snippet("honi"), language="matlab")
        with right:
            st.markdown("**NNI.m**")
            st.code(read_snippet("nni"), language="matlab")


# ========================================================================
# Dispatch — this dict is the v0 forward-compat contract.
# Adding a new ported function = add one renderer + one line here.
# ========================================================================

RENDERERS = {
    # Layer 1 / 2 — tensor utilities
    "tenpow": render_tenpow,
    "tpv": render_tpv,
    "sp_tendiag": render_sp_tendiag,
    "ten2mat": render_ten2mat,
    "sp_Jaco_Ax": render_sp_Jaco_Ax,
    # Other
    "gaussian_blur": render_gaussian_blur,
    # Layer 3 — tensor eigenvalue / multilinear solvers
    "multi (tensor eigenvalue)": render_multi,
    "honi (tensor eigenvalue)": render_honi,
    "nni (tensor eigenvalue)": render_nni,
    # Layer 4 — cross-algorithm comparison
    "hni vs nni (comparison)": render_hni_vs_nni,
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
