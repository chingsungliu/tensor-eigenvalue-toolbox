"""Layer 3 algorithm renderers for the Tensor Eigenvalue Problem.

Exposes four Streamlit renderers via the ``ALGORITHMS`` dict:
  - ``Multi``              : multilinear Newton solver (HNI base layer)
  - ``HONI``               : shift-invert eigenvalue iteration
  - ``NNI``                : single-layer Newton (user's primary algorithm)
  - ``HONI vs NNI comparison`` : cross-algorithm side-by-side

All four default to the Q7-style sparse M-tensor built by
:func:`streamlit_app.problems.tensor_eigenvalue.defaults.build_q7_tensor`.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from streamlit_app._internal.snippets import read_snippet
from streamlit_app.problems.tensor_eigenvalue.defaults import build_q7_tensor
from streamlit_app.problems.tensor_eigenvalue.uploads import load_tensor_file
from tensor_utils import honi, multi, nni


# ---------------------------------------------------------------------------
# Plot / summary helpers (Layer-3 only)
# ---------------------------------------------------------------------------


def _plot_log_history(
    series: dict,
    title: str,
    x_label: str = "iteration",
    y_label: str = "value",
    log_y: bool = True,
) -> None:
    """Line chart for one or more per-iteration history arrays.

    ``series`` maps legend label → 1-D array. When ``log_y=True`` (default,
    suited to residuals), non-positive entries are masked out. When
    ``log_y=False`` (suited to slowly-varying quantities like λ_U), values
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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


def _plot_grouped_bar(
    series: dict,
    title: str,
    x_label: str = "component",
    y_label: str = "value",
) -> None:
    """Grouped bar chart for two or more vectors over a shared index.

    `series` maps legend label → 1-D array. All arrays must share the
    same length; bars from each series are placed side-by-side at each
    index. Used for cross-algorithm eigenvector comparison.
    """
    fig = go.Figure()
    for name, arr in series.items():
        arr = np.asarray(arr).flatten()
        fig.add_trace(go.Bar(
            x=list(range(len(arr))),
            y=list(arr),
            name=name,
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)


def _gmres_info_summary(info_history) -> str:
    """Describe gmres_info_history: total iter + count/locations of non-zero info.

    scipy ``gmres`` returns ``info``: 0 = converged, >0 = not converged in maxiter,
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


# ---------------------------------------------------------------------------
# Data-source helpers (Q7 default / Upload)
# ---------------------------------------------------------------------------


def _fallback_initial_vector(n: int) -> np.ndarray:
    """Default positive initial vector used when an upload omits x0."""
    rng = np.random.default_rng(42)
    return np.abs(rng.random(n)) + 0.1


def _render_data_source_block(renderer_key: str):
    """Render Data source radio + file_uploader OUTSIDE any ``st.form``.

    Returns ``(is_upload, upload_data)``:
      - ``is_upload``  : True if user selected the Upload option
      - ``upload_data``: parsed tensor dict on success, else None. Errors are
        surfaced as ``st.error`` before return; callers should disable the
        Run button when ``is_upload`` is True but ``upload_data`` is None.
    """
    source = st.radio(
        "Data source",
        options=["Q7 default (n=20, m=3)", "Upload (.mat / .npz)"],
        key=f"data_source_{renderer_key}",
    )
    is_upload = source.startswith("Upload")
    if not is_upload:
        return False, None

    uploaded = st.file_uploader(
        "Tensor file",
        type=["mat", "npz"],
        key=f"upload_{renderer_key}",
        help=(
            "Reserved keys tried first: **AA** (2-D mode-1 unfolding) or "
            "**A_tensor** (m-D full tensor); **x0** (optional, 1-D length n). "
            "Auto-detects any ndim≥2 array if reserved names are absent."
        ),
    )
    if uploaded is None:
        st.info(
            "請上傳 .mat 或 .npz 檔。"
            "Reserved 變數名：`AA` / `A_tensor` / `x0`；找不到會自動偵測。"
        )
        return True, None

    try:
        data = load_tensor_file(uploaded)
    except ValueError as e:
        st.error(f"Upload failed — {e}")
        return True, None

    st.success(f"✓ Loaded — {data['source_info']}")
    return True, data


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


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
        is_upload, upload_data = _render_data_source_block("multi")

        with st.form("multi_form"):
            if is_upload and upload_data is not None:
                m = st.number_input(
                    "m (from upload)", value=upload_data["m"], disabled=True
                )
                n = st.number_input(
                    "n (from upload)", value=upload_data["n"], disabled=True
                )
            elif is_upload:
                st.caption("⏳ Waiting for file upload to populate m and n…")
                m, n = 3, 20
            else:
                m = st.number_input(
                    "m (tensor order)", min_value=2, max_value=5, value=3, step=1
                )
                n = st.number_input(
                    "n (per-mode size)", min_value=5, max_value=50, value=20, step=1
                )
            tol = st.number_input("tol", value=1e-10, format="%.0e")
            submitted = st.form_submit_button(
                "Run multi",
                type="primary",
                disabled=(is_upload and upload_data is None),
            )

        if submitted:
            if is_upload:
                AA = upload_data["AA"]
                b = (
                    upload_data["x0"]
                    if upload_data["x0"] is not None
                    else _fallback_initial_vector(int(n))
                )
                source_tag = f"Upload — {upload_data['source_info']}"
            else:
                AA, b = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
                source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"
            try:
                u, nit, hal, history = multi(
                    AA, b, int(m), float(tol), record_history=True
                )
                st.session_state["multi_result"] = {
                    "u": u, "nit": nit, "hal": hal, "history": history,
                    "n": int(n), "m": int(m), "tol": float(tol),
                    "source_tag": source_tag,
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

            st.caption(f"Data source: {result.get('source_tag', 'Q7 default')}")

            _plot_log_history(
                {"residual": history["res_history"]},
                title="Residual convergence",
                y_label="‖A·u^(m-1) - b^(m-1)‖",
            )

            _plot_bar_vector(
                history["hal_history"],
                title="Halving steps per outer iter",
                x_label="outer iteration",
                y_label="halving count",
            )
            st.caption(
                "**讀圖要點**：Multi 在每外層 Newton iter 嘗試的 halving 次數；"
                "Q7 well-conditioned 通常 0、ill-conditioned case 才會 > 0 "
                "（見 `feedback_multi_halving_fragility.md`）。"
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
        is_upload, upload_data = _render_data_source_block("honi")

        with st.form("honi_form"):
            if is_upload and upload_data is not None:
                m = st.number_input(
                    "m (from upload)", value=upload_data["m"], disabled=True
                )
                n = st.number_input(
                    "n (from upload)", value=upload_data["n"], disabled=True
                )
            elif is_upload:
                st.caption("⏳ Waiting for file upload to populate m and n…")
                m, n = 3, 20
            else:
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
            submitted = st.form_submit_button(
                "Run honi",
                type="primary",
                disabled=(is_upload and upload_data is None),
            )

        if submitted:
            if is_upload:
                AA = upload_data["AA"]
                x0 = (
                    upload_data["x0"]
                    if upload_data["x0"] is not None
                    else _fallback_initial_vector(int(n))
                )
                source_tag = f"Upload — {upload_data['source_info']}"
            else:
                AA, x0 = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
                source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"
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
                    "source_tag": source_tag,
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
            history = result["history"]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("outer nit", outer_nit)
            mc2.metric("total inner nit", innit)
            mc3.metric("final λ", f"{lam:.6f}")
            mc4.metric("final residual", f"{float(outer_res[-1]):.3e}")

            st.caption(f"solver branch: `{result['linear_solver']}`")
            st.caption(f"Data source: {result.get('source_tag', 'Q7 default')}")

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
            _plot_bar_vector(
                history["innit_history"],
                title="Multi inner iterations per outer iter",
                x_label="outer iteration",
                y_label="inner iter count",
            )
            st.caption(
                "**讀圖要點**：HONI 外層每步呼叫 Multi 跑 inner Newton iter；"
                "shift-invert 在 `lambda_U → eigenvalue` 尾段 near-singular、"
                "inner iter 顯著增加（fragility 來源、見 "
                "`feedback_honi_multi_fragility_propagation.md`）。"
            )

            _plot_bar_vector(x, title="Final x (eigenvector)", x_label="component")

    with st.expander("📄 對應的 MATLAB 原始碼"):
        st.code(read_snippet("honi"), language="matlab")


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
        is_upload, upload_data = _render_data_source_block("nni")

        with st.form("nni_form"):
            if is_upload and upload_data is not None:
                m = st.number_input(
                    "m (from upload)", value=upload_data["m"], disabled=True
                )
                n = st.number_input(
                    "n (from upload)", value=upload_data["n"], disabled=True
                )
            elif is_upload:
                st.caption("⏳ Waiting for file upload to populate m and n…")
                m, n = 3, 20
            else:
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
            submitted = st.form_submit_button(
                "Run nni",
                type="primary",
                disabled=(is_upload and upload_data is None),
            )

        if submitted:
            if is_upload:
                AA = upload_data["AA"]
                x0 = (
                    upload_data["x0"]
                    if upload_data["x0"] is not None
                    else _fallback_initial_vector(int(n))
                )
                source_tag = f"Upload — {upload_data['source_info']}"
            else:
                AA, x0 = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
                source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"
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
                    "source_tag": source_tag,
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
            st.caption(f"Data source: {result.get('source_tag', 'Q7 default')}")

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

            min_x_per_iter = np.min(history["x_history"], axis=0)
            _plot_log_history(
                {"min(x_i)": min_x_per_iter},
                title="min(x_i) per iteration",
                y_label="min(x_i)",
            )
            st.caption(
                "**讀圖要點**：`min(x_i)` 接近 0 時 Rayleigh quotient noise floor 觸發 — "
                "停止條件浮點抽籤量級 `ε · n^(m-1) / min(x_i)^(m-1)` 主導 "
                "（見 `feedback_nni_rayleigh_quotient_noise_floor.md` 第 10 條 memory + "
                "`docs/papers/rayleigh_quotient_noise_floor_en.md` §4）。"
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
        is_upload, upload_data = _render_data_source_block("cmp")

        with st.form("cmp_form"):
            if is_upload and upload_data is not None:
                m = st.number_input(
                    "m (from upload)", value=upload_data["m"], disabled=True
                )
                n = st.number_input(
                    "n (from upload)", value=upload_data["n"], disabled=True
                )
            elif is_upload:
                st.caption("⏳ Waiting for file upload to populate m and n…")
                m, n = 3, 20
            else:
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
            submitted = st.form_submit_button(
                "Run HONI + NNI",
                type="primary",
                disabled=(is_upload and upload_data is None),
            )

        if submitted:
            if is_upload:
                AA = upload_data["AA"]
                x0 = (
                    upload_data["x0"]
                    if upload_data["x0"] is not None
                    else _fallback_initial_vector(int(n))
                )
                source_tag = f"Upload — {upload_data['source_info']}"
            else:
                AA, x0 = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
                source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"
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
                    "source_tag": source_tag,
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
            st.caption(f"Data source: {result.get('source_tag', 'Q7 default')}")
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

                _plot_log_history(
                    {
                        "HONI λ": h["lam_hist"],
                        "NNI λ_U": nn["lam_U_hist"],
                        "NNI λ_L": nn["lam_L_hist"],
                    },
                    title="λ trajectory — HONI vs NNI (linear y)",
                    y_label="λ",
                    log_y=False,
                )
                st.caption(
                    "**讀圖要點**：HONI 用雙層 shift-invert、NNI 用單層 Newton bracket — "
                    "兩演算法走完全不同的 Newton 路徑、最終都收斂到同一 λ。"
                )

                # Sign-align NNI eigenvector to HONI for component-wise comparison.
                sign = 1.0 if float(np.dot(h["x"], nn["x"])) >= 0 else -1.0
                nn_x_aligned = sign * nn["x"]
                _plot_grouped_bar(
                    {"HONI x": h["x"], "NNI x (sign-aligned)": nn_x_aligned},
                    title="Eigenvectors — HONI vs NNI (sign-aligned)",
                    x_label="component",
                )
                st.caption(
                    "**讀圖要點**：兩演算法的 eigenvector component-wise 一致 — "
                    f"`‖x_HONI − x_NNI‖₂ = {x_diff:.2e}` machine-epsilon 內 cross-validate。"
                )

    with st.expander("📄 對應的 MATLAB 原始碼 — HONI.m + NNI.m"):
        left, right = st.columns(2)
        with left:
            st.markdown("**HONI.m**")
            st.code(read_snippet("honi"), language="matlab")
        with right:
            st.markdown("**NNI.m**")
            st.code(read_snippet("nni"), language="matlab")


def render_eigenvalue_compare() -> None:
    st.header("Eigenvalue solver comparison — 2-3 runs side-by-side")
    st.caption(
        "並排執行 2 或 3 個 HONI / NNI runs，每個 run 獨立選演算法、tol、maxit、solver。"
        "所有 runs 共用同一個 AA + 初始向量，便於比較不同 tol 下的 noise floor 行為、"
        "或 HONI 雙層 shift-invert vs NNI 單層 Newton 的收斂路徑差異。"
    )
    st.caption(
        "📎 比舊的「HONI vs NNI comparison」tile 更 general："
        "2-run 固定 HONI+NNI 版本聚焦單一敘事；這個 renderer 適合 `tol` 掃 / noise floor 研究 / "
        "同一演算法不同 solver 對比（spsolve vs gmres 在近奇異系統的穩定度）。"
    )

    # ------------------------- Section 1: Shared config -------------------------
    st.subheader("Shared configuration")
    col_ds, col_dim, col_nruns = st.columns([2, 1, 1])

    with col_ds:
        is_upload, upload_data = _render_data_source_block("cmp_eig")

    with col_dim:
        if is_upload and upload_data is not None:
            m = st.number_input(
                "m (from upload)", value=upload_data["m"], disabled=True,
                key="cmp_eig_m_up",
            )
            n = st.number_input(
                "n (from upload)", value=upload_data["n"], disabled=True,
                key="cmp_eig_n_up",
            )
        elif is_upload:
            st.caption("⏳ Waiting for upload…")
            m, n = 3, 20
        else:
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1,
                key="cmp_eig_m_q7",
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1,
                key="cmp_eig_n_q7",
            )

    with col_nruns:
        n_runs = st.radio(
            "Number of runs", options=[2, 3], index=0,
            key="cmp_eig_n_runs", horizontal=True,
        )

    st.divider()

    # ------------------------- Section 2: Per-run config -----------------------
    st.subheader("Per-run configuration")
    run_cfg_cols = st.columns(n_runs)
    runs_cfg = []
    for i, col in enumerate(run_cfg_cols):
        with col:
            st.markdown(f"**Run {i + 1}**")
            # Default: Run 1 = HONI, Run 2 = NNI, Run 3 = NNI (to showcase contrast)
            default_algo = "HONI" if i == 0 else "NNI"
            default_idx = ["HONI", "NNI"].index(default_algo)
            algo = st.radio(
                "algorithm", options=["HONI", "NNI"],
                index=default_idx, horizontal=True,
                key=f"cmp_eig_algo_{i}",
            )
            tol = st.number_input(
                "tol", value=1e-10, format="%.0e", key=f"cmp_eig_tol_{i}",
            )
            maxit = st.number_input(
                "maxit", min_value=10, max_value=1000, value=200, step=10,
                key=f"cmp_eig_maxit_{i}",
            )
            if algo == "HONI":
                solver = st.radio(
                    "solver", options=["exact", "inexact"], index=0,
                    horizontal=True, key=f"cmp_eig_solver_{i}",
                )
            else:
                solver = st.radio(
                    "solver", options=["spsolve", "gmres"], index=0,
                    horizontal=True, key=f"cmp_eig_solver_{i}",
                )
            runs_cfg.append({
                "algorithm": algo, "tol": float(tol),
                "maxit": int(maxit), "solver": solver,
            })

    # ------------------------- Section 3: Run button ---------------------------
    run_disabled = is_upload and upload_data is None
    run_pressed = st.button(
        "Run all configurations", type="primary",
        disabled=run_disabled, use_container_width=True,
    )

    if run_pressed:
        # Resolve AA + initial vector (shared across runs)
        if is_upload:
            AA = upload_data["AA"]
            x0 = (
                upload_data["x0"]
                if upload_data["x0"] is not None
                else _fallback_initial_vector(int(n))
            )
            source_tag = f"Upload — {upload_data['source_info']}"
        else:
            AA, x0 = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"

        runs_out = []
        try:
            for i, cfg in enumerate(runs_cfg):
                label = (
                    f"Run {i + 1}: {cfg['algorithm']} {cfg['solver']} "
                    f"(tol={cfg['tol']:.0e}, maxit={cfg['maxit']})"
                )
                if cfg["algorithm"] == "HONI":
                    lam, x, outer_nit, innit, res, lam_hist, history = honi(
                        AA, int(m), cfg["tol"],
                        linear_solver=cfg["solver"], maxit=cfg["maxit"],
                        initial_vector=x0, record_history=True,
                    )
                    runs_out.append({
                        "label": label, "algorithm": "HONI",
                        "solver": cfg["solver"], "tol": cfg["tol"],
                        "maxit": cfg["maxit"],
                        "lam": float(lam), "x": x,
                        "outer_nit": int(outer_nit), "inner_nit": int(innit),
                        "res": np.asarray(res),
                        "lam_hist": np.asarray(lam_hist),
                    })
                else:  # NNI
                    (lam_U, x, nit, lam_L, res, lam_U_hist, hist_nni) = nni(
                        AA, int(m), cfg["tol"],
                        linear_solver=cfg["solver"], maxit=cfg["maxit"],
                        initial_vector=x0, record_history=True,
                    )
                    runs_out.append({
                        "label": label, "algorithm": "NNI",
                        "solver": cfg["solver"], "tol": cfg["tol"],
                        "maxit": cfg["maxit"],
                        "lam_U": float(lam_U), "lam_L": float(lam_L), "x": x,
                        "nit": int(nit),
                        "res": np.asarray(res),
                        "lam_U_hist": np.asarray(lam_U_hist),
                        "gmres_info": hist_nni.get("gmres_info_history"),
                    })
            st.session_state["cmp_eig_results"] = {
                "source_tag": source_tag,
                "runs": runs_out,
            }
        except Exception as e:
            st.session_state["cmp_eig_results"] = {
                "error": f"{type(e).__name__}: {e}",
            }

    # ------------------------- Section 4: Results ------------------------------
    results = st.session_state.get("cmp_eig_results")
    if results is None:
        st.info(
            "Configure shared AA + per-run solvers above, then press "
            "**Run all configurations**."
        )
    elif "error" in results:
        st.error(f"Comparison failed — {results['error']}")
    else:
        st.divider()
        st.caption(f"Data source: {results['source_tag']}")
        st.subheader("Results — per run")

        runs = results["runs"]
        result_cols = st.columns(len(runs))
        for col, run in zip(result_cols, runs):
            with col:
                st.markdown(f"**{run['label']}**")
                if run["algorithm"] == "HONI":
                    mc1, mc2 = st.columns(2)
                    mc1.metric("outer nit", run["outer_nit"])
                    mc2.metric("inner nit", run["inner_nit"])
                    mc3, mc4 = st.columns(2)
                    mc3.metric("final λ", f"{run['lam']:.6f}")
                    mc4.metric("final res", f"{float(run['res'][-1]):.3e}")
                else:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("nit", run["nit"])
                    mc2.metric("spread", f"{run['lam_U'] - run['lam_L']:.3e}")
                    mc3, mc4 = st.columns(2)
                    mc3.metric("final λ_U", f"{run['lam_U']:.6f}")
                    mc4.metric("final res", f"{float(run['res'][-1]):.3e}")
                    if run["solver"] == "gmres":
                        st.caption(_gmres_info_summary(run["gmres_info"]))

                _plot_log_history(
                    {"residual": run["res"]},
                    title=f"{run['algorithm']} {run['solver']} convergence",
                    y_label="residual",
                )

        st.divider()
        st.subheader("Summary — residual convergence overlay")
        overlay = {run["label"]: run["res"] for run in runs}
        _plot_log_history(
            overlay,
            title="All runs on one log-y plot",
            y_label="residual",
        )
        st.caption(
            "**讀圖要點**：x 軸是 iteration index — HONI 的 outer iter 通常少、"
            "NNI 的 Newton iter 通常多；每步工作量不同（HONI 每外層含 Multi 內層、"
            "NNI 每步解一個 linear system）。同 AA 最終殘差落到同一 noise floor "
            "（量級 `machine_eps · n^(m-1) / min(x_i)^(m-1)`、見研究筆記 §4.6）。"
        )

        final_lams = [
            run["lam"] if run["algorithm"] == "HONI" else run["lam_U"]
            for run in runs
        ]
        short_labels = [f"Run {i + 1}" for i in range(len(runs))]
        fig = go.Figure(go.Bar(
            x=short_labels,
            y=final_lams,
            text=[f"{v:.10g}" for v in final_lams],
            textposition="outside",
        ))
        fig.update_layout(
            title="Final λ across runs (cross-validation)",
            xaxis_title="run",
            yaxis_title="λ",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**讀圖要點**：多 run 跑同一 tensor、final λ 應該 machine-epsilon 內一致 — "
            "bar 高度肉眼相同 + text label 精確值對齊到 ~10 位小數即表示 bit-identical 收斂。"
        )

    with st.expander("📄 對應的 MATLAB 原始碼 — HONI.m + NNI.m"):
        left, right = st.columns(2)
        with left:
            st.markdown("**HONI.m**")
            st.code(read_snippet("honi"), language="matlab")
        with right:
            st.markdown("**NNI.m**")
            st.code(read_snippet("nni"), language="matlab")


def render_multilinear_compare() -> None:
    st.header("Multilinear solver comparison — 2-3 Multi runs (tol sweep)")
    st.caption(
        "並排執行 2 或 3 個 Multi runs，所有 runs 共用同一 `A` + `b`，僅 `tol` 不同。"
        "適合觀察 Multi 的 Newton 收斂在不同停止門檻下的 outer iter / halving 行為。"
    )
    st.caption(
        "⚠️ Multi 在 `m≥3` + random AA 常 trap-and-diverge 在 halving path（設計為 near-optimal "
        "微調、不是大幅 overshoot 修正）— 這是演算法本質、非 port bug。"
        "詳見 `memory/feedback_multi_halving_fragility.md`。"
    )
    st.caption(
        "📎 Multi 的外層迭代上限硬編碼 `nit < 100`（MATLAB `Multi.m` 行為）、"
        "無 `maxit` 或 `solver` 參數、所以 per-run 只能 vary `tol`。"
    )

    # ------------------------- Section 1: Shared config -------------------------
    st.subheader("Shared configuration")
    col_ds, col_dim, col_nruns = st.columns([2, 1, 1])

    with col_ds:
        is_upload, upload_data = _render_data_source_block("cmp_multi")

    with col_dim:
        if is_upload and upload_data is not None:
            m = st.number_input(
                "m (from upload)", value=upload_data["m"], disabled=True,
                key="cmp_multi_m_up",
            )
            n = st.number_input(
                "n (from upload)", value=upload_data["n"], disabled=True,
                key="cmp_multi_n_up",
            )
        elif is_upload:
            st.caption("⏳ Waiting for upload…")
            m, n = 3, 20
        else:
            m = st.number_input(
                "m (tensor order)", min_value=2, max_value=5, value=3, step=1,
                key="cmp_multi_m_q7",
            )
            n = st.number_input(
                "n (per-mode size)", min_value=5, max_value=50, value=20, step=1,
                key="cmp_multi_n_q7",
            )

    with col_nruns:
        n_runs = st.radio(
            "Number of runs", options=[2, 3], index=0,
            key="cmp_multi_n_runs", horizontal=True,
        )

    st.divider()

    # ------------------------- Section 2: Per-run config -----------------------
    st.subheader("Per-run configuration (tol only)")
    default_tols = [1e-8, 1e-10, 1e-12]
    run_cfg_cols = st.columns(n_runs)
    runs_cfg = []
    for i, col in enumerate(run_cfg_cols):
        with col:
            st.markdown(f"**Run {i + 1}**")
            tol = st.number_input(
                "tol", value=default_tols[i], format="%.0e",
                key=f"cmp_multi_tol_{i}",
            )
            runs_cfg.append({"tol": float(tol)})

    # ------------------------- Section 3: Run button ---------------------------
    run_disabled = is_upload and upload_data is None
    run_pressed = st.button(
        "Run all configurations", type="primary",
        disabled=run_disabled, use_container_width=True,
        key="cmp_multi_run",
    )

    if run_pressed:
        if is_upload:
            AA = upload_data["AA"]
            b = (
                upload_data["x0"]
                if upload_data["x0"] is not None
                else _fallback_initial_vector(int(n))
            )
            source_tag = f"Upload — {upload_data['source_info']}"
        else:
            AA, b = build_q7_tensor(n=int(n), m=int(m), rng_seed=42)
            source_tag = f"Q7 default (n={int(n)}, m={int(m)}, rng=42)"

        runs_out = []
        try:
            for i, cfg in enumerate(runs_cfg):
                label = f"Run {i + 1}: Multi (tol={cfg['tol']:.0e})"
                u, nit, hal, history = multi(
                    AA, b, int(m), cfg["tol"], record_history=True,
                )
                total_hal = int(hal[: nit + 1].sum())
                runs_out.append({
                    "label": label,
                    "tol": cfg["tol"],
                    "u": u,
                    "nit": int(nit),
                    "total_hal": total_hal,
                    "res": np.asarray(history["res_history"]),
                })
            st.session_state["cmp_multi_results"] = {
                "source_tag": source_tag,
                "runs": runs_out,
            }
        except Exception as e:
            st.session_state["cmp_multi_results"] = {
                "error": f"{type(e).__name__}: {e}",
            }

    # ------------------------- Section 4: Results ------------------------------
    results = st.session_state.get("cmp_multi_results")
    if results is None:
        st.info(
            "Configure shared AA + per-run tol above, then press "
            "**Run all configurations**."
        )
    elif "error" in results:
        st.error(f"Comparison failed — {results['error']}")
    else:
        st.divider()
        st.caption(f"Data source: {results['source_tag']}")
        st.subheader("Results — per run")

        runs = results["runs"]
        result_cols = st.columns(len(runs))
        for col, run in zip(result_cols, runs):
            with col:
                st.markdown(f"**{run['label']}**")
                mc1, mc2 = st.columns(2)
                mc1.metric("outer nit", run["nit"])
                mc2.metric("total halving", run["total_hal"])
                st.metric("final residual", f"{float(run['res'][-1]):.3e}")
                _plot_log_history(
                    {"residual": run["res"]},
                    title=f"Multi convergence (tol={run['tol']:.0e})",
                    y_label="‖A·u^(m-1) - b^(m-1)‖",
                )

        st.divider()
        st.subheader("Summary — residual convergence overlay")
        overlay = {run["label"]: run["res"] for run in runs}
        _plot_log_history(
            overlay,
            title="All runs on one log-y plot",
            y_label="residual",
        )
        st.caption(
            "**讀圖要點**：較嚴格的 `tol` 會要求外層 Newton 跑更多 iter — "
            "曲線延伸更長。若 tol 低於 Multi 的 inner halving 可達精度、"
            "最終殘差會在同一 floor 震盪（halving path trap 或 line search 找不到下降方向），"
            "iter count 可能直接頂到硬編碼上限 100。"
        )

    with st.expander("📄 對應的 MATLAB 原始碼 — Multi.m"):
        st.code(read_snippet("multi"), language="matlab")


ALGORITHMS = {
    "Multi": render_multi,
    "HONI": render_honi,
    "NNI": render_nni,
    "HONI vs NNI comparison": render_hni_vs_nni,
    "Eigenvalue solver comparison (HONI / NNI, multi-run)": render_eigenvalue_compare,
    "Multilinear solver comparison (Multi, multi-run)": render_multilinear_compare,
}
