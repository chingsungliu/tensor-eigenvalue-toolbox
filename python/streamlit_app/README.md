# Tensor Utils Demo v0（Streamlit）

## 這是什麼

Phase D 中期的 **minimum viable demo** — 給目前已 port 完成的 5 個 tensor 工具
（`tenpow`、`tpv`、`sp_tendiag`、`ten2mat`、`sp_Jaco_Ax`）做一個可在瀏覽器
互動的介面。

**目前只面向作者本人使用**（內部驗證 / port 進度儀表板）。醜沒關係、能跑就好。

對應的完整設計文件：
`docs/superpowers/specs/2026-04-21-streamlit-demo-v0-design.md`

---

## 本機啟動

### 第一次使用（安裝依賴）

在 repo 根目錄 `~/Projects/my-toolbox/`：

```bash
# 啟動既有 venv（或新建一個）
python -m venv python/.venv    # 如尚未建立
source python/.venv/bin/activate

# 安裝核心 + UI 依賴
pip install -r requirements.txt -r requirements_ui.txt
```

### 跑起來

```bash
cd python
.venv/bin/streamlit run streamlit_app/demo_v0.py
```

Browser 會自動開 `http://localhost:8501`。

---

## 功能清單

- **5 個函式**各自有獨立頁面（sidebar selectbox 切換）：
  - `tenpow(x, p)` — Kronecker 次方
  - `tpv(AA, x, m)` — Tensor-vector product
  - `sp_tendiag(d, m)` — Sparse 對角張量的 mode-1 展開
  - `ten2mat(A, k)` — Mode-k unfolding
  - `sp_Jaco_Ax(AA, x, m)` — Jacobian of `Ax^(m-1)`
- **Auto-rerun**：拖 slider 即時看新結果，不需要 Run 按鈕
- **Plotly 互動式 heatmap**（2-D 輸出）：hover 看值、zoom、pan
- **MATLAB 原始碼 expander**（每個函式底部）：顯示對應的 MATLAB 片段（逐字複製自
  `matlab_ref/hni/HONI.m` 和 `Multi.m`）— 這是「Python = MATLAB 的證據」，
  讓想核對 port 正確性的人可以一鍵對照
- **共用 RNG seed**（sidebar）：跨函式切換時保留相同隨機輸入，方便重現

---

## 未來擴充計劃

本 demo 是一條長期發展的起點。預期會依照以下順序長大：

### 短期（v0 → v0.1、擴充現有工具）
- HNI Layer 3 的 `Multi`、`HONI` port 完成後加進 `RENDERERS`
- NNI canonical 版本（待決定）port 完成後加進
- LENT baseline power method 加進

### 中期（v1 — 擴充受眾）
- 對應受眾 B：領域內數學家、合作者
- 加入：自訂輸入（手打 / 上傳 `.mat`）、結果匯出、更完整的說明文字

### 長期（v2 — 對外部署）
- 部署到 HuggingFace Spaces
- 加入計算 quota 機制（避免濫用）
- 完整的 onboarding：術語解釋、範例資料集、錯誤訊息翻譯成人話
- 對應受眾 C：完全沒碰過 tensor eigenvalue 的外部研究者 / 工程師 / 學生

---

## v0 不做的事（明確聲明）

- ❌ **不做網站部署** — v0 純本機執行
- ❌ **不做使用者認證或 quota** — v0 是開個人的機器、只有作者在跑
- ❌ **不做錯誤訊息翻譯** — 讓 Streamlit 原生 traceback 露出來
- ❌ **不做自動化 UI 測試** — 底層函式有 parity test；UI 層靠手動驗證
- ❌ **不做 `matlab_compat` 切換 UI** — 已 port 的 5 個函式的 flag 都是 no-op

---

## 擴充 contract（給未來的我）

若要加新函式進 demo（例如 port 完的 `Multi` / `HONI`）：

1. 在 `demo_v0.py` 新增 `render_<func>()` 函式，遵守既有的 helper 介面
   （`render_output_1d` / `render_output_2d`）
2. 若有對應的 MATLAB 原碼，新增 `matlab_snippets/<func>.m`（逐字複製，
   加上 `% Verbatim from ...` header）
3. 在 `RENDERERS` dict 加一行

**不需要改既有任何 renderer 或 helper**。這是 v0 唯一的向後相容承諾。

其他細節（sidebar 佈局、helper 實作、視覺化 library）未來可以自由重構。
