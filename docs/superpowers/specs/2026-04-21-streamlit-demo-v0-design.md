# Streamlit Demo v0 — Design Spec

**作者**：CSLiu（使用者）+ Claude（brainstorm partner）
**日期**：2026-04-21
**狀態**：approved 設計完成，待 spec review 後實作
**對應 issue / milestone**：Phase D 中期第一個 UI 里程碑

---

## 1. 願景與 v0 的定位

**長期願景**：把十年累積的 MATLAB 研究程式碼變成**別人能實際使用的工具**。雙模式：
- **本機下載模式**：無限制、完整 API、pip installable
- **網頁即用模式**：有運算 quota、零安裝、HuggingFace Spaces 部署

**目標受眾**：數學界 + 鄰近領域的研究者 / 學生 / 工程師

**v0 在這個大圖裡**：整條「Web UI」線的起手式，但範圍刻意收到最小 —

- ✅ 驗證 Streamlit 在這個 domain 可以跑
- ✅ 對未來 port 的**ergonomics 壓力測試**（函式好不好從 UI 呼叫、API 該長怎樣）
- ✅ 建立**擴充骨架**，之後 port 新演算法是「機械式加一行」
- ❌ 不面向外部使用者（audience = A：只有作者本人）
- ❌ 不做部署、quota、認證
- ❌ 不做 polish（醜沒關係）

---

## 2. Scope

### 2.1 In-scope

- 5 個已 port 完成的 tensor 工具：`tenpow`、`tpv`、`sp_tendiag`、`ten2mat`、`sp_Jaco_Ax`
- 每個函式一個 UI 介面（輸入 widget + 輸出顯示 + MATLAB 原始碼 expander）
- 共用 sidebar 設定（函式選擇、RNG seed）
- 本機啟動文件（`streamlit_app/README.md`）

### 2.2 Out-of-scope (v0)

| 項目 | 理由 / 延後到 |
|---|---|
| 手打 tensor 數字輸入 | 受眾 A 不需要；將來 B/C 再加 |
| 上傳 `.csv` / `.mat` | 同上 |
| dtype 切換（float32/64） | 硬寫 float64；需要時再加 |
| Error handling 包裝 | 讓 Streamlit traceback 露出來 |
| 自動化 UI 測試 | 手動驗證清單即可 |
| `matlab_compat=True/False` 切換 UI | 對 5 個 util 都是 no-op，切了沒差異 |
| 歷史記錄 / 結果下載 | v0 不持久化 |
| HuggingFace Spaces 部署 | 之後另立 sub-project |
| 計算 quota / 認證 | 同上 |
| 1-D 輸出的折線圖 | `tenpow`/`tpv` 的 index 沒空間意義，折線誤導 |
| `st.cache_data` 快取 | 函式毫秒級、不需要；未來 HNI main 再加 |

---

## 3. 檔案架構

```
~/Projects/my-toolbox/                       ← repo root
├── requirements.txt                          核心：numpy, scipy（Python API 最小集）
├── requirements_ui.txt                       UI 額外：streamlit, plotly
└── python/
    ├── tensor_utils.py                       （既有）
    ├── test_tensor_utils.py                  （既有）
    └── streamlit_app/                        ← 新目錄
        ├── __init__.py                       空檔，讓 streamlit_app/ 是 package
        ├── demo_v0.py                        主 Streamlit app
        ├── README.md                         本機啟動 + 未來規劃，繁體中文
        └── matlab_snippets/
            ├── tenpow.m
            ├── tpv.m
            ├── sp_tendiag.m
            ├── ten2mat.m                     （含 idx_create helper）
            └── sp_Jaco_Ax.m
```

### 3.1 命名依據

- **`streamlit_app/` 而不是 `ui/`**：跟 HuggingFace Spaces 官方慣例一致（Spaces 預設找 `streamlit_app/app.py` 或類似路徑），也避免未來若加 Gradio / Dash 時命名衝突
- **`demo_v0.py` 而不是 `app.py`**：明確版號，之後的 v1/v2 可以 side-by-side 存（`demo_v1.py` 等穩了再 deprecate v0）
- **`requirements.txt` + `requirements_ui.txt` 分離**：對「只想用 Python API 不裝 UI」的使用者是公開 toolbox 的基本禮貌；未來遷移 `pyproject.toml` 時對應 `extras_require={'ui': [...]}`

### 3.2 依賴

**`requirements.txt`**（repo root）：
```
numpy
scipy
```

**`requirements_ui.txt`**（repo root）：
```
streamlit
plotly
```

不寫版本釘選（v0 沒必要）。將來打包成 pypi package 時再固化。

---

## 4. Component 設計

### 4.1 `demo_v0.py` 主骨架

```
┌─ imports
│
├─ module-level constants：MATLAB_SNIPPETS_DIR = Path(__file__).parent / "matlab_snippets"
│
├─ Helper 函式
│  ├─ read_snippet(name) -> str           讀對應的 MATLAB 片段
│  ├─ render_output_1d(arr) -> None        1-D 輸出顯示
│  └─ render_output_2d(arr) -> None        2-D 輸出顯示（sparse / dense 自動偵測）
│
├─ 5 個 renderer 函式（無參數，全從 st.sidebar / st.slider 讀狀態）
│  ├─ render_tenpow()
│  ├─ render_tpv()
│  ├─ render_sp_tendiag()
│  ├─ render_ten2mat()
│  └─ render_sp_Jaco_Ax()
│
├─ RENDERERS = {"tenpow": render_tenpow, ...}    ← 擴充點
│
└─ if __name__ == "__main__" 或直接 top-level：
   sidebar 畫出 selectbox + rng seed
   呼叫 RENDERERS[chosen]()
```

### 4.2 Sidebar 內容

```
# 📊 Tensor Utils Demo v0

Function
  ▼ tenpow
    tpv
    sp_tendiag
    ten2mat
    sp_Jaco_Ax

─── Shared settings ───
RNG seed: [42]   (number_input)
```

- `st.sidebar.selectbox("Function", list(RENDERERS.keys()))`
- `st.sidebar.number_input("RNG seed", value=42, min_value=0, step=1)`

### 4.3 每個 renderer 的 I/O 規格

| Renderer | 輸入 widgets | 生成的資料 | 輸出 shape | 輸出 helper |
|---|---|---|---|---|
| `render_tenpow` | `n` slider [1, 10]=5、`p` slider [0, 4]=3 | `x = rng.random(n)` | `(n**p,)`（`p=0` 時 `(1,)`） | `render_output_1d` |
| `render_tpv` | `n` slider [2, 6]=3、`m` slider [2, 4]=3 | `AA = rng.random((n, n**(m-1)))`、`x = rng.random(n)` | `(n,)` | `render_output_1d`，另外顯示 `AA.shape` |
| `render_sp_tendiag` | `n` slider [2, 5]=3、`m` slider [2, 5]=3 | `d = rng.random(n)` | sparse `(n, n**(m-1))` | `render_output_2d` |
| `render_ten2mat` | `n` slider [2, 5]=3、`m` slider [2, 5]=3、`k` slider [0, m-1]=0 | `A = rng.random((n,)*m)` | `(n, n**(m-1))` | `render_output_2d` |
| `render_sp_Jaco_Ax` | `n` slider [2, 4]=3、`m` slider [2, 4]=3 | `AA = rng.random((n, n**(m-1)))`、`x = rng.random(n)` | sparse `(n, n)` | `render_output_2d` |

**為什麼 `sp_Jaco_Ax` 上限收斂到 n=4, m=4**：中間 kron 到 `(n^(m-1) × n^(m-1), n)` 量級，n=5, m=5 變成 `(390625, 5)`，Streamlit 每次拉 slider 重跑會明顯卡；n=4, m=4 是 `(4096, 4)` 舒服。不做 `st.cache_data`，用 slider 上限守住最省事。

### 4.4 每個 renderer 的結尾：MATLAB expander

```python
with st.expander("📄 對應的 MATLAB 原始碼"):
    st.code(read_snippet("tenpow"), language="matlab")
```

`read_snippet(name)` 從 `matlab_snippets/<name>.m` 讀純文字、整份丟進 `st.code`。Streamlit 對 `language="matlab"` 有內建 Pygments syntax highlighting。

### 4.5 Helper：`render_output_1d(arr)`

顯示三層：
1. `st.write` 一行 shape + dtype + length
2. 用 `st.dataframe(arr[:10])` 顯示前 10 個元素；若 `len(arr) <= 10` 就全顯示
3. `st.caption` 摘要：max / min / mean / std

**不加折線圖**（理由見 Section 2.2）。

### 4.6 Helper：`render_output_2d(arr)`

1. `st.write` shape + dtype + `nnz`（若 sparse）
2. 偵測 `scipy.sparse.issparse(arr)` → `.toarray()` 後畫
3. **Plotly heatmap**：`px.imshow(arr_dense, aspect='auto', color_continuous_scale='Viridis')`
4. `st.plotly_chart(fig, use_container_width=True)`
5. `st.caption` 摘要：max / min / mean

---

## 5. 資料流

### 5.1 使用者互動流程

```
$ cd python && .venv/bin/streamlit run streamlit_app/demo_v0.py
   ↓
Browser 開 http://localhost:8501
   ↓
預設顯示第一個函式（tenpow，selectbox 第一項）
   ↓
拉 slider → Streamlit auto-rerun（全 script）→ 新輸出 → 即時顯示
   ↓
切換 selectbox → dispatch 到另一個 renderer
   ↓
展開 expander → 看 MATLAB 原始碼
```

### 5.2 Per-renderer 資料流

```
widgets 讀取 (n, m, ...)
    │
    ▼
rng = np.random.default_rng(seed)    ← 從 sidebar 讀 seed
    │
    ▼
input 張量生成 (rng.random(...))
    │
    ▼
output = tensor_utils.func(input, ...)
    │
    ▼
render_output_1d 或 render_output_2d
    │
    ▼
st.expander 含 MATLAB snippet
```

### 5.3 Auto-rerun 選擇

- **採用 Streamlit 預設 auto-rerun**（每次 widget 變動觸發全 script 重跑）
- **不加 Run 按鈕**
- 理由：函式毫秒級；即時反饋是 demo 核心價值
- `sp_Jaco_Ax` 在 n=4, m=4 邊界的延遲**刻意保留** — 那是演算法「大尺寸會慢」的真實訊號，不是 UI bug

---

## 6. MATLAB snippet 管理

### 6.1 抽取來源

| Snippet 檔名 | 來源檔 | 行範圍 |
|---|---|---|
| `tenpow.m` | `matlab_ref/hni/HONI.m` | 129-139 |
| `tpv.m` | `matlab_ref/hni/HONI.m` | 123-127 |
| `sp_tendiag.m` | `matlab_ref/hni/HONI.m` | 142-149 |
| `ten2mat.m` | `matlab_ref/hni/HONI.m` | 152-177（含 `idx_create` helper） |
| `sp_Jaco_Ax.m` | `matlab_ref/hni/Multi.m` | 79-87 |

### 6.2 抽取規則

- **逐字複製**，不清理
- **保留 MATLAB 原樣 comment、拼字、空白**（即使原碼有 typo 也不修 — 例如 `Multi.m:47` 有 `'suitible'` typo，v0 的 5 個 snippet 剛好都不含這行，但原則照樣：未來抽 `Multi` 函式本體時要留著 typo）
- 每個 snippet 檔**最頂端加一行註解**標明來源：

```matlab
% Verbatim from matlab_ref/hni/HONI.m line 129-139.
% Do not edit — this is kept in sync with the canonical source.
```

### 6.3 不自動抽取的理由

**手動一次性抽取、不寫自動抽取器**。理由：
- 5 個 snippet 是一次性工作
- 未來若改動 `matlab_ref/hni/` 的 canonical 檔（不太會發生，source 是 frozen），手動同步負擔極小
- 自動抽取器要處理「找 function header → 找 `end`」的 MATLAB 語法，投資不划算

---

## 7. 錯誤處理

**v0 策略：最小化** — 讓 Streamlit 原生 traceback 露出來。不加 try/except 包裝。

### 7.1 唯一主動處理：動態 slider 上限

`ten2mat` 的 `k` 參數範圍 `[0, m-1]` 依賴 `m`。當 `m` 改小時（例如 `m=5, k=4` → `m=3`），`k` 的 max_value 會變。

**作法**：直接用動態 `max_value` 參數：
```python
m = st.slider("m (tensor order)", 2, 5, 3)
k = st.slider("k (mode)", 0, m - 1, 0)
```

Streamlit 對「max_value 下修、現有 value 超出」的預設行為是自動夾回去、不 crash。v0 接受這個預設行為，不另外管 session_state。

---

## 8. 測試

**v0 不寫自動化測試。**

理由：
- 底層函式有 `test_tensor_utils.py`（sanity）+ `test_*_parity.py`（parity）完整覆蓋
- UI 層本質是「組合已驗證工具 + Streamlit 渲染」，沒有新邏輯需要驗證
- UI 自動化測試（Selenium / Playwright）成本遠超 v0 價值

### 8.1 Commit 前的手動驗證清單

1. `cd python && .venv/bin/streamlit run streamlit_app/demo_v0.py` 能起得來，browser 有正常畫面
2. 切過 5 個 renderer 各一次，每個拉 slider 2-3 下
3. 每次切 renderer 都不該有 exception
4. 輸出 shape / heatmap 目視合理（與 sanity test 的輸出對得上）
5. 每個 renderer 的 MATLAB snippet expander 展開後內容正確（跟 `matlab_ref/hni/*.m` 對一下）
6. 改 sidebar `RNG seed` 為其他值（例如 100）→ 重跑 → 確認輸入確實變了

---

## 9. 擴充 Contract（給未來版本的承諾）

**唯一擴充點**：`RENDERERS` dispatch dict。

```python
RENDERERS = {
    "tenpow":     render_tenpow,
    "tpv":        render_tpv,
    "sp_tendiag": render_sp_tendiag,
    "ten2mat":    render_ten2mat,
    "sp_Jaco_Ax": render_sp_Jaco_Ax,
}
```

未來任何新功能 port 完成（例如 `Multi`、`HONI`、NNI）加進 demo 的流程：

1. 寫 `render_<func>()` renderer（遵守 4.5 / 4.6 的 helper 介面）
2. 若是 MATLAB 有原檔的函式，加 `matlab_snippets/<func>.m`
3. 在 `RENDERERS` dict 加一行
4. **不改既有任何程式碼**

這是 v0 唯一必須守住的向後相容承諾。其他細節（sidebar 佈局、helper 實作、plotly 換別的 library）都可以未來自由改。

---

## 10. 交付清單

### 10.1 新建檔（10 個）

1. `requirements.txt`（repo root）
2. `requirements_ui.txt`（repo root）
3. `python/streamlit_app/__init__.py`
4. `python/streamlit_app/demo_v0.py`
5. `python/streamlit_app/README.md`
6. `python/streamlit_app/matlab_snippets/tenpow.m`
7. `python/streamlit_app/matlab_snippets/tpv.m`
8. `python/streamlit_app/matlab_snippets/sp_tendiag.m`
9. `python/streamlit_app/matlab_snippets/ten2mat.m`
10. `python/streamlit_app/matlab_snippets/sp_Jaco_Ax.m`

### 10.2 修改檔

無（`.gitignore` 已涵蓋 `.venv/` / `*.mat`）。

### 10.3 環境變更

在 `python/.venv/` 裝 `streamlit` 和 `plotly`（不動系統 Python）。

### 10.4 `README.md` 內容

繁體中文，含：
- **本檔的定位**：Phase D 中期 minimum viable demo，audience = A（作者內部驗證用）
- **本機啟動步驟**：`pip install -r requirements.txt -r requirements_ui.txt` → `streamlit run streamlit_app/demo_v0.py`
- **功能清單**：5 個函式、Auto-rerun、Plotly heatmap、MATLAB snippet expander
- **未來擴充計劃**：
  - 更多演算法 port 完後加進 `RENDERERS`
  - v1/v2 考慮受眾 B/C、加範例資料、錯誤訊息翻譯
  - 最終部署到 HuggingFace Spaces，含 compute quota
- **明確聲明「v0 不做部署」**

---

## 11. 決策紀錄

| 決策 | 選擇 | 替代方案 | 理由 |
|---|---|---|---|
| 目錄命名 | `streamlit_app/` | `ui/` | 跟 HF Spaces 慣例一致、避免未來 Gradio/Dash 命名衝突 |
| requirements 結構 | 雙層 .txt（核心 / UI） | 單一 .txt / 直接 pyproject.toml | 對外部使用者友善；pyproject.toml 之後再做 |
| 視覺化 | plotly | matplotlib / seaborn | 互動式、~30MB 比 matplotlib 輕、HF Spaces 原生支援 |
| 版面 | sidebar selectbox | tabs / dashboard | 擴充最友善；未來加到 20+ 函式也清爽 |
| 觸發 | auto-rerun | Run 按鈕 | 函式毫秒級，即時反饋是 demo 核心 |
| 1-D 輸出視覺化 | 不 plot、只 dataframe | line chart | 輸出 index 無空間意義 |
| 2-D 輸出視覺化 | Plotly heatmap | 純數字 / static image | 互動式 hover 看值、位置正確性一眼就清楚 |
| MATLAB snippet 存放 | 獨立 `matlab_snippets/` 目錄 | hardcoded string in demo_v0.py | 獨立檔案可 review、demo code 乾淨 |
| 錯誤處理 | 無 try/except | 全 try/except | Internal tool 看到錯就 debug |
| 測試 | 無自動化 + 手動 checklist | Playwright/Selenium | UI 自動化對 v0 不划算；底層已 parity-tested |
| 跳過 writing-plans skill | yes | 正規 skill flow | v0 規模小，spec 已足夠詳細 |

---

## 12. 開放項目

**無**。所有決策已定案。

（若實作時浮現未預期的決策需求，停下來問使用者再寫。）
