# my-toolbox

**Live demo**：<https://csliu-toolbox.streamlit.app>

## Scope

This toolbox targets tensor eigenvalue problems (third-order tensors
and higher, m ≥ 3) following the Newton-Noda Iteration framework
introduced in Liu, Guo, and Lin (Numer. Math. 2017) and the
Higher-Order Newton Iteration in Liu (J. Sci. Comput. 2022).

For m = 2 (matrix eigenvalue) problems, please use
`scipy.sparse.linalg.eigs` or `numpy.linalg.eig` — those are
industrial-grade implementations with better convergence guarantees
on the matrix case. For m = 1 (linear scaling), no eigenvalue problem
exists. The `nni` / `honi` / `multi` entry points raise `ValueError`
when called with `m < 3` and the demo's `m` selector is bounded
`[3, 5]`.

## 這是什麼

這個 repo 是一個**練習用 sandbox**，目的是建立「把 MATLAB 演算法 port 成 Python 並驗證數值一致性」的標準工作流程。目前還不是正式的演算法庫，裡面只有一個示範用的 `gaussian_blur` 實作、一套對帳機制，以及一份把踩過的坑寫下來的文件。

流程穩固之後，再用來 port 真正的研究演算法。

---

## 目錄結構

```
my-toolbox/
├── README.md                        你在看的這份
├── .gitignore
├── matlab_exercise/
│   ├── gaussian_blur.m              示範演算法，用 base MATLAB 寫
│   ├── generate_reference.m         固定種子、跑演算法、存 reference.mat
│   └── reference.mat                （git 不追蹤）MATLAB 輸出，對帳基準
└── python/
    ├── .venv/                       （git 不追蹤）numpy + scipy
    ├── gaussian_blur.py             Python port
    ├── test_gaussian_blur.py        單元測試（不依賴 MATLAB）
    └── test_parity.py               與 MATLAB 逐位元對帳
```

---

## 環境設定

Python 端用一個**本地 venv**，不污染系統 Python。第一次用：

```bash
cd ~/Projects/my-toolbox/python
python3 -m venv .venv
.venv/bin/pip install numpy scipy
```

之後執行：

```bash
# 單元測試（不需要 MATLAB）
.venv/bin/python test_gaussian_blur.py

# 數值對帳（需要先產生 reference.mat）
.venv/bin/python test_parity.py
```

在 MATLAB 裡產生 `reference.mat`（每次改動 `gaussian_blur.m` 時重跑）：

```matlab
cd ~/Projects/my-toolbox/matlab_exercise
generate_reference
```

---

## 關鍵設計：`matlab_compat` flag

`python/gaussian_blur.py` 的函數有一個 `matlab_compat=False` 參數。這是個**刻意的設計選擇**：

- **一般使用** (`matlab_compat=False`)：scipy 自然預設（`mode='reflect'`, `truncate=4.0`），是 Python 生態系的慣例，物理上也比較合理。
- **對帳模式** (`matlab_compat=True`)：切到 MATLAB `conv2(h, h, A, 'same')` 的行為（`mode='constant'`, `truncate=3.0`）。只有 `test_parity.py` 會用這個模式。

這樣 Python 版保留 pythonic 的預設介面、不被 port 需求污染，但仍然能驗證跟 MATLAB reference 逐位元一致（誤差 `~1e-16`）。

這個設計是在第一次做 `gaussian_blur` port 時發現的：不加 flag、兩邊直接各自預設跑，max error 是 `3e-1`（邊界差異主導）；對齊邊界降到 `2e-4`（截斷寬度差異）；同時對齊才到 `1.1e-16`（machine epsilon）。

---

## 新增一個 port 的標準流程

### Phase 1 — 規劃
- 確認輸入/輸出的型別、形狀、數值範圍
- 檢查 MATLAB 版有沒有用到 toolbox（見「環境限制」）
- 預先考慮「三大陷阱」會不會踩到

### Phase 2 — MATLAB 端
- 寫 `matlab_exercise/<algo>.m`，只用 base MATLAB
- 寫 `matlab_exercise/generate_<algo>_reference.m`：`rng(42)` 固定輸入、跑演算法、存 `<algo>_reference.mat`
- 手動在 MATLAB 跑一次，確認產出正常

### Phase 3 — Python 端
- 寫 `python/<algo>.py`，簽章**預設用 pythonic 慣例**
- 寫 `python/test_<algo>.py` — 不依賴 MATLAB 的單元測試（對稱性、守恆律、已知解）
- 寫 `python/test_<algo>_parity.py` — 讀 `.mat`、呼叫演算法、比對；門檻 `1e-10`（目標 `~1e-16`）

### Phase 4 — 對帳與修正
跑 parity。如果失敗，照誤差量級診斷：

| 誤差量級 | 主要嫌疑 |
|---|---|
| `~1e-1` | 邊界處理不一致 |
| `~1e-4` | 核/濾波器截斷寬度不一致 |
| 整齊的位移 | 索引起點（1-based vs 0-based） |

修正方式：**不要破壞 Python 預設介面**。加 `matlab_compat` flag，預設走 pythonic、`matlab_compat=True` 才切換到 MATLAB 行為。

### Phase 5 — 記錄
如果這次 port 遇到**新類型**的差異（不是三大陷阱之一），把它補進這份 README 的清單。

---

## 三大 MATLAB→Python 陷阱（每次主動檢查）

1. **邊界處理**
   - MATLAB `conv2('same')` → 零填充
   - MATLAB `imgaussfilt`/`imfilter` 預設 → replicate
   - scipy `gaussian_filter`/`convolve` 預設 → reflect
   - 三種都不一樣，邊界誤差量級通常 `~1e-1`。

2. **濾波器/核截斷寬度**
   - MATLAB `imgaussfilt` 預設 `FilterSize = 2*ceil(2*sigma)+1`
   - 自訂 `conv2` 腳本常用 `ceil(3*sigma)` 半寬
   - scipy 預設 `truncate=4.0`
   - 誤差量級通常 `~1e-4`。

3. **索引起點**
   - MATLAB 1-based、Python 0-based
   - 算術計算索引（`A(i+1, j) - A(i, j)` 之類）時最容易忘記調整

---

## 環境限制

本機 MATLAB **沒有 Image Processing Toolbox**。意思是：

- `imgaussfilt`, `imfilter`, `fspecial`, `imresize`, `imrotate` 等都不能用
- reference 實作必須用 **base MATLAB**：`conv2`, `fft2`, `ifft2`, 基本矩陣運算 OK
- 濾波器核要自己手搓，例如 `h = exp(-x.^2 / (2*sigma^2)); h = h / sum(h);`

如果未來裝了新 toolbox，記得回來更新這段。

---

## Deployment workflow

The Streamlit Cloud demo at <https://csliu-toolbox.streamlit.app> is
auto-rebuilt by a webhook on `git push origin main`. Day 8 saw two
cases where the webhook did not fire and the demo kept serving an
older container, so the workflow below treats every push as
"verify before assuming live".

After pushing to `main`:

1. Wait 2-3 minutes for Streamlit Cloud auto-rebuild.
2. Run `python scripts/check_deploy.py` to verify push state.
3. Open the demo URL in a browser and scroll the sidebar to the
   bottom. Look at the `Build: <sha>` caption.
   - If the shown SHA matches the "Expected build" printed by the
     check script, deploy is current.
   - If the shown SHA differs (older), Cloud has not redeployed.
     Open <https://share.streamlit.io/>, find the `csliu-toolbox`
     app, click **Manage app → Reboot**. Wait ~2 minutes, refresh
     demo, recheck sidebar.
   - If `Build: unknown` is shown, the Cloud container has no
     `.git/` directory; see "Fallback B" below.
4. If reboot does not pick up the latest commit, force a rebuild
   with a trivial commit:

   ```bash
   echo "" >> README.md
   git commit -am "redeploy"
   git push
   ```

### Fallback B — `.git/` not present on Cloud container

If the demo sidebar shows `Build: unknown` after a successful local
read, the Cloud container is not preserving `.git/` after build. In
that case, switch the build-SHA mechanism from file-read to one of:

- **Streamlit secrets** — set `BUILD_SHA = "<short-sha>"` in the app's
  Secrets panel and read via `st.secrets["BUILD_SHA"]`. Manual update
  per push, so this is a stop-gap only.
- **GitHub Actions write-and-commit** — a workflow that updates a
  tracked `python/streamlit_app/_build_info.py` with the head SHA
  and commits before the deploy webhook fires. Adds CI complexity but
  removes the manual step.

Pick the path with the User before implementing.
