# PROGRESS.md — Session checkpoint

**最後更新**：2026-04-28（Day 7 完整收工、P5 deploy 成功）
**狀態**：**Session 7 方向變更為 D（Streamlit demo 美化 + 部署）、P1-P5 全部完成**。Live demo 上線：<https://csliu-toolbox.streamlit.app>。GitHub repo `chingsungliu/tensor-eigenvalue-toolbox`（public）上 commit `b829b1c` build 成功部署。本機 streamlit process（PID 1088）已殺、cloud demo 為主。前置（Day 1-6）：4 個 Layer 3 演算法 port 全綠、memory 11 條、fragility 三類別。Day 8 起進 Level 2（收斂速度優化）Phase 1：Baseline & Profiling。

---

## Day 1 (2026-04-21) 完成摘要

### 一、HNI Layer 1 + 2 完成（5 個 tensor 工具）

Phase D 的前兩層全部 port 完、parity 通過。

| 函式 | 位置 | Max error | 關鍵測試 |
|---|---|---|---|
| `tenpow` | `python/tensor_utils.py::tenpow` | **0**（bit-identical） | 純 Kronecker 次方 |
| `tpv` | `python/tensor_utils.py::tpv` | ~1e-16（machine epsilon） | 矩陣-向量乘 |
| `sp_tendiag` | `python/tensor_utils.py::sp_tendiag` | **0** | 對角張量 mode-1 unfolding |
| `ten2mat` | `python/tensor_utils.py::ten2mat` | **0** | ⭐ column-major 主檢查點（`np.moveaxis` + `reshape(order='F')`） |
| `sp_Jaco_Ax` | `python/tensor_utils.py::sp_Jaco_Ax` | ~1e-15 | Jacobian via sparse kron |

**HNI 原始碼 canonical 版本**：`matlab_ref/hni/` 裡的 `HONI.m`、`Multi.m`、`main_Heig.m`（來自 `source_code/Tensor Eigenvalue Problem/2020_HNI_submitt/`，2020/12 撰寫、2021/06 定稿）。

詳細 port 進度表：`matlab_ref/hni/README.md` 的 Section「Port 進度」。

### 二、source_code/ 全局盤點（490 MB、1304 `.m` 檔）

- 5 個頂層類別：Tensor Eigenvalue（83%）、Optimization/QP（ISM）、Nonlinear Schrödinger/BEC、M-matrix Iteration、Generalized Eigenvalue (GINI)
- **549 次**重複定義已 port 的 5 個函式 — 未來 port 其他 tensor eigenvalue 演算法時全部可重用
- 30+ 組疑似重複資料夾（版本保存、副本）
- 詳細報告：`matlab_ref/GLOBAL_INVENTORY.md`（391 行）
- `source_code/` 已加入 `.gitignore`，不入 git

### 三、Streamlit demo v0（6 個函式）

- 瀏覽器內互動介面：`python/streamlit_app/demo_v0.py`
- 包含：Layer 1+2 的 5 個函式 + `gaussian_blur`（今早的第一個 port）
- 每個 renderer 有 MATLAB 原碼 expander（逐字從 `matlab_ref/hni/*.m` 和 `matlab_exercise/gaussian_blur.m` 抽取）
- **§9 擴充 contract 實測成立** — 加 gaussian_blur 只動 4 處（2 import、1 helper、1 renderer、1 dict 行）
- 設計 spec：`docs/superpowers/specs/2026-04-21-streamlit-demo-v0-design.md`

**本機啟動**：
```bash
cd ~/Projects/my-toolbox/python
.venv/bin/streamlit run streamlit_app/demo_v0.py
```

### 四、Per-iteration parity POC（D3a 前置）

為 Layer 3 的迭代演算法 port（Multi / HONI / NNI）建立對帳框架。

- Toy example：Newton's method 求 sqrt(2)、10 步迭代
- 核心設計：`find_divergence` / `report` / `print_neighborhood` 三件組
- **關鍵**：不只報 `max_err`，要報「第一個超過 tolerance 的 iteration」(`first_bad_iter`)
- 位置：`python/poc_iteration/`、`matlab_ref/poc_iteration/`
- POC 結果：`x_history` 和 `res_history` 全 11 步 bit-identical（max_err = 0）
- POC README（繁體中文）含延伸到 Multi / HONI 的樣板說明

### 五、Memory 寫入（Day 1 結束時 7 條）

| Memory 檔 | type | 說明 |
|---|---|---|
| `user_role.md` | user | 使用者背景 / 偏好 |
| `project_layout.md` | project | 專案目錄結構 |
| `feedback_matlab_to_python_port.md` | feedback | **四大陷阱** checklist（boundary / truncation / 1-vs-0 indexing / column-major） |
| `project_matlab_environment.md` | project | 無 Image Processing Toolbox、用 base MATLAB |
| `project_big5_encoding.md` | project | Big5 編碼的舊 MATLAB 程式處理法 |
| `feedback_multi_version_research_code.md` | feedback | 多版本研究程式碼要列所有版本、讓使用者決定 canonical |
| `feedback_per_iteration_parity.md` | feedback | Day 1 新增：迭代演算法的 parity 機制 |

---

## Day 2 (2026-04-22) 完成摘要

### 一、HNI Layer 3 step 1/2（Multi port）完成

Multi.m 的 Newton + 三等分 halving line search 完整 port 到 Python，Q5 case parity 通過到 machine epsilon。

| 欄位 | max_err | 語意 |
|---|---|---|
| `hal_history` | **0.000e+00** | bit-identical |
| `theta_history[1:]` | **0.000e+00** | bit-identical |
| `final u` | 1.110e-16 | 0.5× machine epsilon |
| `u_history` (2D) | 3.331e-16 | 1.5× machine epsilon |
| `res_history` | 4.441e-16 | 2× machine epsilon |
| `v_history[:, 1:]` (2D) | 6.661e-16 | 3× machine epsilon |

Port 檔案：
- `python/tensor_utils.py::multi` — 加 `multi(AA, b, m, tol, record_history=False)` + `sparse_norm` / `spsolve` import
- `python/test_tensor_utils.py::test_multi_basic` — Python-only sanity（nit=5、residual 6e-12、invariant check）
- `python/test_multi_parity.py` — 讀 .mat、5 欄逐 iter parity、final u 比對
- `matlab_ref/hni/Multi_with_history.m` — Multi.m 逐字複本 + 5 欄 history output
- `matlab_ref/hni/generate_multi_reference.m` — Q5 test case 驅動

Port 前的 hazard analysis：`docs/superpowers/notes/multi_hazard_analysis.md`（232 行、Day 2 階段 A 產出）。五個 open questions 在 Session 2 初都由使用者決定（pre-alloc 100、0-based indexing、.tocsc()、b 不 mutate、Q5 穩定 test case）。

### 二、Per-iteration parity 三件組抽成 module

`poc_iteration/parity_utils.py` 新檔 — 把 `find_divergence / report / print_neighborhood` 從 POC 測試檔抽出，未來 Multi / HONI / NNI parity test 可直接 `from poc_iteration import ...`。POC 測試檔改用 parity_utils、仍 bit-identical 通過。

### 三、Halving path parity 延後（技術決定、非偷懶）

Q5b 試圖以 ill-conditioned AA 觸發 halving 連跑兩輪參數（scale 0.01 → 0.30 → 0.10）都 trap-and-diverge。**根因**：m=3 Multi 在 random perturbation AA 下沒有「健康 halving」sweet spot — u 一翻負、`temp = AA·u²` 出負分量、halving 拉回 u_old 但 u_old 也已負、越踩越深。

這不是 port bug、是 Multi 原演算法的 fragility：halving 設計上是「near-optimal 附近微調」、不是「大幅 overshoot 修正」。HONI 會呼叫 Multi 時 u 已接近最優，halving 自然觸發、parity 在 HONI integration test 才是真實 workload。

Fragility 寫進 `memory/feedback_multi_halving_fragility.md`（Day 2 新增第 8 條 memory）。

### 四、Memory 寫入（Day 2 結束時 8 條）

比 Day 1 多一條：
| Memory 檔 | type | 說明 |
|---|---|---|
| `feedback_multi_halving_fragility.md` | feedback | **Day 2 新增**：Multi 的 halving 在 m≥3 + random AA 沒 healthy sweet spot、延後到 HONI integration test；MATLAB 原碼 `theta=3^(-hal)` 自我驗證到 8.7e-19（未來 Python 對不上的話是 Python 端 bug） |

---

## Day 2 末（2026-04-22 evening）+ Day 3 (2026-04-22) 完成摘要

### 一、HNI Layer 3 step 2/2（HONI port）完成

HONI.m（232 行、外層 eigenvalue iteration + 內層呼叫 Multi）完整 port 到 Python，**exact 與 inexact 兩個分支**都通過三層 tier parity（PASS condition：Tier 1 + Tier 2 全過）。

| Tier | 度量類型 | 容差 | 對象欄位 |
|---|---|---|---|
| **1 STRICT** | abs | `1e-10` | final λ / final x、`x_history`、`lambda_history`、`res`、`inner_tol_history`、`chit_history`、`hal_per_outer_history`（slots 0..K-2） |
| **1 STRICT** | **rel** | `1e-8` | **`y_history`**（slots 1..K-2、量級隨 `lambda_U → eigenvalue` 指數放大、必須用相對誤差） |
| **2 APPROX** | rel | `1e-2` | `y_history[:, K-1]`（last-iter shift-invert near-singular fragility 爆點） |
| **3 INFO** | — | no-assert | last-iter `chit/hal_per_outer/hal_accum`、`nit/innit/hal` scalars（fragility 傳播後的計數差） |

**兩分支實測 fragility**（HONI Q5 case，m=3、n=10、rng(42) initial vector）：

| 分支 | y_history strict 段 max rel | y_history Tier 2 last-iter rel | 最終 λ / x |
|---|---|---|---|
| `exact` | ~5e-12 (machine eps) | **2.1e-5** | bit-identical（abs ≤ 1e-10） |
| `inexact` | up to ~5e-9 at iter 4 | **1.1e-3** | bit-identical（abs ≤ 1e-10） |

**Fragility 根因**：`lambda_U` 收斂到真特徵值時，`lambda_U * II - AA` 變 near-singular、內層 Multi 解出的 `y` 量級從 O(1) → O(10^6)、scipy `spsolve` (SuperLU) vs MATLAB `mldivide` (LU) 的 pivot 差在這個 ill-conditioned 區段被放大。**inexact 比 exact ~50× 敏感**，因為 `lambda_U = max(temp)` 整個重算 vs `lambda_U -= min(temp)^(m-1)` 增量更新；增量自帶 damping、重算把 y 的相對誤差直接餵下一輪。

詳細分析見 `memory/feedback_honi_multi_fragility_propagation.md`（Day 3 新增、第 9 條 memory）。

**Port 檔案**：
- `python/tensor_utils.py::honi` — exact + inexact 兩分支、polymorphic A（2-D unfolding 或 m-D tensor）、`record_history` flag、9 個外層 history 欄位
- `python/test_tensor_utils.py::test_honi_basic` 等 — Python-only sanity（不依賴 MATLAB）
- `python/test_honi_parity.py` — 三 tier parity 框架、兩 case（exact + inexact）各跑一次
- `matlab_ref/hni/HONI_with_history.m` — HONI.m 逐字複本 + 9 欄 history output
- `matlab_ref/hni/generate_honi_reference.m` — 兩 case 驅動，輸出 `honi_reference_exact.mat` + `honi_reference_inexact.mat`

Port 前的 hazard analysis：`docs/superpowers/notes/honi_hazard_analysis.md`（Day 2 末產出、352 行、含 8 個 open questions 全部由使用者決定）。

### 二、Memory 寫入（Day 3 結束時 9 條）

比 Day 2 多一條：
| Memory 檔 | type | 說明 |
|---|---|---|
| `feedback_honi_multi_fragility_propagation.md` | feedback | **Day 3 新增**：HONI+Multi 在 near-singular shift-invert 的 fragility 傳播；y-like 欄位 parity 必須 rtol；inexact 分支 ~50× 比 exact 敏感；最終 λ/x 仍 bit-identical；三 Tier 框架可重用於任何 shift-invert 類迭代演算法 |

### 三、HNI 系統里程碑文件

新增 `docs/hni_status.md`（繁體中文）— Layer 1/2/3 完整狀態、parity 結果表（含 Tier 機制）、已知 fragility 摘要（halving + shift-invert）、下一步 scope。供使用者隨時 1 分鐘對齊 HNI 進度，**不重複 PROGRESS.md 的 session 紀錄**、focus 在「系統現況」。

---

## Git state（Day 3 收工）

Day 3 新增（topmost、3 筆）：

```
<pending3> Add HNI system-level status document after Layer 3 complete   ← 本批最後 commit
<pending2> Update CLAUDE.md context loader for Day 3
<pending1> Update PROGRESS.md after Day 2 session (Layer 3 complete)
```

Day 2 末（HONI 收工、未及時更新 PROGRESS）：

```
ccbe5c4 Port HONI (exact + inexact) to Python with tiered parity validation (Layer 3 step 2/2)
```

Day 2 中段（Multi port、含當時的 PROGRESS update）：

```
d5c6885 Update PROGRESS.md and CLAUDE.md after Multi port completion
3269ab7 Simplify multi_reference to Q5-only; document halving fragility
21ca097 Port Multi to Python with per-iteration parity validation (Layer 3 step 1/2)
8288758 Add CLAUDE.md as project-level context loader for new sessions
ef38f9c Add Multi.m hazard analysis before port (階段 A)
```

Day 1 結束時 15 筆（最底下的 `d7af773` 是初始 commit）：

```
1908137 Add PROGRESS.md session checkpoint after Day 1
9ad878f Add gaussian_blur to demo_v0 (6 functions total)
28d8aad Add per-iteration parity POC (Newton sqrt(2)) for Layer 3 preparation
05a04ef Implement Streamlit demo v0 per design spec
c12252d Add design spec for Streamlit demo v0
3cb607a Add global inventory of source_code/ (1304 .m files, 113 folders)
c6cc76c Exclude source_code/ from tracking (490MB MATLAB archive)
30153cc Port sp_Jaco_Ax to Python with parity validation (layer 2 complete)
53a2fec Port ten2mat to Python with parity validation (critical column-major checkpoint)
b552941 Port sp_tendiag to Python with parity validation
b98ea0d Port tpv to Python with parity validation
45c3fe9 Port tenpow to Python with parity validation
5b0d54a Import HNI canonical source from 2020_HNI_submitt and initial hazard analysis
39ec462 Append Section H: NNI version comparison and canonical candidates
803d26f Initial inventory of NNI and HNI MATLAB source folders
d7af773 Initial toolbox: gaussian_blur port with parity validation
```

---


## Day 4 (2026-04-22) 完成摘要 — NNI port（Session 4 階段 A + B）

### 一、Layer 3+ step 3/3（NNI canonical port）完成

NNI.m（`source_code/Tensor Eigenvalue Problem/2020_HNI_Revised/NNI.m`、271 行、canonical 無 halving 版）port 到 Python。**NNI 是使用者主演算法**（HNI 是另一個角度、Multi 是共用基礎）。

**Scope 擴展**：`linear_solver in {'spsolve', 'gmres'}`
- `spsolve`：對應 MATLAB backslash、parity target
- `gmres`：Python-only 大 sparse 支援、sanity test 驗兩 solver λ 差 < 1e-6

**Parity 結果**（Q7 test case = Multi Q5 / HONI Q4 同款 rng(42) 參數）：

| Tier | 度量 | 結果 |
|---|---|---|
| **Tier 1 STRICT**（overlap [0:22]）| x/w/y abs + y rel | max_err ≤ 1e-11；x/w/y machine eps |
| **Tier 2 APPROX**（final outputs）| λ_U abs, x rel | 3.5e-13, 4.9e-17 |
| **Tier 3 INFO**（no-assert）| MATLAB 多 5 iter | iter 22-26 於 noise floor 震盪 |

**Pattern (C) 根因 — Rayleigh quotient noise floor**（**非** sparse vs dense LU pivot 差異）：
- `temp = tpv(AA, x, m) / (x^(m-1))` 的 element-wise 除法在 `min(x_i) = 4.8e-6` 時把 tpv 的 1e-16 誤差放大到 ~1e-12
- 跟 tol=1e-12 同量級、res 停止條件浮點抽籤
- 公式：`eigenvalue_noise_floor ≈ machine_eps / min(x_i)^(m-1)`
- 詳見 `memory/feedback_nni_rayleigh_quotient_noise_floor.md`（Day 4 新增、第 10 條 memory）

**Port 檔案**：
- `python/tensor_utils.py::nni` — spsolve + gmres 雙分支、polymorphic A、8 欄 history + gmres_info_history、§五 NNI 專屬 hazard checklist 4 條
- `python/test_tensor_utils.py::test_nni_basic` — 10 sub-check Python-only sanity（含 gmres 分支覆蓋 3 處）
- `python/test_nni_parity.py` — 3-Tier parity 框架（overlap STRICT + final APPROX + extra-iter INFO）
- `matlab_ref/nni/NNI_with_history.m` — canonical NNI.m GE 路徑 + step_length no-halving + 7 欄 history
- `matlab_ref/nni/generate_nni_reference.m` — Q7 driver，rng(42) 完全複製 Multi Q5 參數

Port 前的 hazard analysis：`docs/superpowers/notes/nni_hazard_analysis.md`（Session 4 階段 A、653 行、12 open questions 全部由使用者決定、commit `ab454ea`）。

### 二、Memory 寫入（Day 4 結束時 10 條）

比 Day 3 多一條：
| Memory 檔 | type | 說明 |
|---|---|---|
| `feedback_nni_rayleigh_quotient_noise_floor.md` | feedback | **Day 4 新增**：NNI 的 Rayleigh quotient noise floor 公式 `machine_eps/min(x_i)^(m-1)`；parity tier 設計原則；與 halving（Multi）/ shift-invert（HONI）並列**第三種 fragility 模式** |

### 三、三種 fragility 模式收斂

三條 memory 涵蓋 port 的三種 fragility 源頭：

| Fragility | 觸發者 | 位置 | 參考 memory |
|---|---|---|---|
| halving sweet spot | Multi | line search 設計邊界（m≥3 random AA） | Day 2 |
| shift-invert global | HONI | lambda_U → eigenvalue 整個 M 近奇異 | Day 3 |
| Rayleigh quotient local | NNI | min(x_i) → 0 的 element-wise 除法放大 | Day 4 |

---

## Day 5 (2026-04-23) 完成摘要 — Layer 4 demo 整合 + 架構重構

### 一、Layer 4 demo 整合（CP1–CP5）

三個 Layer 3 演算法 + 對比 tile 全部進 Streamlit demo，demo v0 從 6 函式擴到 10 個 renderer：

| Renderer | 輸入 | 輸出 metrics | Plots |
|---|---|---|---|
| `render_multi` | m, n, tol（Q7 AA）| nit / final res / total halving | residual log + final u bar |
| `render_honi` | m, n, tol, maxit, linear_solver ∈ {exact, inexact} | outer nit / inner nit / λ / res | residual log + λ_U linear + final x bar |
| `render_nni` | m, n, tol, maxit, linear_solver ∈ {spsolve, gmres} | nit / λ_U / λ_L / spread / res | residual log + bracket linear 雙線 + final x bar + gmres warnings caption |
| `render_hni_vs_nni` | 共用 inputs + 兩個 solver radio | 3-tab（HONI / NNI / Comparison）；cross-validate \|Δλ\| + sign-aligned ‖x_H − x_N‖ | 並排 residual log 雙曲線 |

Q7 test case 驗證：HONI exact 與 NNI spsolve 最終 **λ = 10.75623426**（bit-identical 到 machine epsilon、\|Δλ\| = 1.8e-15、‖x\_H − x\_N‖ = 2.7e-16 sign-aligned）— 兩獨立演算法 cross-validate 的事實從 parity test 裡提到 demo 首頁級敘事。

### 二、Streamlit 架構重構（CP6）— problem-driven 兩層選單

原本 10 entries 扁平 `RENDERERS` dict → 重組為 `PROBLEM_ALGORITHMS` 兩層 dict：

```
python/streamlit_app/
├── demo_v0.py (79 行 router、比原 961 行削減 92%)
├── _internal/
│   ├── snippets.py                   # read_snippet (shared helper)
│   └── utility_renderers.py          # 6 Layer 1/2 renderers，從 UI 隱藏
└── problems/tensor_eigenvalue/
    ├── defaults.py                   # build_q7_tensor
    └── algorithms.py                 # 4 Layer 3 renderers + ALGORITHMS dict
```

Sidebar 變成 Step 1（問題類別）→ Step 2（演算法）。NLS / NCP 用 🚧 coming-soon placeholder 佔位、點到顯示 st.info 不 crash。Utilities 保留在 `_internal/` 可 import、但不暴露於 UI（demo 焦點收斂到「使用者研究演算法」）。

Helper 歸屬經 grep 實地 verify、零重複：
- `read_snippet` 共用 → `_internal/snippets.py`
- `_get_rng` / `render_output_1d/2d` / `_make_sample_image` 只 Layer 1/2 用 → `_internal/`
- `_plot_log_history` / `_plot_bar_vector` / `_gmres_info_summary` 只 Layer 3 用 → `algorithms.py`
- `build_q7_tensor` 只 Layer 3 用 → `defaults.py`（underscore 去掉、現在是 sub-package public API）

### 三、研究筆記英文版 + KaTeX 清理

- 新增 `docs/papers/rayleigh_quotient_noise_floor_en.md`（English draft）— NNI 演算法架構（§2）+ noise floor 觀察/量化（§3-4）+ tolerance 選擇實務建議（§5）
- 公式 bound: $\lambda_U - \lambda_L \lesssim n^{m-1} \lambda_{\max} \varepsilon_{\text{machine}} / (\min_i x_i)^{m-1}$
- 初版有 KaTeX render + LaTeX source 交錯亂碼；清理後純 LaTeX source（512 → 494 行），可在 GitHub markdown / VS Code preview 正確顯示
- LaTeX → PDF 轉換延後

### 四、Streamlit demo 發布狀態驗證

- 非互動：module import + 4 Layer 3 algorithms + Q7 data paths 全綠
- Headless boot（CP5 Part A / Part B / CP6 共三次）：health 200 / home 200 / 0 errors in log
- 互動測試（切選單、點 Run、切 tab）需使用者自己跑

---

## Day 6 (2026-04-24) 完成摘要 — Session 6 雙題：CP7 UI 擴展 + NNI_ha port

Session 6 原本三選一（CP7 UI 擴展 / NNI_ha port / LaTeX 轉換）、本日做完前兩個、LaTeX 延後 Session 7。

### 一、CP7 — Streamlit demo UI 擴展（選項 A）

三個 sub-checkpoint、ALGORITHMS dict 從 4 → 6 entries、upload 和 multi-run compare **正交可組合**（上傳的 AA 可餵任一 comparison mode）。

**CP7a — 檔案上傳** (commit `bd0d919`)：
- `problems/tensor_eigenvalue/uploads.py` 189 行、reserved key (`AA` / `A_tensor` / `x0`) 優先 + auto-detect fallback、同時支援 2-D mode-1 unfolding 和 m-D full tensor（`ten2mat(k=0)` auto-unfolding）、`.mat` / `.npz` / scipy.sparse.save_npz 三格式
- `algorithms.py` 4 個 renderer 各加 `Data source` radio + file_uploader，m/n upload 時 disabled & prefilled
- 8 unit tests 全綠（含 bad x0 shape / no-tensor / bad extension 的 error path）

**CP7b-1 — Eigenvalue solver 多-run 對比** (commit `a96dbdd`)：
- `render_eigenvalue_compare`：2 或 3 runs、每 run 獨立選 HONI/NNI + tol/maxit/solver、共用 AA + initial_vector
- 垂直堆疊 layout（break col_in/col_out pattern、適合 comparison 天然橫向並排）
- 不用 `st.form`（algorithm radio 切換要 immediate rerun 讓 solver options 改）
- Smoke test: 3-run mixed (HONI exact + NNI spsolve + NNI gmres@1e-12) 全收到 λ=10.75623426、max |Δλ|=1.8e-15

**CP7b-2 — Multilinear solver tol-sweep** (commit `a93d6af`)：
- `render_multilinear_compare`：2 或 3 Multi runs、只 vary tol（Multi 無 maxit/solver kwarg）
- 預設 tol: Run 1=1e-8, 2=1e-10, 3=1e-12（讓 fresh user 看到有意義 sweep）
- Q7 Multi well-conditioned、三 tol 於 scaled 停止條件下都 nit=5 收到同結果（這是 Multi 本質、非 demo bug）

### 二、NNI_ha port（Session 6 選項 B、Phase 0-3）

Session 4 port 的 canonical `NNI.m` 是 halving 註解版；`Test_Heig2.m` paper benchmark 呼叫的 `NNI_ha.m` 是 halving active 版。本日 port NNI_ha、補上重現 paper 實驗的能力。

**Phase 0 — diff 分析**：`NNI.m` vs `NNI_ha.m` 兩檔各 270 行 ASCII；Unix `diff` 只 3 處差異（函式名 / `tol_theta` 1e-8 dead code vs 1e-12 / 13 行 halving while block 註解 vs active）；差異全集中在 `step_length` subfunction、100% halving-only、零 substantive 差異。推薦選項 A：`nni(halving=False, tol_theta=1e-12)` 兩 kwarg。

**Phase 1 — hazard §九 addendum**：`nni_hazard_analysis.md` 653 → 736 行（+83）加 §9.1-9.5。列三個開放問題（`tol_theta` default / `hit` 計數型別 / `hit_per_outer_history` dead→live 語意）全採最小變動答案；新增「Halving break 後非單調 λ_U」風險（MATLAB 原碼主 loop 無 fallback、port 照抄）。

**Phase 2 — `nni()` port** (commit `a1a334b` 內含)：7 個 surgical edits（signature + 5 段 docstring + step_length `if halving:` 分支）。halving=True 逐字 mirror NNI_ha.m line 167-186。Backward compat 驗證：`test_nni_parity.py` + `test_nni_basic` bit-identical to Session 4。

**Phase 3 — parity test + MATLAB reference** (commit `a1a334b`)：新增 `NNI_ha_with_history.m` (188) + `generate_nni_ha_reference.m` (129) + `test_nni_ha_parity.py` (362) + `nni_ha_reference.mat` (入 repo、force-added、跟 canonical 同策略)。

**Parity empirical finding（§9.6 補記）**：
- **iter 0-29 bit-identical**（前 30 slot 各 history 欄位全 machine-eps 內一致）
- **iter 30 noise-floor 分岔**：`min(temp)` 的 attained-index 在 MATLAB / Python 抽籤 flip、`lambda_L / res / chit / hit / x_history` 五個欄位同時 diverge
- **`halving=True` 把 stopping lottery 擴成 full-path lottery**：
  - MATLAB: nit=59, chit(halving)=109
  - Python: nit=32, chit=19
  - 最終 λ **bit-identical** |Δλ|=3.02e-14，eigenvector rel=4.44e-16
- **Parity 框架新增 multi-canary k_star**：掃 lambda_U/L / res / chit / hit / y_history 5 canaries 取最早 first_bad_iter、Tier 1 STRICT 縮到 `[0:k_star)`、`[k_star:K_common)` 降 Tier 3 INFO。`y_history` 單一 canary 晚一步（y 在 iter k 開頭用 iter k-1 的 clean x 算、attained-index flip 在 iter k 末才發生）、multi-canary 設計接住這種時間差
- **`print_neighborhood` bug fix**：clip `hi` 到 `min(len_ml, len_py)`、防 asymmetric-length 序列 IndexError

Q7 case final 結果：
| Tier | 結果 | 備註 |
|---|---|---|
| GATE | PASS | ML hit=109, PY hit=19（halving exercised 驗證）|
| Tier 1 STRICT [0:30) | 8/8 PASS | 各欄位 max err ≤ 2.84e-13 |
| Tier 2 APPROX | PASS | \|Δλ\|=3.02e-14、x rel=4.44e-16 |
| Tier 3 INFO | 記錄 | [30:32) 各欄位 diff、MATLAB 多 27 iter noise-floor 震盪 |

### 三、Fragility 模式擴充 — Rayleigh noise floor 的 halving-amplified 版本

**這不是第四種 fragility**、是 **#3 既有 `Rayleigh quotient local`**（NNI canonical 就揭露的現象、Day 4 memory `feedback_nni_rayleigh_quotient_noise_floor.md` 記錄）的 halving 延伸版本。`halving=True` 啟用時、每輪 Newton 多一個「`lambda_U_new > lambda_U + 1e-13` 才 halve」的 attained-index check；同一個 noise-floor rounding 差就在此處從「只影響停止條件」擴成「影響整個 halving 決策路徑」。結果是 MATLAB / Python 從某個 iter 起走完全不同路徑、但最終 λ 仍 bit-identical（Tier 2 PASS）。

memory 新增 `feedback_nni_ha_path_lottery.md`（Day 6 第 11 條）記錄 **parity pattern**（multi-canary k_star + Tier 1 STRICT `[0:k_star)` + Tier 3 INFO `[k_star:K_common)`）、**不記錄成新 fragility 類別**。對應 fragility 分類仍維持三種、只是 #3 增加「halving-amplified」subtype 標籤。

### 四、Streamlit demo 發布狀態驗證
- `.venv/bin/streamlit run streamlit_app/demo_v0.py` headless boot 三次（CP7a / CP7b-1 / CP7b-2 後）全綠：health 200 / home 200 / log 無 error
- `.venv/bin/python test_nni_parity.py` + `test_nni_ha_parity.py` + `test_tensor_utils.py::test_nni_basic` 全綠

---

## Day 7 (2026-04-27 → 2026-04-28) 完成摘要 — Session 7 方向變更（A/B/C → D：Streamlit demo 美化 + 部署）；P1-P5 全部完成、Live demo 上線

### 一、Session 7 方向變更

原計畫三選一（A LaTeX / B main_Heig / C 其他類別）、Session 7 開場後使用者**改方向為選項 D：Streamlit demo 美化 + 部署到 Streamlit Community Cloud**。

理由：使用者真實目標是把多年研究演算法做成 UI 網站給同行訪問；研究主軸（HNI vs NNI cross-validate）的 demo Day 5/6 已就緒、欠的是 polish + 公開可訪問。A/B/C 都是演算法 / 筆記方向、跟「對外可訪問」需求正交。

5 個 Phase 規劃（8h、簡化版）：
- P1  GitHub repo + push (1h) ✓
- P2  Streamlit theme + 自訂 CSS (1.5h) ✓
- P3  About 頁 (1h) ✓
- P4  收斂曲線 plot 強化 (1.5h) ✓
- P5  Deploy to Streamlit Community Cloud (0.5h) ✓ **2026-04-28 上線**：<https://csliu-toolbox.streamlit.app>

### 二、P1 — GitHub repo + push

- 建立 GitHub repo `chingsungliu/tensor-eigenvalue-toolbox`（public、Streamlit Community Cloud 免費版需要）
- repo URL：<https://github.com/chingsungliu/tensor-eigenvalue-toolbox>
- 第一次 push 流程：本機 `git remote add` → push 失敗（auth: `Device not configured`、Claude bash 無 TTY）→ 走 PAT + 真 Terminal.app 路徑、osxkeychain 自動存
- 中途有一次 PAT 洩漏在對話 transcript、立刻 revoke + 重新生成；建立「不貼 PAT 進對話、走真 Terminal」的 SOP

### 三、P2 — Streamlit theme + custom CSS

學術 light theme — numpy.org 風骨 + 個人海軍藍。

**`.streamlit/config.toml`**（repo root、Streamlit Cloud + 本機 cwd=repo-root 都讀）：
- `primaryColor = "#1B4F72"`（深海軍藍 academic accent）
- `backgroundColor = "#FFFFFF"` / `secondaryBackgroundColor = "#F5F7FA"` / `textColor = "#1F2937"`
- `[client] toolbarMode = "minimal"`（隱藏 Deploy 按鈕、保留 hamburger menu）

**`python/streamlit_app/assets/styles.css`**（91 行）：
- @import Source Serif Pro + JetBrains Mono Google Fonts
- Headings serif + navy；h2 加 1px navy 底線（學術 section 感）
- code/pre 用 JetBrains Mono；metric tile / form / expander 細灰邊（`1px solid #E5E7EB`、無陰影）
- sidebar h1 用 `!important` 強制 serif（emotion-CSS specificity battle 解法）
- 注入機制：`demo_v0.py` 加 `_inject_custom_css()` helper、`main()` 開頭呼叫

**Plotly deprecation 修正**：3 個 `st.plotly_chart` call site 從 `width="stretch"` 改 `use_container_width=True`（Streamlit 1.50 plotly_chart signature 沒 `width` 參數、字串落進 `**kwargs` → forward 給 plotly → warning）。同時驗證 `st.dataframe` / `st.image` 的 `width="stretch"|"content"` 是 1.50 first-class 參數**不是** deprecated、5 個 utility_renderers 的 call 不動。

本機 run 命令更新：`python/.venv/bin/streamlit run python/streamlit_app/demo_v0.py`（從 repo root、確保 config.toml 被讀到）。

### 四、P3 — About page

`python/streamlit_app/about.py` (128 行) — 研究者 bio + paper citations + GitHub link。

- 兩欄 columns([1, 2])：左 Author/Contact（中英文姓名 + 系所 + 雙 email mailto + phone + 地址 + 個人網頁）、右 Research interests / Education / Academic positions
- 4 個 paper placeholder（`PAPERS = [...]` constant、未來填 BibTeX 改一行 dict）
- GitHub URL + 一句 source code 說明
- 頂部「← Back to algorithms」button、走 session_state["show_about"] 路由

**Sidebar 整合**：sidebar 底部加 `st.divider()` + 「ℹ️ About this toolbox」full-width button。Step 1/Step 2 hierarchy 不動。

### 五、P4 — Result visualization (6 個新 plot + 1 個 helper)

按優先順序加 6 個 plot、覆蓋 5/6 個 renderer。`render_multilinear_compare` 維持原樣不動。

| # | Priority | Plot | Renderer | History 欄位 |
|---|---|---|---|---|
| 1 | (C) | min(x_i) per iter log | render_nni | `x_history` → `np.min(axis=0)` |
| 2 | (A) | Eigenvector grouped bar (sign-aligned) | render_hni_vs_nni Comparison tab | `h["x"]` + `nn["x"]` + dot-sign flip |
| 3 | (B) | λ trajectory linear overlay (3 series) | render_hni_vs_nni Comparison tab | `h["lam_hist"]` + `nn["lam_U_hist"]` + `nn["lam_L_hist"]` |
| 4 | (G) | Multi inner iter per outer bar | render_honi | `history["innit_history"]` |
| 5 | (E) | Multi halving per outer bar | render_multi | `history["hal_history"]` |
| 6 | (I) | Final λ across runs bar | render_eigenvalue_compare summary | `run["lam"]` (HONI) / `run["lam_U"]` (NNI) |

新 helper：`_plot_grouped_bar(series, title, ...)` — barmode="group" 兩 series 比對。

每 plot 都加「**讀圖要點**」caption + 對應 memory / research note 引用（fragility 三類別 + Rayleigh quotient noise floor + path lottery）。配色全程不 override colorway、單 series 自動 navy、多 overlay 用 plotly default (navy + 橘 + 綠)、grouped bar navy + 橘對比。

### 六、P5 — 部署成功（2026-04-28 上線）

P5 step 1+2（Day 7 上半場）：
- 拿掉 demo_v0.py 頂部 `st.warning("⚠️ Internal preview...")`（部署前一刻的決議）
- `requirements.txt` 補 `streamlit + plotly`（原本只有 `numpy + scipy`、Cloud 跑 streamlit 必裝；保留 `requirements_ui.txt` 不動向後相容）
- commit `b829b1c` — "Add Streamlit demo theme, About page, and result visualization" 整合 P2-P5
- push 上 GitHub `origin/main`

P5 step 3（Day 7 上半場暫停、2026-04-28 retry）：GitHub OAuth 恢復正常後重試、share.streamlit.io 部署成功。

**Live demo URL**：<https://csliu-toolbox.streamlit.app>
- Repository: `chingsungliu/tensor-eigenvalue-toolbox`
- Branch: `main`
- Main file: `python/streamlit_app/demo_v0.py`
- 部署成功後驗證：首頁 numpy 風海軍藍 theme 正常 / sidebar 兩層選單 / About 頁 / 4 Layer 3 renderer 全部可跑

### 七、本地 streamlit process 收尾

Day 7 留的 PID 1088（background ID `bbmr7yz3w`、port 8501）Day 8 開場一併殺掉、cloud demo 為主、本機 demo 之後需要才重啟。

### 八、Memory 寫入

無新 memory（P1-P5 都是工程實作 / 部署、無新 fragility 模式或數值學發現）。

---

## Git state（Day 7 完整收工、2026-04-28）

Day 7 上半場（1 筆、整合 P2-P5）：

```
b829b1c   Add Streamlit demo theme, About page, and result visualization
            (P2 theme + P3 About + P4 6 plots + P5 deploy prep)
```

Day 7 收尾（docs 更新、本日新增）：見 `git log` 最新一筆 — 更新 PROGRESS.md（P5 deploy 完成、Day 8 Level 2 規劃）+ README.md（live demo URL）+ 新建 journal.txt。

Day 6 結束時（6 筆）：

```
23d2d62   Update project docs after Session 6 Option B completion (NNI_ha port)
a1a334b   Port NNI_ha (halving variant) with parity framework extensions
a93d6af   Add multilinear solver comparison mode (CP7b-2)
a96dbdd   Add eigenvalue solver comparison mode (CP7b-1)
bd0d919   Add tensor upload support to Layer 3 renderers (CP7a)
```

Day 5 結束時：

```
0344ec8 Update CLAUDE.md context loader for Day 5 completion
8614197 Update PROGRESS.md after Day 5 (Layer 4 demo + architecture restructure)
068d894 Restructure demo to problem-driven two-level navigation (CP6)
817ccf8 Add HNI vs NNI comparison tile to Streamlit demo (CP5 Part B)
0e40f03 Integrate Layer 3 (Multi + HONI + NNI) into Streamlit demo (Session 5 Layer 4)
b6a2f07 Clean up KaTeX duplication artifacts in Rayleigh quotient report
a923768 Add Rayleigh quotient noise floor research note (English draft)
```

Day 4 結束時：

```
cb65311 Rename hni_status.md → algorithms_status.md with NNI as primary algorithm
7526d6c Update CLAUDE.md context loader for NNI completion
33f24c9 Update PROGRESS.md after NNI port (Layer 3+ complete, all algorithms ported)
55921bc Port NNI (canonical, 2020_HNI_Revised) to Python with linear solver menu (Session 4 階段 B)
ab454ea Add NNI.m hazard analysis before port (Session 4 階段 A)
```

---

## 下一個動作：Day 8 — Level 2（收斂速度優化）Phase 1: Baseline & Profiling

Day 7 部署完成、Level 1（port + parity + demo + 部署）整條線完整。Day 8 起切到 **Level 2：收斂速度優化**。

### Phase 1 任務概要

**Sub-step 1.1**（Day 8 開場、明天會給詳細 prompt）：建立 benchmark suite — 為 4 個 Layer 3 演算法（Multi / HONI exact / HONI inexact / NNI spsolve）在固定 test cases 上 measure baseline 收斂速度（iter count、wall time、residual trajectory），作為後續 Phase 2+ 優化的對照基準。

**詳細 prompt 明天提供**、Day 7 不開新工作。

---

### Day 8 開場 checklist

1. `git log --oneline -10` 看最近 commits（包含今日 docs 收尾的 commit）
2. `cat PROGRESS.md` 看 Day 7 收工狀態 + Day 8 任務（本檔）
3. 確認 cloud demo 仍正常：開 <https://csliu-toolbox.streamlit.app> 載入首頁
4. 等使用者貼 Sub-step 1.1 詳細 prompt

### 本機 streamlit 重啟（如果 Day 8 過程需要）

Day 7 收尾已殺 PID 1088。需要時：
```bash
cd ~/Projects/my-toolbox
python/.venv/bin/streamlit run python/streamlit_app/demo_v0.py --server.headless=true --server.port=8501
```

---

### 後續方向（Level 2 完成後、原 Session 7 A/B/C 三選一仍適用）

A：LaTeX 轉換研究筆記 PDF 化（0.5–1h）／B：`main_Heig.m` driver port 重現 paper benchmark（2–3h）／C：其他類別 port（Optimization/QP、NLS/BEC、M-matrix、GINI、視類別工作量不一）。詳見前一版 PROGRESS.md（git log）或 `matlab_ref/GLOBAL_INVENTORY.md`。

---

## 重啟這個專案的步驟

```bash
cd ~/Projects/my-toolbox
claude
```

進新 session 後，跟 Claude 說：

> 我回來了。用 `git log --oneline` 看進度、`cat PROGRESS.md` 看下個動作。

Claude 會：
1. 讀 PROGRESS.md 了解 Day 1-5 完整狀態
2. 讀 memory（10 條、可主動 Read）
3. 看 `docs/algorithms_status.md` 了解三個演算法系統現況（1 分鐘）
4. 看 `docs/papers/rayleigh_quotient_noise_floor_en.md`（Day 5 新增的 NNI 研究筆記）
5. 確認下一個動作是 Session 6 的 A / B / C（或其他方向）
6. 在使用者同意後啟動 Session 6

---

## 重要檔案地圖（給未來的自己或 Claude）

| 檔案 | 用途 |
|---|---|
| `PROGRESS.md`（本檔） | Session checkpoint、下一步指南 |
| `docs/algorithms_status.md` | 三個演算法系統（NNI/HNI/Multi）整體狀態 |
| `docs/papers/rayleigh_quotient_noise_floor_en.md` | NNI 研究筆記英文版（Day 5 新增、KaTeX 清理後純 LaTeX source） |
| `matlab_ref/hni/README.md` | HNI port 進度表 |
| `matlab_ref/GLOBAL_INVENTORY.md` | 1304 個 `.m` 檔全局地圖 |
| `matlab_ref/NNI_HNI_inventory.md` | HNI/NNI 線的詳細盤點（含 Section H：NNI canonical 決策） |
| `matlab_ref/poc_iteration/README.md` | Per-iteration parity 框架的設計文件 |
| `docs/superpowers/specs/2026-04-21-streamlit-demo-v0-design.md` | Demo v0 正式 spec |
| `docs/superpowers/notes/nni_hazard_analysis.md` | NNI port 階段 A 的 hazard analysis（653 行） |
| `python/tensor_utils.py` | Layer 1/2/3 共 8 個函式（5 工具 + Multi + HONI + NNI） |
| `python/streamlit_app/demo_v0.py` | Router（79 行）、dispatch 到 problem → algorithm 兩層 |
| `python/streamlit_app/_internal/` | Layer 1/2 utilities（UI 隱藏、保留可 import） |
| `python/streamlit_app/problems/tensor_eigenvalue/` | 4 個 Layer 3 renderer + Q7 defaults |
| `.gitignore` | 已排除 `.venv/`、`*.mat`、`.DS_Store`、`__pycache__/`、`source_code/` |

---

**收工狀態檢查**：
- [x] Layer 1+2 port 完、parity 全部通過
- [x] demo v0 上線、6 函式可互動
- [x] per-iteration parity 框架 POC 驗證
- [x] Layer 3 Multi port（Day 2 完成、Q5 parity 到 machine epsilon）
- [x] halving fragility 分析 + 延後策略記錄
- [x] Layer 3 HONI port（Day 2 末完成、exact + inexact 通過 tier parity）
- [x] shift-invert fragility 分析 + 三 Tier 框架記錄
- [x] **Layer 3+ NNI port**（**Day 4 完成、spsolve + gmres、3-Tier parity 全綠**）
- [x] **Rayleigh quotient noise floor 分析 + 跨三演算法 fragility 模式整合**
- [x] memory 寫入（Day 6 結束 11 條、新增第 11 條 `feedback_nni_ha_path_lottery.md`）
- [x] 演算法系統 milestone 文件（`docs/algorithms_status.md`、重構自 `hni_status.md`）
- [x] **Layer 4 demo 整合**（**Day 5 CP1–CP5 完成、Multi + HONI + NNI + HONI vs NNI tile 全部上線**）
- [x] **Streamlit 架構重構**（**Day 5 CP6 完成、problem-driven 兩層選單、demo_v0.py 削減至 79 行**）
- [x] **研究筆記英文版 + KaTeX 清理**（**Day 5 完成、`docs/papers/rayleigh_quotient_noise_floor_en.md`**）
- [x] **CP7 — Streamlit demo UI 擴展**（**Day 6 完成、upload + multi-run comparison、ALGORITHMS 6 entries**）
- [x] **NNI_ha port**（**Day 6 完成、Phase 0-3、`halving=True` kwarg、multi-canary k_star parity 框架**）
- [x] **Fragility 模式擴充 #3 halving-amplified subtype**（**Day 6、parity pattern 可重用於 shift-invert / Rayleigh 類 solver**）
- [x] **P1 — GitHub repo + push**（**Day 7、`chingsungliu/tensor-eigenvalue-toolbox` public repo、PAT + osxkeychain SOP**）
- [x] **P2 — Streamlit theme + custom CSS**（**Day 7、numpy 風海軍藍 `#1B4F72` + serif headings + JetBrains Mono、`.streamlit/config.toml` + `styles.css`**）
- [x] **P3 — About page**（**Day 7、`about.py` 128 行、研究者 bio + 4 paper placeholder + GitHub link、sidebar 底部 button 路由**）
- [x] **P4 — Result visualization**（**Day 7、6 個新 plot：NNI min(x_i) / HNI vs NNI eigenvector overlap + λ trajectory / HONI innit bar / Multi hal bar / 多 run final λ bar、+ `_plot_grouped_bar` helper**）
- [x] **P5 — Deploy to Streamlit Community Cloud**（**2026-04-28 上線：<https://csliu-toolbox.streamlit.app>**）
- [ ] **Day 8 起：Level 2 收斂速度優化、Phase 1 Baseline & Profiling、Sub-step 1.1 明日詳細 prompt**
- [ ] 英文筆記 LaTeX/PDF（原 Session 7 選項 A、Level 2 後可選）
- [ ] `main_Heig.m` driver port（原 Session 7 選項 B、Level 2 後可選）
- [ ] 其他類別 port（原 Session 7 選項 C、Level 2 後可選）
