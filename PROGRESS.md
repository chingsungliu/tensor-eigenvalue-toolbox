# PROGRESS.md — Session checkpoint

**最後更新**：2026-04-22（Day 3 收工）
**狀態**：Layer 3（Multi + HONI）完整完成、HNI 系統 port 全綠、待決定 Layer 4 整合 vs NNI canonical port

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

## 下一個動作：Session 4 — 二選一（待使用者決定）

### 選項 A：Layer 4 整合（Streamlit demo 加 Multi + HONI 兩個 renderer）

**為什麼**：
- HNI 系統 Layer 1/2/3 全部 port 完、parity 全綠 → 把使用者面向的 demo 補齊、HNI 的「可瀏覽器互動」價值才實現
- 新增 2 個 renderer（Multi 一個、HONI 一個），照 demo_v0 已驗證的 §9 擴充 contract（4 處改動：import / helper / renderer / dict 行）
- Multi renderer：左欄輸入 m / n / tol、右欄印 nit、residual decay、最終 u（plot u_history 收斂軌跡）
- HONI renderer：左欄輸入 m / n / tol / linear_solver、右欄印 lambda、x、收斂曲線；可選擇 exact / inexact 對比

**預計工作量**：1.5-2 小時（demo 已成熟、純擴充）

**產出**：`python/streamlit_app/demo_v0.py` 從 6 函式擴到 8 函式；HNI 線端到端可 demo

---

### 選項 B：NNI canonical port（HNI 並列演算法）

**為什麼**：
- NNI（Non-negative tensor Iteration）是 HNI 的兄弟線、同樣解非負張量最大特徵值、但 algorithm 完全不同
- canonical 候選 + 版本 inventory 已在 `matlab_ref/NNI_HNI_inventory.md` Section H 列出、待使用者拍板
- 一旦 NNI 也 port 完、可在 demo 對比 HNI vs NNI 的收斂行為（同一 AA 兩個演算法）

**預計工作量**：unknown — 取決於 canonical 版本的複雜度
- 階段 A：確認 NNI canonical 版本（使用者決定）+ hazard analysis
- 階段 B：port + parity（依 Multi/HONI 經驗、預計 4-8 小時）

**產出**：`python/tensor_utils.py::nni`（或拆檔）+ parity test + hazard analysis 文件

---

### 對比（給使用者決策參考）

| 維度 | A：Layer 4 整合 | B：NNI port |
|---|---|---|
| 工作量 | 短（1.5-2h） | 中-長（4-8h） |
| 風險 | 低（demo pattern 已驗證） | 中（新演算法、新陷阱） |
| 立即價值 | HNI 線可瀏覽器 demo | 工具箱演算法庫 +1 |
| 適合時機 | 想看 HNI 階段成果 | 想推進 port 廣度 |
| 後續解鎖 | NNI 完成後可加進同一 demo | 完成 + 整合後可 HNI vs NNI 對比 demo |

---

## 重啟這個專案的步驟

```bash
cd ~/Projects/my-toolbox
claude
```

進新 session 後，跟 Claude 說：

> 我回來了。用 `git log --oneline` 看進度、`cat PROGRESS.md` 看下個動作。

Claude 會：
1. 讀 PROGRESS.md 了解 Day 1 + Day 2 + Day 3 完整狀態
2. 讀 memory（9 條、可主動 Read）
3. 看 `docs/hni_status.md` 了解 HNI 系統現況（1 分鐘）
4. 確認下一個動作是 A 或 B（或使用者指定其他方向）
5. 在使用者同意後啟動 Session 4

---

## 重要檔案地圖（給未來的自己或 Claude）

| 檔案 | 用途 |
|---|---|
| `PROGRESS.md`（本檔） | Session checkpoint、下一步指南 |
| `matlab_ref/hni/README.md` | HNI port 進度表（每個 layer 的狀態） |
| `matlab_ref/GLOBAL_INVENTORY.md` | 1304 個 `.m` 檔全局地圖 |
| `matlab_ref/NNI_HNI_inventory.md` | HNI/NNI 線的詳細盤點（含 Section H：NNI canonical 決策） |
| `matlab_ref/poc_iteration/README.md` | Per-iteration parity 框架的設計文件 |
| `docs/superpowers/specs/2026-04-21-streamlit-demo-v0-design.md` | Demo v0 正式 spec |
| `python/tensor_utils.py` | 5 個已 port 的 tensor 工具 |
| `python/streamlit_app/demo_v0.py` | 6 個函式的 Streamlit UI |
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
- [x] memory 寫入（9 條）
- [x] HNI 系統 milestone 文件（`docs/hni_status.md`）
- [ ] Layer 4：demo 加 Multi + HONI（選項 A）
- [ ] NNI canonical 決策 + port（選項 B）
