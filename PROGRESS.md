# PROGRESS.md — Session checkpoint

**最後更新**：2026-04-22（Day 2 收工）
**狀態**：21 commits on `main`、working tree clean、無未 commit 變更

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

## Git state（21 commits、Day 2 新增 4 筆）

Day 2 新增（topmost）：

```
<pending> Update PROGRESS.md and CLAUDE.md after Multi port completion   ← 本 commit
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

## 下一個動作：Session 3 = Port `HONI`（Layer 3 step 2/3）

**為什麼 HONI 緊接 Multi**：
- Multi 是 HONI 的內層 solver、Multi port 已完成 + parity 到 machine epsilon
- HONI 會在外層 eigenvalue iteration 中**自然呼叫 Multi**，halving path 會在 HONI 的「困難 iter」被觸發 — 這是 Multi halving 被設計要 cover 的 regime（見 `memory/feedback_multi_halving_fragility.md`）
- HONI 完成後才有完整的 eigenvalue 解算、Streamlit demo 才能加新 tile

**檔案**：`matlab_ref/hni/HONI.m`

**預計工作量**：2-3 小時
- HONI 外層結構預計約 150-200 行 MATLAB（比 Multi 大）
- 主要 port 風險類型跟 Multi 類似（iteration control flow + 可能有 column-major 在 unfolding 處理）
- Multi 已 port、Multi 的 5 個 dependency 都已 port、所以 HONI 只需 port 自身外層結構 + possibly 新 helper

**執行前應做的事（建議）**：
- 先寫 HONI hazard analysis（類似 Multi 的 A 階段），放 `docs/superpowers/notes/honi_hazard_analysis.md`
- 逐行找 HONI.m 裡的 indexing / reshape / control-flow / sparse dispatch 陷阱
- 先列 open questions、讓使用者決定後再動手實作

---

## 重啟這個專案的步驟

```bash
cd ~/Projects/my-toolbox
claude
```

進新 session 後，跟 Claude 說：

> 我回來了。用 `git log --oneline` 看進度、`cat PROGRESS.md` 看下個動作。

Claude 會：
1. 讀 PROGRESS.md 了解 Day 1 + Day 2 完整狀態
2. 讀 memory（8 條、可主動 Read）
3. 確認下一個動作是 Port HONI（或使用者指定其他方向）
4. 在使用者同意後啟動 Session 3

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
- [x] memory 寫入（8 條）
- [x] Git working tree clean
- [ ] Layer 3 HONI port（Session 3）
- [ ] Layer 3 整合 + demo 加 Multi/HONI
- [ ] NNI canonical 決策 + port
