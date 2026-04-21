# PROGRESS.md — Session checkpoint

**最後更新**：2026-04-21（Day 1 收工）
**狀態**：14 commits on `main`、working tree clean、無未 commit 變更

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

### 五、Memory 寫入（現共 7 條）

| Memory 檔 | type | 說明 |
|---|---|---|
| `user_role.md` | user | 使用者背景 / 偏好 |
| `project_layout.md` | project | 專案目錄結構 |
| `feedback_matlab_to_python_port.md` | feedback | **四大陷阱** checklist（boundary / truncation / 1-vs-0 indexing / column-major） |
| `project_matlab_environment.md` | project | 無 Image Processing Toolbox、用 base MATLAB |
| `project_big5_encoding.md` | project | Big5 編碼的舊 MATLAB 程式處理法 |
| `feedback_multi_version_research_code.md` | feedback | 多版本研究程式碼要列所有版本、讓使用者決定 canonical |
| `feedback_per_iteration_parity.md` | feedback | **今天新增**：迭代演算法的 parity 機制 |

---

## Git state（14 commits）

```
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

（這次 commit 後會變 15 筆：最後一筆是 PROGRESS.md 本身）

---

## 下一個動作：Session 2 = Port `Multi`（Layer 3 第一個迭代演算法）

**為什麼先做 Multi**：
- 它是 Layer 3 最小的一個（89 行 MATLAB，Newton 迭代 + 三等分 halving line search）
- 是 HONI 的內層 solver，port Multi 是 port HONI 的前置
- 會成為**per-iteration parity 框架**的第一次實戰驗證
- POC（D3a）已經把所有機制跑過一次、框架待用

**預計工作量**：1-2 小時（比 Layer 1/2 的單函式多、因為要設計 history 儲存 + 逐步對帳）

**執行前需確認**：
- Layer 3 的 port 順序照 `matlab_ref/hni/README.md` 的 C5 計畫（Multi 先、然後 HONI）
- `feedback_per_iteration_parity.md` 的 mechanism 原則要主動套用

---

## 重啟這個專案的步驟

```bash
cd ~/Projects/my-toolbox
claude
```

進新 session 後，跟 Claude 說：

> 我回來了。用 `git log --oneline` 看進度、`cat PROGRESS.md` 看下個動作。

Claude 會：
1. 讀 PROGRESS.md 了解 Day 1 完整狀態
2. 讀 memory（7 條、自動載入）
3. 確認下一個動作是 Port Multi
4. 在你同意後啟動 Session 2

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
- [x] memory 寫入（7 條）
- [x] Git working tree clean
- [ ] Layer 3 Multi port（明天 Session 2）
- [ ] Layer 3 HONI port
- [ ] Layer 4 整合 + demo 加 Multi/HONI
- [ ] NNI canonical 決策 + port
