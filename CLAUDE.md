# CLAUDE.md — my-toolbox 新 session context loader

**這個檔的用途**：讓 Claude Code 每次開新 session 都能在 5 秒內把專案 context 還原到上次收工狀態。Claude Code 在啟動時會**自動讀取**專案根目錄的 `CLAUDE.md`（這是 built-in 行為，不需要任何設定），所以這裡寫的內容一定會進入新 session 的 context window。

---

## 1. 專案簡介

這個 repo 是一個**個人演算法工具箱 + MATLAB→Python port sandbox**。

**起點**（已完成）：用 `gaussian_blur` 建立完整 port 流程（MATLAB reference → Python 實作 → parity test，`matlab_compat` flag 設計），max error ~1e-16。

**當前階段**：把使用者自己多年累積的 **Tensor Eigenvalue 系列研究演算法**（`source_code/Tensor Eigenvalue Problem/`，1304 個 `.m` 檔中最密集的一塊，83%）逐層 port 到 Python。路徑是：

1. **HNI 線**（先）：5 個張量工具 `tenpow/tpv/sp_tendiag/ten2mat/sp_Jaco_Ax` → `Multi`（內層 Newton + 三等分 halving solver）→ `HONI`（外層 eigenvalue 迭代）→ `main_Heig` demo
2. **NNI 線**（後）：並列獨立的另一個求非負張量最大特徵值演算法
3. **其他類別**：Optimization/QP (ISM)、Nonlinear Schrödinger/BEC、M-matrix Iteration、Generalized Eigenvalue (GINI)

**Demo 端**：`python/streamlit_app/demo_v0.py` 是瀏覽器內互動介面，每新 port 完一個函式就加進去。目前 6 個函式（gaussian_blur + 5 個張量工具）。

**Day 1 收工狀態（2026-04-21）**：HNI Layer 1+2 完成、5 個工具 parity 全過、Streamlit demo 上線、per-iteration parity POC 驗證完、14 個 commit。
**Day 2 收工狀態（2026-04-22）**：HNI Layer 3 step 1/2（Multi port）完成、Q5 case parity 通過到 machine epsilon（max err 6.7e-16）、five history 欄位全綠（u/res/theta/hal/v）。halving path 因 Multi 在 m≥3 + random AA 沒有「健康 halving」regime，延後到 HONI integration test 自然觸發（見 `memory/feedback_multi_halving_fragility.md`）。**下一步是 HONI port（Session 3）**。

長期願景：所有研究演算法都有 Python port + parity + Streamlit demo；`source_code/` 的 490MB MATLAB 遺產變成一個活的、可 reproduce 的個人工具箱。

---

## 2. 新 session 必讀 — 8 條 memory

Claude Code 的 memory auto-load 機制在當前版本**實測沒有自動載入**（Day 2 session 開場 context 裡完全沒有 memory 內容）。為避免下次又遺漏，**新 session 一開始請主動 `view` 下列 8 條 memory 一遍**：

**路徑**：`/Users/csliu/.claude/projects/-Users-csliu-Projects-my-toolbox/memory/`

| 檔案 | 類別 | 一行摘要 |
|---|---|---|
| `MEMORY.md` | 索引 | 其他 8 條的快速 overview |
| `user_role.md` | user | 使用者角色、溝通語言、collaboration 風格 |
| `project_layout.md` | project | `~/Projects/my-toolbox` 結構 + venv 位置 + 與 Google Drive mirror 的關係 |
| `project_matlab_environment.md` | project | 本機 MATLAB **沒有** Image Processing Toolbox — reference 實作只能用 base MATLAB |
| `project_big5_encoding.md` | project | 舊研究 `.m` 檔可能是 Big5 而非 UTF-8 — 用 `iconv` 救；那些註解是作者自己的重要筆記 |
| `feedback_matlab_to_python_port.md` | feedback | **五大陷阱 checklist**（port 時主動對照）— 下面第 4 節有 headline 速查 |
| `feedback_multi_version_research_code.md` | feedback | 遇到 `ver2/ver3/ver7.1/ver8.0/...` 不要亂挑、列出所有版本讓使用者決定 canonical |
| `feedback_per_iteration_parity.md` | feedback | 迭代演算法的 parity 要**逐 iter 比對、報 first_bad_iter**、不要只比最終輸出 — 下面第 5 節有 headline 速查 |
| `feedback_multi_halving_fragility.md` | feedback | **Day 2 新增**：Multi.m 的 halving path 在 m≥3 + random AA 沒有「健康 halving」sweet spot；halving 的 parity 延後到 HONI integration test |

**開 session 建議動作**：
```bash
ls ~/.claude/projects/-Users-csliu-Projects-my-toolbox/memory/
```
然後**用 Read 工具讀需要的那幾條**（不用全讀、看當下任務相關的）。

---

## 3. 使用者簡介（核心關鍵、不依賴 memory）

> 即使 memory 完全讀不到，下面這幾點也必須當預設。

- **溝通語言**：**繁體中文**（使用者一律繁中、回覆也用繁中）
- **專業背景**：**研究數學家**（非 software engineer）— 熟 MATLAB、熟線性代數/張量/特徵值等數學，對 Python 生態（numpy、scipy、pip、venv）和 Git/GitHub 不熟
- **合作風格**：
  - 解釋技術決策時講「為什麼」而不只是「怎麼做」
  - 不要假設他會 `pip install` / `git rebase` / `.venv/bin/activate` 這類操作、需要時直接給指令
  - 對「數值正確性」極度在意（所以才有 parity testing 框架）
  - 想建立「可 reproduce、可 demo」的個人工具箱（不是要發 SaaS 產品）
- **節奏偏好**：重要任務會要求「先讀懂再動手」（例如 Multi port 分 A/B 兩階段、階段 A 只寫 hazard analysis）
- **決策權**：**多版本研究程式碼的 canonical 選擇、algorithm 替換、benchmark 選擇 — 使用者決定，我負責把選項攤開，不自行拍板**

---

## 4. 五大 MATLAB→Python 陷阱（headline 速查）

詳細版在 `memory/feedback_matlab_to_python_port.md`。下面是 headline：

1. **邊界處理**：`conv2('same')` = zero-pad；`imgaussfilt` default = replicate；scipy default = reflect。三個互不相同、邊界誤差 ~1e-1。
2. **濾波器/核截斷寬度**：MATLAB `imgaussfilt` 是 `2*ceil(2σ)+1`；常見 script 用 `ceil(3σ)`；scipy `truncate=4.0`。差異 ~1e-4。
3. **Indexing 1-based vs 0-based**：MATLAB 是 1-based，Python 是 0-based。`for i = 1:n`、`A(i,j)`、`find(...)` 輸出索引、任何用索引做算術的地方都要重新 offset。
4. **Array storage order（column-major vs row-major）**：MATLAB 是 Fortran 序、numpy 是 C 序。**只要出現 `reshape` / `flatten` / `A(:)` / n-dim linear index 就會觸發**。Fix：用 `order='F'`，或改用 `moveaxis`/`einsum` 完全避開 flatten-reshape。錯誤量級 1e-1 到 O(1)。
5. **迭代控制流 + sparse 數值**（2026-04-22 新增）：迭代演算法特有的坑。
   - Line search 比例（`/3` 不是 `/2` — 看原碼字面值、別看註解）
   - State snapshot（`u_old = u; v_old = v` 凍結、halving 都從 snapshot 重算）
   - Pre-allocate residual buffer（`res = (na+nb)*ones(100,1)`）
   - 硬上限 `nit < 100`（別改成 `<=`）
   - Dtype 保存（`b ** (m-1)` on int → int；進函式就 `asarray(dtype=float64)`）
   - `spsolve` 偏好 CSC、CSR 會 warn；`scipy.sparse.linalg.norm` 才吃 sparse
   - `||`/`&&` short-circuit 順序逐字照抄、別 clean up
   - 不要 mutate 呼叫端傳入的 array（MATLAB 有 copy-on-write 隱藏這個語意）

---

## 5. Per-iteration parity 機制（headline 速查）

詳細版在 `memory/feedback_per_iteration_parity.md`。一句話總結：**迭代演算法的 parity 不能只比最終輸出、要逐 iter 比對、報 first_bad_iter（最早發散的 iteration），不是 max_err**。

三件組 API（reference: `python/poc_iteration/test_sqrt2_newton_parity.py`）：
- `find_divergence(matlab_seq, python_seq, tolerance, name="")` → dict with `first_bad_iter`
- `report(result)` → PASS 一行摘要、FAIL 顯示 first_bad_iter + MATLAB/Python 值 + diff
- `print_neighborhood(matlab, python, center, radius=2)` → 發散點前後各 N 步的值

**MATLAB side**：reference function 存每 iter 的 state，initial state 放 index 1（MATLAB）/ index 0（Python）。`u_history` / `res_history` / `theta_history` / `hal_history` ...，最後 `save` 成 `.mat`。
**Python side**：演算法加 `record_history=False` flag，`True` 時 return dict of history arrays。Production 用不受影響、parity test 時才打開。
**Tolerance 慣例**：通過門檻 `1e-10`、期望 `0`（bit-identical）或 `1e-14 ~ 1e-16`（float 累積）。若某 iter 超過 `1e-10` 就是真分岔、去 debug 別放水。

POC 已驗證通過（`matlab_ref/poc_iteration/` + `python/poc_iteration/`）— 11 步 Newton sqrt(2) bit-identical。

---

## 6. 專案結構速查

```
~/Projects/my-toolbox/
├── README.md                 初代 README（僅介紹 gaussian_blur + 三大陷阱、**相對於現況較舊**）
├── CLAUDE.md                 ★ 你在讀的這份（context loader）
├── PROGRESS.md               每次 session 收工會更新的 checkpoint
├── requirements.txt          numpy, scipy
├── requirements_ui.txt       streamlit 等 UI 依賴
├── .gitignore                排除 .venv/、*.mat、__pycache__/、source_code/
│
├── matlab_exercise/          Port 練習的 MATLAB 原碼 + reference 生成
│   ├── gaussian_blur.m
│   ├── generate_reference.m
│   └── reference.mat         （git 不追蹤）
│
├── matlab_ref/               正式 port 對象的 MATLAB reference（canonical 副本）
│   ├── hni/                  HNI 線 3 個檔（Multi.m / HONI.m / main_Heig.m） + README
│   ├── poc_iteration/        per-iteration parity POC 的 MATLAB 端
│   ├── GLOBAL_INVENTORY.md   source_code/ 的全局盤點（391 行、5 大類）
│   └── NNI_HNI_inventory.md  HNI / NNI 詳細盤點（Section H = NNI canonical 決策）
│
├── python/                   所有 Python port + 測試
│   ├── .venv/                （git 不追蹤）Python 3.9.6、numpy 2.0.2、scipy 1.13.1
│   ├── gaussian_blur.py      第一支 port、machine epsilon parity
│   ├── tensor_utils.py       5 個張量工具 (tenpow/tpv/sp_tendiag/ten2mat/sp_Jaco_Ax)
│   ├── test_tenpow_parity.py     (逐支函式 parity test)
│   ├── test_tpv_parity.py
│   ├── test_sp_tendiag_parity.py
│   ├── test_ten2mat_parity.py
│   ├── test_sp_Jaco_Ax_parity.py
│   ├── test_tensor_utils.py  (sanity unit tests, 不依賴 MATLAB)
│   ├── test_gaussian_blur.py
│   ├── test_parity.py
│   ├── poc_iteration/        per-iteration parity POC 的 Python 端
│   └── streamlit_app/
│       └── demo_v0.py        ★ 瀏覽器 demo（目前 6 函式）
│
├── docs/
│   └── superpowers/
│       ├── specs/            正式 spec（例：streamlit demo v0 design）
│       └── notes/            工作筆記（例：multi_hazard_analysis.md）
│
└── source_code/              （git 不追蹤、490MB）Google Drive 副本的 MATLAB 檔
                              1304 個 .m、5 大類演算法，是後續 port 的 mining ground
```

**本機啟動 Streamlit demo**：
```bash
cd ~/Projects/my-toolbox/python
.venv/bin/streamlit run streamlit_app/demo_v0.py
```

---

## 7. 每次 session 開場 checklist

依序跑完這 4 步、然後回報「Day X 收工狀態、下個動作是 Y、要不要開工？」：

1. **看最近 commits**
   ```bash
   git log --oneline -10
   ```

2. **讀 PROGRESS.md 看下個動作**
   ```bash
   cat PROGRESS.md
   ```
   （或用 Read 工具讀；PROGRESS.md 每次 session 收工都會更新）

3. **若正在做某支 port**：讀對應的 hazard analysis
   ```bash
   ls docs/superpowers/notes/
   # 例如正在做 Multi：
   cat docs/superpowers/notes/multi_hazard_analysis.md
   ```

4. **需要時主動讀 memory**（不用全讀、按任務需要）
   ```bash
   ls ~/.claude/projects/-Users-csliu-Projects-my-toolbox/memory/
   # 然後用 Read 工具讀相關的幾條
   ```

**關鍵原則**：**讀完才動手**。使用者喜歡把大任務拆 A/B 階段（例如 Multi port 分階段 A=hazard analysis、階段 B=實作），階段 A 收工等確認、再進階段 B。不要搶跑。

---

## 8. 環境 & 已知限制

- **Python venv**：`~/Projects/my-toolbox/python/.venv/`（Python 3.9.6、numpy 2.0.2、scipy 1.13.1）— 用 `.venv/bin/python`、不要用 system `python3`（system 沒 numpy）
- **MATLAB toolbox**：本機 MATLAB **沒有 Image Processing Toolbox**（也未確認 Signal Processing Toolbox）— reference 實作只能用 base MATLAB（`conv2`/`fft2`/`ifft2` 可以、`imgaussfilt`/`imfilter` 不行）
- **Google Drive mirror**：`~/我的雲端硬碟/CSLiu/my-toolbox/` 是**只讀**參考、不要動
- **`source_code/`**：490MB、1304 個 `.m`、被 `.gitignore` 排除 — 是 port 的 mining ground、不是 port 對象本身（canonical 會挑出來 copy 到 `matlab_ref/`）
- **Memory auto-load**：**當前 Claude Code 版本實測沒有自動載入 memory**（Day 2 驗證）。本 CLAUDE.md 是 fallback — 關鍵資訊（使用者簡介、五大陷阱、per-iter parity）已經內嵌，就算 memory 一個都沒讀到也能工作

---

## 9. 這份 CLAUDE.md 本身的維護

**何時更新**：
- 加新 memory → 更新第 2 節清單
- 陷阱 checklist 有新項目 → 更新第 4 節 headline
- Per-iteration parity 機制改版 → 更新第 5 節
- 專案大結構變動（新增 top-level 目錄、改 venv、換 MATLAB 環境）→ 更新第 6 / 8 節
- **當前階段變化**（例如 Multi port 完成、進 HONI）→ 更新第 1 節「當前階段」

**何時不動**：
- 單一 commit 的小進度 → 只寫進 PROGRESS.md、本檔不動
- 特定函式的 hazard analysis → 寫在 `docs/superpowers/notes/`、本檔不重複
