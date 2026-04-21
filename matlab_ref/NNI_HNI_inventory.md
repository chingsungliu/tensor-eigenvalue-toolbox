# NNI / HNI MATLAB 原始碼統整報告

**盤點範圍**：`/Users/csliu/我的雲端硬碟/CSLiu/my-toolbox/MATLAB/Tensor Eigenvalue Problem/` 底下 5 個子資料夾
**掃描方式**：5 個 Explore subagent 平行讀取，只看 `.m` 檔，忽略 `.mat` / `.pdf` / `.docx`
**原檔未動**：所有原檔仍在 Google Drive，本次沒有複製任何 `.m` 到主 repo

---

## 摘要（先看這段）

1. **全部 5 個資料夾都是 zero toolbox dependency**：沒有人用 Tensor Toolbox（`ttv`/`ttm`/`tenmat` 等都沒出現），也沒有用 Optimization / Image Processing / Signal Processing / Symbolic。純 base MATLAB（主要靠 `sparse`、`kron`、`reshape`、`norm`、`\`）。**非常適合 port** — 不會因為 toolbox 缺失而卡住。

2. **三個 `2020_HNI_*` 確認是同一演算法的版本演進**：
   - `Heig` = 開發版（有 `untitled*.m` 佔位檔、`Multi_shift.m` 有語法錯誤、`NNI.m`/`NNI_two.m` 幾乎重複）
   - `Revised` = 修訂版（`HNI.m` 被改名為 `HONI.m`、多了 `Test_Heig2/3/4.m` 參數掃描腳本、Big5 編碼的中文註解）
   - `submitt` = 投稿精簡版（只保留 3 個檔：`main_Heig.m` + `HONI.m` + `Multi.m`，`HONI.m` 日期 2020/12，`main_Heig.m`/`Multi.m` 更新到 2021/06）
   - **你的猜測完全正確。**

3. **`Nonnegative tensor` 資料夾比你描述的大很多**：不是 13 個 `.m`，**實際總共 659 個 `.m` 檔**（含 7 個 `tensor_packege_ver*` 子版本、獨立 `H_eigenpair_ver6/`、`NT_arnoldi/`、甚至還嵌了一份 `tensorlab_2016-03-28/` 第三方函式庫）。這不是一個可以「整包 port」的 target，需要先縮小範圍（見 Section E）。

4. **NNI 和 HNI 不是兩條平行演算法，而是同一個問題的兩個 solver**（見 Section A）。

5. **`2020_HNI_Revised/Test_Heig*.m` 的中文註解**：Big5 編碼，UTF-8 工具讀出是亂碼。我已用 `iconv` 抽出來，附在 Section B。port 時如果要保留註解，需要做編碼轉換。

---

## A. NNI / HNI 分類

原本你的猜測是兩條獨立的演算法線。實際情況更細：

### 兩者都是「非負張量最大特徵值」的演算法，但策略不同

| | HNI / HONI | NNI |
|---|---|---|
| 全名推測 | **H-eigenvalue Newton Iteration** / H-eigenvalue Optimization via Newton Iteration | **Nonnegative Newton Iteration** |
| 結構 | 雙層迴圈：外層更新 λ、內層用 Multi.m 解多線性系統 `(λI - A)u^(m-1) = b` | 單層 Newton：直接對 `F(x) = Ax^(m-1) - λx^(m-1)` 做 Jacobian-based iteration |
| 特性 | 收斂需要內層 linear solve，成本高但穩定 | 收斂快但你自己在 `Test_Heig3.m` 裡面有註解警告：**「一般來說 NNI 相當穩定，不會隨著 n,m 影響，但接下來的例子會讓 NNI 難以使用」** |
| 收斂理論 | 需要 m-matrix 性質、shifted-inverse-style 論證 | 需要 Jacobian non-singular 條件 |

### 資料夾歸類

```
HNI / HONI 線（主要）
├── 2020_HNI_Heig         ← 開發版（含 HNI.m 原名）
├── 2020_HNI_Revised      ← 修訂版（HNI → HONI 改名）
└── 2020_HNI_submitt      ← 投稿版（最小子集）

NNI 線（主要）
└── Nonnegative tensor    ← 大型歷史倉庫，含多年的 NNI / NLI / LENT 疊代

兩者共用的內層 multilinear 解法（被包在 HNI 外層裡）
└── positive multilinear  ← 只有 Multi.m + 3 個 tensor generator；是 HNI 內層 Multi 的獨立打包
```

**關鍵觀察**：`2020_HNI_Heig` 和 `2020_HNI_Revised` 裡面**同時有 HNI.m 和 NNI.m**，因為那些 repo 是用來做演算法比較（Test_Heig2/3 是 HNI vs NNI vs NQZ 的 benchmark）。所以你在 HNI 資料夾裡看到 NNI 檔案，不是分類錯誤，是「同一份實驗腳本要比的多個對照組」。

---

## B. 各資料夾逐一盤點

### B.1 `2020_HNI_Heig`（23 `.m` 檔，約 1948 行）

**角色**：HNI 線的**開發工作空間**。含 HNI + NNI + NQZ + HSI + 多個 Multi 變體的實驗實作。

**主入口**：
- `main_Heig.m`（28 行）— 基本 demo：m=4, n=20 隨機張量、呼叫 `NNI_two`
- `Test_Heig.m`（84 行）— benchmark：HNI vs IHNI 在 m=4, n=30

**核心演算法檔**：
| 檔案 | 行數 | 演算法 |
|---|---|---|
| `HNI.m` | 91 | Newton 雙層疊代，精確 inner solve |
| `IHNI.m` | 91 | Inexact HNI，inner solve 用寬鬆 tolerance |
| `NNI.m` | 269 | 直接 Newton + step length 控制 |
| `NNI_two.m` | 273 | NNI 的微幅變體（跟 NNI.m 幾乎一樣） |
| `NQZ.m` | 171 | Power method baseline |
| `HSI.m` | 73 | Shifted-inverse 變體 |
| `Multi.m`, `Multi_inex.m`, `Multi_shift.m` | 77-87 | Inner linear solver 的三種版本 |
| `geM.m` | 54 | M-matrix 特化高斯消去（Alfa-Xue-Ye 演算法，用於數值穩定） |

**張量工具**：`tengen.m`, `tengen_4D.m`, `tengen_HD_weakly.m`, `ten2mat.m`

**結構問題**（⚠️ port 前要處理）：
- `Multi_shift.m` 第 20 行有 syntax error：`u = (1-theta/(m-1))*u + theta * v + ;` — 跑不起來
- `untitled.m`（8 行空的）和 `untitled2.m`（154 行實驗代碼）看起來是遺留檔
- `NNI.m` 和 `NNI_two.m` 幾乎相同，很可能是中途 fork 沒清理
- 所有 helper（`tpv`, `tenpow`, `sp_Jaco_Ax`）都 embed 在每個主檔裡面（不是獨立檔）→ 重複代碼嚴重

**中文註解**：沒有

---

### B.2 `2020_HNI_Revised`（21 `.m` 檔，約 1860 行）

**角色**：HNI 線的**修訂版**，主演算法從 `HNI.m` 改名為 `HONI.m`。代碼比 Heig 版更乾淨（沒有 untitled 檔、沒有語法錯誤），但多了比較用的 Test 腳本。

**主入口**：
- `main_Heig.m`（27 行）— ⚠️ 裡面呼叫的 `NNI_two` 這個資料夾裡**不存在**，會報錯（遺漏的檔案，可能跟 Heig 版共用的假設沒帶過來）
- `Test_Heig2.m`（77 行）— 完整 benchmark：HONI exact/inexact vs NNI vs NQZ，掃描 ω ∈ [0.01, 80]，m=4
- `Test_Heig3.m`（70 行）— 掃描 n ∈ [10, 300]，m=3
- `Test_Heig4.m`（78 行）— 單點測試 n=300

**核心演算法檔**：
| 檔案 | 行數 | 演算法 |
|---|---|---|
| `HONI.m` | 213 | **HNI 的改良版**，明確處理 exact/inexact 兩種 inner solver 選項 |
| `NNI.m` | 270 | Newton 直接法 |
| `NNI_ha.m` | 270 | 幾乎等於 `NNI.m`（待釐清哪個是現役） |
| `NQZ.m` | 173 | Power method |
| `Multi.m` | 101 | Inner solver |
| `Multi_res.m` | 103 | 同上，多了繪圖 |

**張量工具**：多了 `tengen_4Dplus.m`（帶 ε 參數的 4 階張量生成器），`tpv.m`/`tenpow.m` 在這版變成**獨立檔**（Heig 版是 embed），比較好 port。

**你的中文註解**（Big5 編碼，已用 iconv 抽出，如實保留）：

`Test_Heig.m` 第 34-37 行：
```
%%% 一般來說NNI相當穩定，不會隨著n,m 影響，但接下來的例子會讓NNI難以使用
%%% 測試 4D, w = 0.01~60, i=1:5
%%% 測試 3D, n=10~200, Q = 1*D + 1e-1*A;
%%% 給出內外迭代分別的收斂曲線
```

`Test_Heig3.m` 第 12-15 行：同上四行（幾乎一字不差複製在兩個 Test 檔）

**解讀**：你在提醒未來的自己「NNI 看起來很穩，但某些情境會壞掉」— 這是 port 時該納入的 edge case 考量。

---

### B.3 `2020_HNI_submitt`（3 `.m` 檔，約 360 行）

**角色**：論文投稿版的**極簡打包**。只有主 driver、HONI 本體、inner solver。

**檔案清單**：
| 檔案 | 行數 | 角色 |
|---|---|---|
| `main_Heig.m` | 38 | Demo：m=3, n=20 隨機張量、HONI exact + inexact 各跑一次 |
| `HONI.m` | 233 | H-eigenvalue 主算法 |
| `Multi.m` | 89 | Inner multilinear solver（Newton + 三等分 line search） |

**時間戳線索**：`HONI.m` 日期 2020/12，`main_Heig.m` 和 `Multi.m` 更新到 2021/06 — 符合「2020 投稿、2021 修訂發表」的典型時程。

**中文註解**：沒有（投稿版清掉了）

**這個資料夾的價值**：**`HONI.m` + `Multi.m` 就是 HNI 演算法的「canonical 最小實作」**。如果你要 port HNI，從這裡下手最簡單。

---

### B.4 `Nonnegative tensor`（13 `.m` at top level，但含子資料夾總共 **659 `.m` + 45 `.mat`**，213 MB）

⚠️ **這個資料夾的規模遠超你原本的描述**，不能整包處理。

**頂層結構（省略 tensorlab 等第三方）**：
```
Nonnegative tensor/
├── test.m, test2.m, test3.m             頂層測試腳本
├── matrixprod*.m, matrixshift*.m        頂層小工具（有 matrixshift_old.m 這種版本尾巴）
│
├── H_eigenpair_ver6/                    ⭐ 32 檔，NNI + LENT 最新實作
│   ├── NNI.m  (150 行)                  — NNI 核心
│   ├── LENT.m (140 行)                  — Power method baseline
│   └── main.m                           — demo driver
│
├── tensor_packege_ver4/                 ⭐ 20 檔，較穩定的打包版本
│   ├── LENT.m
│   └── eigen_solver/NLI_2015.m          — 2015 改良版 Newton
│
├── NLI_2015/                            4 檔，NLI 的最後一版獨立打包
├── Nonneagtive_tensodr_Lin/             7 檔（資料夾名拼錯 "tensodr"）
├── Non_tensor_Lin/                      14 檔
├── NT_arnoldi/                          19 檔，Arnoldi / Krylov-Schur 方法
│
├── tensor_packege_ver2 / ver3 / ver5 / ver5.1 / ver7.1 / ver7.3 / ver8.0
│                                        歷年版本演進，多半是實驗疊代
│
├── 2019_HNI_Heig/                       ⚠️ 這包也在裡面（HNI 的早期版？）
├── TenEig/, Z_eigenvalue*/, ALS/, BigData/
├── tensorlab_2016-03-28/                ⚠️ 第三方函式庫 snapshot（port 時要排除）
```

**核心演算法（抽出共通骨架）**：
- `NNI` — Newton-based 非負張量 largest eigenvalue
- `LENT` — Power method variant（論文裡是 Largest Eigenvector of Nonnegative Tensor）
- `NLI_2015` — Newton Linear Iteration 的 2015 版，應該是最成熟的
- Arnoldi / Krylov-Schur（`NT_arnoldi/`）— 進階 Krylov 子空間方法，相對獨立

**中文註解**：沒找到（主要英文註解）

**一個關鍵觀察**：`Nonnegative tensor/2019_HNI_Heig/` 存在 — 所以 HNI 這條線**最早期的版本**可能也藏在這個 folder 裡。如果你之後要做完整版本史溯源，這個路徑要進去。

---

### B.5 `positive multilinear`（5 `.m` 檔，20 KB）

**角色**：把 HNI 的**內層 solver（`Multi.m`）獨立打包**成一個可以單獨測試的小工具。不是完整的特徵值演算法。

**檔案清單**：
| 檔案 | 行數 | 角色 |
|---|---|---|
| `main_PM.m` | 51 | Demo：生成張量、呼叫 Multi、驗證 residual |
| `Multi.m` | 102 | **Newton + halving line search 解 `AA u^(m-1) = b`** |
| `tengen_4D.m` | 36 | 4 階張量生成（alternating sign） |
| `tengen_4Dplus.m` | 38 | 4 階張量生成（all-nonnegative，ε 參數） |
| `tengen_HD_weakly.m` | 39 | 高階弱對稱張量 |

**解讀**：這像是你把 HNI 的內層拆出來獨立驗證的實驗版（或是教學/示範用）。**它不能算是 NNI 線的實作**，雖然主題是 positive multilinear system。

**中文註解**：沒有

**一個待確認點**：`Multi.m` 第 34 行有一個 bare `u'*temp`（沒分號）— 應該是調試殘留，會 print 結果到 console，port 時要注意這是偵錯輸出還是刻意的收斂監控。

---

## C. 跨資料夾比較

### C.1 同名但內容可能有差異的檔案

| 檔名 | 出現在 | 差異評估 |
|---|---|---|
| `main_Heig.m` | Heig、Revised、submitt | 三個版本參數設定不同（m, n 範圍），Revised 版呼叫了不存在的 `NNI_two` 會壞。⚠️ |
| `Multi.m` | Heig、Revised、submitt、positive multilinear | 行數 77 → 101 → 89 → 102 — 至少四個版本，需要 diff 比對 |
| `HNI.m` / `HONI.m` | Heig 叫 HNI.m (91 行)、Revised 叫 HONI.m (213 行)、submitt 也叫 HONI.m (233 行) | **重大演進**：從 Heig 的 91 行成長到 submitt 的 233 行，演算法明顯擴充（很可能加了 inexact 分支、更多輸入檢查、convergence instrumentation） |
| `NNI.m` | Heig、Revised、都在 Nonnegative tensor 裡面各版本 | 多個演化版本，需要 case-by-case 比對 |
| `NQZ.m` | Heig (171 行)、Revised (173 行) | 幾乎相同 |
| `tengen_4D.m` | Heig、Revised、positive multilinear | 三份不同規模（82 / 35 / 36 行）— 看起來 Heig 版是完整版，Revised/pos multilinear 是精簡版 |
| `tengen_HD_weakly.m` | Heig、Revised、positive multilinear | 都是 38-39 行，可能相同或極相似 |
| `tenpow.m`, `tpv.m`, `ten2mat.m` | Revised 有獨立檔，Heig 和 submitt embed 在主檔裡 | Revised 版已經做了輕度模組化 |
| `geM.m` | Heig、Revised | 54 行，可能相同 |

**關鍵**：沒有任何兩個資料夾是另一個的嚴格超集。要決定 canonical 版本，需要對核心檔案做 diff。

### C.2 完全獨立的檔案

- `HSI.m`（Heig 獨有）— shifted-inverse 方法
- `Shiftedinverse.m`（Heig 獨有）
- `IHNI.m`（Heig 獨有）— inexact HNI
- `Doubleshift.m`（Heig 獨有）— double-shift 策略
- `Multi_shift.m`（Heig 獨有，但有語法錯誤）
- `Test_fsolve.m`（Heig 和 Revised 都有）— 用 `fsolve` 當對照組
- `HONI.m`（Revised 和 submitt 獨有）
- `tengen_4Dplus.m`（Revised 和 positive multilinear 獨有）
- `Test_Heig2/3/4.m`（Revised 獨有）
- `NNI_ha.m`（Revised 獨有）— 跟 NNI.m 幾乎一樣
- 整個 `Nonnegative tensor/NT_arnoldi/` 子包是獨立的 Krylov 方法

### C.3 三個 2020_HNI_* 版本差別（你的猜測驗證）

| | Heig（開發） | Revised（修訂） | submitt（投稿精簡） |
|---|---|---|---|
| `.m` 檔數 | 23 | 21 | 3 |
| 主算法檔名 | `HNI.m`（91 行） | `HONI.m`（213 行） | `HONI.m`（233 行） |
| `untitled*.m` | 有 2 個 | 無 | 無 |
| 語法錯誤 | `Multi_shift.m` 有 | 無 | 無 |
| 獨立 helper 檔 | 無（全 embed） | 有 `tpv.m`, `tenpow.m`, `ten2mat.m` | 無（回到 embed） |
| Benchmark 腳本 | `Test_Heig.m`（單一） | `Test_Heig2/3/4.m`（參數掃描） | 無（只有 `main_Heig.m`） |
| 中文註解 | 無 | 有（Big5） | 無 |
| 演算法時間戳 | - | - | `HONI.m` = 2020/12, `main`/`Multi` = 2021/06 |

**你的猜測正確**：`Heig` → `Revised` → `submitt` 是時間順序，最後一個是縮減版投稿。

---

## D. Toolbox 依賴檢查

**全部 5 個資料夾都沒有用到 MATLAB toolbox。** 使用的都是 base MATLAB 函式：

| 函式 | 使用頻率 | Python 對應 |
|---|---|---|
| `sparse`, `speye`, `spdiags` | 普遍 | `scipy.sparse.csr_matrix`, `scipy.sparse.eye`, `scipy.sparse.diags` |
| `kron` | 普遍（用來做 Kronecker 次方） | `numpy.kron` 或 `scipy.sparse.kron` |
| `reshape` | 普遍 | `numpy.reshape`（⚠️ order 預設不同，見 Section F） |
| `norm` | 普遍 | `numpy.linalg.norm` |
| 反斜線 `\`（線性系統） | 普遍 | `scipy.sparse.linalg.spsolve` |
| `fsolve` | 只有 `Test_fsolve.m` | `scipy.optimize.fsolve` — 但這是 reference/對照組，不是主算法 |

**零 Tensor Toolbox 依賴**：沒有任何檔案用 `tensor()`, `sptensor()`, `ttv()`, `ttm()`, `tenmat()`, `khatrirao()`, `cp_als()` 等。**所有張量運算都是手動用 sparse matrix + Kronecker product 實作的**。這對 port 有兩層意義：

1. **好消息**：不用處理「如何在 Python 找到 Tensor Toolbox 對應」這種麻煩
2. **壞消息**：既然是手搓 sparse Kronecker，**column-major 陷阱會直接命中**（詳見 Section F）

---

## E. Canonical Version 建議

### HNI 要 port 的話 → `2020_HNI_submitt`（強烈推薦）

**為什麼**：
- 只有 3 個檔：`main_Heig.m` + `HONI.m` + `Multi.m`
- 是你投稿版的精選，已經過你自己的「哪些要留」的篩選
- `HONI.m`（233 行）是演化最末期、最成熟的版本
- 沒有遺漏的依賴、沒有語法錯誤、沒有 untitled 檔

**可能的補充**：如果你發現投稿版少了某個有用的 test case，可以從 `2020_HNI_Revised/Test_Heig2.m` 或 `Test_Heig3.m` 單獨挑一兩個腳本補進來當 port 後的驗證資料。

### NNI 要 port 的話 → **請你再縮小範圍**

`Nonnegative tensor/` 有 659 個 `.m` 檔、7 個子版本、還嵌了第三方函式庫。這不是一個可以直接指向「這就是 canonical NNI」的狀態。建議你從下面三個候選裡挑一個：

1. **`Nonnegative tensor/H_eigenpair_ver6/`**（32 檔）— agent 認為是最新的 benchmark 套件，含 NNI + LENT + utility
2. **`Nonnegative tensor/tensor_packege_ver4/`**（20 檔）— 可能是「穩定版封裝」
3. **`Nonnegative tensor/NLI_2015/`**（4 檔）— 最小子集、命名含時間戳（2015），可能是你想突出的「NLI 2015 final」

我需要你告訴我：**你心中的 canonical NNI 是哪一個？** 或者我再派一個 agent 去比對這三個資料夾的內容、幫你找出「演算法最成熟 + 最少依賴」的那個。

### 「不需要 port」的資料夾

- **`2020_HNI_Heig`**：有語法錯誤、重複檔（NNI/NNI_two）、untitled 遺留。被 `Revised` 和 `submitt` 完全取代。可視為歷史參考。
- **`positive multilinear`**：只是 `Multi.m` + tensor generator 的獨立打包。當你 port 了 HNI（裡面就會有 Multi），這份就冗餘了。除非你想單獨 benchmark inner solver。

---

## F. Port 工作量預估

**基準尺（以之前的 gaussian_blur 為 1 單位）**：
- gaussian_blur = 1 個 MATLAB 函式（5 行） + 1 Python 函式 + 1 generate_reference（12 行） + 1 parity test（40 行）
- 實際花費：~30-60 分鐘主動工作（含對帳、除錯、寫 matlab_compat flag）

### HNI（`2020_HNI_submitt`, 3 檔, 360 行）

**估計**：**~10× gaussian_blur**（約 5-10 小時主動工作）

**分解**：
- `HONI.m` (233 行) → Python class/function + parity：**~6×**
- `Multi.m` (89 行) → Python 內層 solver + parity：**~2×**
- `main_Heig.m` (38 行) → Python demo + 整合 parity：**~1×**
- 加上重新打造 `matlab_compat` 模式 + tensor 生成 + 單元測試：**~1×**

**主要風險點**：
1. **Column-major 陷阱會直接命中**：HONI/Multi 內部大量使用 `kron`、`reshape`、`ten2mat`（張量展開）。MATLAB 的 column-major 和 numpy 預設的 C-order 在這裡會產生**系統性的元素錯位**，不是邊界效應那種局部差異。如果沒意識到就直接 port，parity test 會在**內部**就整個錯掉，而且錯得很一致（不是 ~1e-1，是量級就不對）。
2. **Sparse Jacobian 結構**：`sp_Jaco_Ax` 函式建構稀疏 Jacobian，要確認 `scipy.sparse` 對應操作的順序和儲存格式。
3. **M-matrix 數值穩定性**：`geM.m`（Alfa-Xue-Ye 演算法）是為了數值穩定性特別設計的，scipy 沒有對應現成實作，需要手動 port。

### NNI（候選資料夾，假設選 `H_eigenpair_ver6/` 或 `tensor_packege_ver4/`）

**估計**：**~10-15× gaussian_blur**（約 10-20 小時主動工作）

**分解**：
- 核心演算法（`NNI.m` + `LENT.m`，~300 行）：**~6×**
- 張量工具（`tenpow`, `ten2mat`, `tpv`, `tengen`, `tenrand`, ~300 行）：**~3×**
- Parity 測試 + `matlab_compat` flag：**~2×**
- 挑選/清理版本之間的歧異：**~2×**（這是 NNI 特有的額外工作，因為版本多）

### 陷阱 checklist（這次 port 時要主動檢查）

根據 memory 的三大陷阱 + 張量特有的 column-major，標記每個檔案的風險：

| 陷阱 | 會踩到的檔案 | 嚴重度 |
|---|---|---|
| **Column-major**（MATLAB 欄優先 vs numpy 列優先） | `tenpow.m`, `tpv.m`, `ten2mat.m`, `sp_Jaco_Ax`, `sp_tendiag`, 所有 `reshape` 呼叫 | **極高** — 系統性錯位，parity 會整個壞掉 |
| **邊界處理** | N/A — 這些演算法沒有邊界概念（不是 filter） | 無 |
| **核截斷寬度** | N/A | 無 |
| **索引起點** | 所有 `for i = 1:n` 迴圈、`A(i,j)` 存取、`find()` 輸出 | 中 — 最常踩到的是 `find`、切片邊界 |

**結論**：這次 port 最大的風險**不在 memory 記的三大陷阱裡**。應該把 **column-major** 正式加入 memory，當作第四個標準檢查項。我們完成 `gaussian_blur` 時就講到了要加，這次 port 開工前就應該入 memory。

---

## G. 我對下一步的建議（不替你決定，只提方案）

1. **先確認 NNI canonical version**：回答 Section E 的問題 — 要我派 agent 比對 `H_eigenpair_ver6` vs `tensor_packege_ver4` vs `NLI_2015`，還是你直接告訴我選哪個？

2. **先做 HNI 再做 NNI**（建議順序）：HNI 目標資料夾（`submitt`）已經明確、規模可控、文件時間戳最新。NNI 還需要收斂到一個 canonical version。先用 HNI 練手 + 打磨 workflow。

3. **開工前先把 column-major 陷阱寫進 memory**：這是這次盤點學到的重要教訓。寫進去後 memory 裡的 port checklist 會從「三大陷阱」變「四大陷阱」。

4. **先寫一份精簡的 MATLAB→Python 張量運算對照表**（`reshape`, `kron`, `find`, slicing 等）當作 port 開工前的 reference，避免每次重查。

以上僅是建議，等你確認下一步再動手。這份盤點報告的目的是讓你能下決策，不是我幫你下決策。
