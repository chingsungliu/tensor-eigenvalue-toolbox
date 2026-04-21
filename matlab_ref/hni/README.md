# HNI MATLAB Reference

## 來源

- **原始路徑**：`/Users/csliu/我的雲端硬碟/CSLiu/my-toolbox/MATLAB/Tensor Eigenvalue Problem/2020_HNI_submitt/`
- **匯入日期**：2026-04-21
- **Canonical 依據**：`matlab_ref/NNI_HNI_inventory.md` Section E 的結論 — 這是 2020/12 撰寫、2021/06 定稿的 HNI 投稿精簡版，3 個檔完整覆蓋演算法本體
- **Google Drive 原檔未動**，本資料夾是本地副本

## 檔案清單

| 檔案 | 行數 | 編碼 | 行尾 | Function signature | 角色 |
|---|---|---|---|---|---|
| `main_Heig.m` | 37 | ASCII | **CRLF** | (script) | Demo driver — `m=3, n=20` 隨機張量，跑 HONI 的 `exact` + `inexact` 兩種 linear solver 模式各一次 |
| `HONI.m` | 232 | ASCII | LF | `function varargout = HONI(varargin)` | **主演算法** — H-eigenvalue Newton Iteration，含外層 eigenvalue 更新；內嵌 5 個工具函式（`tpv`, `tenpow`, `sp_tendiag`, `ten2mat`, `idx_create`, `input_check`） |
| `Multi.m` | 88 | ASCII | LF | `function [u,nit,hal] = Multi(AA,b,m,tol)` | **內層 solver** — 解多線性系統 `AA·u^(m-1) = b`，Newton 法 + 三等分 line search；內嵌 3 個工具函式（`tpv`, `tenpow`, `sp_Jaco_Ax`） |

## 依賴關係

```
main_Heig.m  (script, 設定 m=3, n=20, x_0 = abs(ones(n,1)))
└── HONI.m  主演算法
    ├── input_check        (內嵌 arg parser)
    ├── ten2mat            (內嵌張量→矩陣展開)
    │   └── idx_create     (內嵌 index 字串構造)
    ├── sp_tendiag         (內嵌 sparse 對角張量)
    ├── tpv                (內嵌 tensor-vector product，定義於 HONI 內)
    │   └── tenpow         (內嵌 Kronecker power，定義於 HONI 內)
    └── Multi.m            內層 multilinear solver
        ├── tpv            (重新定義於 Multi 內)
        │   └── tenpow     (重新定義於 Multi 內)
        └── sp_Jaco_Ax     (內嵌 sparse Jacobian)
            └── tenpow     (再度呼叫 Multi 內版本)
```

**重要觀察**：`tpv` 和 `tenpow` 在 **HONI.m 和 Multi.m 各有一份獨立定義**（兩邊 embed 的版本功能一致）。Port 到 Python 時應**合併成共用 util**（例如 `hni_utils.py`），不要複製兩份。

## 中文註解盤點

**沒有**。3 個檔都是 ASCII 編碼（byte range 0-127），沒有任何非 ASCII 字元，因此不可能包含中文。所有註解與 docstring 都是英文。

**補充（非編碼層面）**：
- `Multi.m` line 47 有一個**拼字錯誤**的警告訊息：`'Can''t find a suitible step length...'` —「suitible」應為「suitable」。照抄進 Python 保留原樣 + 寫註解「MATLAB original typo preserved」。

---

## C4：四大陷阱預判

針對 memory `feedback_matlab_to_python_port.md` 的四大陷阱逐一分析：

### 陷阱 #1：邊界處理
**N/A** — 這不是 filter / convolution 演算法，沒有邊界處理概念。

### 陷阱 #2：濾波器 / 核截斷寬度
**N/A** — 沒有 filter / kernel 概念。

### 陷阱 #3：1-based vs 0-based indexing

| 位置 | 程式碼 | 風險 |
|---|---|---|
| `HONI.m:135` | `for i = 1:p-1` → `x_p = kron(x, x_p)` | 低 — 普通迴圈，Python 翻成 `range(p-1)` |
| `HONI.m:146-147` | `S = linspace(1,n^m,n); D(S) = d;` | ⚠️ **中** — `linspace(1,n^m,n)` 產生 1-based 的 n 個線性索引；在 numpy 要換成 `np.linspace(0, n**m-1, n).astype(int)` |
| `HONI.m:162-163` | `for i = 1:n; B(i,:) = eval(express);` | 高 — 迴圈 i 從 1 到 n，被 eval 塞進 MATLAB 字串表達式。Python 版必須**完全重寫**，不能用 eval |
| `HONI.m:170-175` | `if i < type ... elseif i > type` (idx_create) | ⚠️ **高** — 這是 1-based 的比較邏輯（`type` 是 MATLAB 語義下的「第幾個 mode」，從 1 數）。Python 重寫時要把 `type` 也統一成 0-based，或保留 1-based 語意但註明 |
| `Multi.m:39` | `while res(nit) - res(nit-1) > 0` | 低 — `res(nit-1)` 是前一次 iteration 的殘差，`-1` 是字面數學減一而非 index 調整 |
| `Multi.m:84` | `for i = 1:p` → `kron(tenpow(x,i-1), kron(I, tenpow(x,p-i)))` | ⚠️ **高** — 這是建 Jacobian 的核心迴圈，`i-1` 和 `p-i` 直接拿來當 `tenpow` 的 exponent。translate 時要**小心**：MATLAB 的 `i=1..p` 對應 Python 的 `for i in range(1, p+1)`（保留 1-based），或 `for i in range(p)` 配合 `tenpow(x, i)` 和 `tenpow(x, p-1-i)`（改 0-based）。**兩種都可，但必須在 port 的第一步就決定並一以貫之。** |

### 陷阱 #4：Column-major vs Row-major ⭐ **主要風險**

這是這份 port 最危險的陷阱，會直接影響 parity。

| 位置 | 程式碼 | 風險 | Python 對應要點 |
|---|---|---|---|
| `HONI.m:136` | `x_p = kron(x, x_p)` (在 tenpow 內) | **高** | MATLAB 的 `kron(A,B)` 是 block-by-block 的 Kronecker。numpy 的 `np.kron(A,B)` 定義相同 **但當 A/B 是向量時，column vs row 的取向影響結果 shape**。必須確認 x 在 Python 是 column vector `(n,1)` 還是 1-D `(n,)`，並和 MATLAB 一致。 |
| `HONI.m:148` | `D = reshape(D, n, n^(m-1))` (在 sp_tendiag 內) | ⭐ **極高** | MATLAB 的 `reshape` 是 **column-major**：從第一維開始填。numpy `reshape` 預設 C-order（row-major）。**必須用 `order='F'`**，不然整個 sparse 對角張量的索引位置會錯。 |
| `HONI.m:159` | `express = ['reshape(','A(',temp1,'i',temp2,')',',',int2str(1),',',int2str(n^(m-1)),');']` + `eval(express)` | ⭐ **極高** | 這個 `ten2mat` 完全依賴 MATLAB 的 column-major linear indexing。`A(:,:,...,i,:,...)` 的切片順序 + `reshape(..., 1, n^(m-1))` 的 flatten 順序**都是 column-major**。Python 版必須用 `np.moveaxis` + `reshape(order='F')` 或直接 `np.einsum` 重寫，**禁止**直接翻譯 eval string。 |
| `HONI.m:218` | `A = ten2mat(A, prodct_type)` | ⭐ **極高** | 繼承自 `ten2mat` 的風險。這是**整個 HONI 的第一步資料轉換**：如果 unfold 順序不一致，後面所有計算都會錯。 |
| `Multi.m:27` | `M = sp_Jaco_Ax(AA,u,m)/(m-1)` | 高 | Jacobian 建構，見下。 |
| `Multi.m:85` | `J = J + AA*kron( tenpow(x,i-1), kron(I, tenpow(x,p-i)) )` | ⭐ **極高** | Jacobian 公式 `F'(x) = Σ_{i=1..p} A * (x^{i-1} ⊗ I ⊗ x^{p-i})`。**Kronecker 順序直接對應張量 unfolding 的 mode 順序**；numpy 預設 row-major 會讓 ⊗ 順序反轉，Jacobian 整個錯亂。 |
| `Multi.m:28` | `v = M\b` | 低（非 layout） | MATLAB 的 backslash 對 sparse = `scipy.sparse.linalg.spsolve(M, b)`；只要 M 和 b 對了，解也對。 |
| `Multi.m:14-15` | `u = b/norm(b); b = b.^(m-1);` | 低 | Element-wise 運算，layout 無關。 |

**陷阱 #4 的結論**：**port 最大的工程量在把 `reshape(..., order='F')` 的慣例一以貫之**，以及重寫 `ten2mat` 和 `sp_Jaco_Ax`。一旦搞對，其他東西就順了。

### 額外風險：迭代收斂判準

Port iterative algorithm **不能只比最終輸出**，因為任何中間步驟的微小 layout 差異都會讓迭代路徑發散。

| 位置 | 程式碼 | parity 含意 |
|---|---|---|
| `HONI.m:46` | `while min(res) > tolerance && nit < maxit` | 外層收斂判準 |
| `HONI.m:49, 63` | 根據 `linear_solver` 分兩條分支 | **需要兩個 parity test**，exact 模式一個、inexact 模式一個。不能合併。 |
| `HONI.m:67` | `inner_tol = max(1e-10, min(res)*min(x)^(m-1)/nit)` | Inexact 模式的 adaptive tolerance，是 `nit` 的函數。Python 版必須完全一樣。 |
| `Multi.m:22` | `while min(res) > tol*(na*norm(u)+nb) && nit < 100` | 內層收斂判準 |
| `Multi.m:39-50` | 三等分 halving loop | 每次 halving 都會改變 `theta` 和 `u`，halving 次數 (`hit`) 會累加到 `hal(nit)`。**halving 次數必須兩邊一致**，否則迭代路徑發散。 |

**Parity 設計建議**：
1. **工具函式（`tenpow`, `tpv`, `sp_tendiag`, `ten2mat`, `sp_Jaco_Ax`）用 bit-level parity**（一次呼叫、確定性輸入、直接比輸出）。
2. **`Multi` 和 `HONI` 用「逐 iteration 比對」的 parity**：MATLAB reference 每次 iteration 都存下 `u`、`res`、`theta`、`hal`、`lambda_U`（用 struct array 或 cell array 存），Python 版每 iteration 比對一次。這樣**哪一步開始發散就能立刻看到**。

### 額外風險：隨機輸入

| 位置 | 程式碼 | parity 含意 |
|---|---|---|
| `main_Heig.m:10` | `Q = rand(n*ones(1,m))` | ⚠️ **隨機張量** — reference 必須 `rng(42)` 固定種子 + 把 `Q` 存進 `.mat`，Python 端讀同一份 |
| `HONI.m:225` | `option.initial_vector = rand(n,1)` (fallback when unset) | 中 — 在 demo 裡 `main_Heig.m` 明確給 `x_0 = abs(ones(n,1))` 蓋過這個 default；但 port 時要**強制傳入 initial vector**，不要依賴 fallback |

---

## C5：Port 順序建議

按依賴關係由底往上（leaf first），同時把 column-major 風險高的放前面（越早驗證越早抓到 bug）：

### 第 1 層：純工具，立即可 parity
| # | MATLAB 原檔 | Python 檔 | 工作量（以 gaussian_blur = 1×） |
|---|---|---|---|
| 1 | `HONI.m` line 129-139 `tenpow` | `hni_utils.py : tenpow(x, p)` | **0.5×** — Kronecker power，迴圈 `p-1` 次；需要對 `np.kron` + column vector 的 shape 取向達成約定 |
| 2 | `HONI.m` line 123-127 `tpv` | `hni_utils.py : tpv(AA, x, m)` | **0.3×** — 薄包裝，靠 tenpow |
| 3 | `HONI.m` line 142-149 `sp_tendiag` | `hni_utils.py : sp_tendiag(d, m)` | **0.5×** — 測試 `reshape(order='F')` 的第一個練習點 |

### 第 2 層：高風險核心工具
| # | MATLAB 原檔 | Python 檔 | 工作量 |
|---|---|---|---|
| 4 | `HONI.m` line 152-177 `ten2mat` + `idx_create` | `hni_utils.py : ten2mat(A, k)` | **⭐ 2×** — **最危險的一個**。必須徹底重寫（eval 的字串 hack 完全不能用）。改用 `np.moveaxis` + `reshape(order='F')`。**建議單獨做一輪 parity（小 tensor，m=3, n=4）完整驗證所有 k 值**。 |
| 5 | `Multi.m` line 79-87 `sp_Jaco_Ax` | `hni_utils.py : sp_Jaco_Ax(AA, x, m)` | **⭐ 1.5×** — Jacobian 的 Kronecker 鏈，column-major 敏感；parity 要對整個 sparse 矩陣 bit-level 比對 |

### 第 3 層：迭代 solver
| # | MATLAB 原檔 | Python 檔 | 工作量 |
|---|---|---|---|
| 6 | `Multi.m` 主體 | `python/multi.py : multi(AA, b, m, tol)` | **2.5×** — Newton 迭代 + halving；parity 需要**逐 iteration 比對**（MATLAB reference 要存每一步的 `u, res, theta, hal`） |
| 7 | `HONI.m` 主體（兩個分支） | `python/honi.py : honi(AA, ...)` | **3.5×** — 兩個 linear_solver 分支、外層收斂、兩種 output contract（nargout 1-6）；parity **兩條（exact / inexact）** |

### 第 4 層：整合與 demo
| # | MATLAB 原檔 | Python 檔 | 工作量 |
|---|---|---|---|
| 8 | `main_Heig.m` | `python/demo_honi.py` + 測試基礎 | **0.5×** — Demo 腳本 + 整體 smoke test |
| 9 | `matlab_ref/hni/generate_reference.m` | （MATLAB 端，手動跑） | **0.5×** — 跟 gaussian_blur 同套模板：rng(42) + 存 .mat（需存 Q、x_0、所有 iterate 中間值、最終 EV/EW）|
| 10 | `python/test_parity_hni.py` | — | **1×** — 整合 parity，驗證 exact + inexact 兩條、逐 iteration 對帳 |

### 總工作量估算

| 層級 | 工作量 |
|---|---|
| 第 1 層 (工具) | 1.3× |
| 第 2 層 (高風險核心) | 3.5× |
| 第 3 層 (迭代 solver) | 6.0× |
| 第 4 層 (整合) | 2.0× |
| **合計** | **~13× gaussian_blur** |

比之前 inventory 裡預估的「~10×」略高，主要差在 **把「逐 iteration parity」這一設計納入後**，`Multi` 和 `HONI` 的 parity 測試複雜度比單點比對高出一截。但這是必要的 — 沒有逐步比對，迭代路徑一旦發散就很難 debug。

---

## Port 進度

本 port 依 C5 的 4 層順序進行。每個函式 port 成功（bit-level / machine-epsilon parity）後更新下表。

| 層級 | 函式 | Python 位置 | Parity test 檔 | Max error | 狀態 |
|---|---|---|---|---|---|
| 1 | `tenpow` | `tensor_utils.py::tenpow` | `test_tenpow_parity.py` | **0** | ✓ 完成 |
| 1 | `tpv` | `tensor_utils.py::tpv` | `test_tpv_parity.py` | ~1e-16 | ✓ 完成 |
| 1 | `sp_tendiag` | `tensor_utils.py::sp_tendiag` | `test_sp_tendiag_parity.py` | **0** | ✓ 完成 |
| 2 | `ten2mat` | `tensor_utils.py::ten2mat` | `test_ten2mat_parity.py` | **0** | ✓ 完成 ⭐ column-major 主檢查點 cleared |
| 2 | `sp_Jaco_Ax` | — | — | — | ☐ 下一步 |
| 3 | `Multi` | — | — | — | ☐ 待開始 |
| 3 | `HONI` | — | — | — | ☐ 待開始 |
| 4 | `main_Heig` demo | — | — | — | ☐ 待開始 |

**層 1 完成後總結**：3 個純工具函式全部 parity 通過。`tenpow` 和 `sp_tendiag` 是 bit-identical（無浮點運算）；`tpv` 是 machine epsilon（矩陣-向量浮點加總）。

**層 2 第一步（ten2mat）完成後總結**：column-major 主檢查點安全通過 — 對 (3,3,3)、(4,4,4)、(2,2,2,2,2) 三個 shape 全部 `max_err = 0`。`np.moveaxis + reshape(order='F')` 的策略被證明是正確對應 MATLAB `eval(express)` 動態組裝的做法。剩下 `sp_Jaco_Ax` 是 layer 2 最後一關。

---

## 為什麼 NNI 不在這個資料夾

**HNI**（H-eigenvalue Newton Iteration）和 **NNI**（Nonnegative Newton Iteration）是**並列獨立的演算法**，沒有呼叫關係：

- HNI 解「求非負張量最大 H-特徵值」用**外層 eigenvalue 更新 + 內層 Newton 解多線性系統**的雙層結構
- NNI 直接對 `F(x) = Ax^(m-1) - λ·x^(m-1)` 做 Jacobian-based single-loop Newton

`2020_HNI_submitt/` 裡的 3 個檔（`main_Heig.m`, `HONI.m`, `Multi.m`）**全部是 HNI 實作**，不含 NNI。

NNI 將來會從 `Nonnegative tensor/tensor_packege_ver8.0/`（或你最終選的 canonical 版本）獨立 port，會放在 `matlab_ref/nni/`。兩邊的 Python 實作會分別在 `python/honi.py` 和 `python/nni.py`，**不共用主演算法代碼**，但可能共用 `python/hni_utils.py` 裡的 `tenpow` / `tpv` / `ten2mat` 等張量工具（這些在兩條線是同樣的數學運算）。

---

## Port 開工前要做的準備（不在這次 Phase C 的範圍內，但先記下來）

當你指示進入 Phase D（Port 執行）時：

1. 先單獨寫 `hni_utils.py` 裡的 `tenpow` → 跑 bit-level parity（需要 MATLAB 跑一個 `generate_tenpow_ref.m` 腳本）
2. 再寫 `ten2mat` → 這是最危險的一步，**在這一步就用足夠多的 tensor shape 做 parity**（m=3, n=4；m=4, n=3；m=4, n=5），確認 column-major 完全對齊
3. 之後再按上面第 1-4 層的順序推進
4. 每一層都有自己的 parity test；**不要把所有東西寫完才一次比對**，那會讓 debug 變成惡夢
