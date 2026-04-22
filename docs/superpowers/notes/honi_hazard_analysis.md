# HONI.m — Port 前 Hazard Analysis

**日期**：2026-04-22（Day 2 晚，Session 3 階段 A）
**對象**：`matlab_ref/hni/HONI.m`（232 行；主體 line 33-119，其餘是 local helpers + input_check）
**目的**：在動手 port 之前釐清外層 eigenvalue iteration 結構、exact/inexact 兩個分支、所有 port 陷阱、per-iteration parity 欄位。下一步（階段 B）才寫 Python 實作 + 比對。

**前置**：Multi port 已完成、Q5 parity 通過到 machine epsilon（Day 2 下午）。5 個 tensor 工具（tenpow/tpv/sp_tendiag/ten2mat/sp_Jaco_Ax）都已 port + parity 通過（Day 1）。

---

## 一、函式簽章

MATLAB 原型（line 1）：
```matlab
function varargout = HONI( varargin )
```

**變長介面**：透過 `input_check`（line 182-231）把 varargin 拆成 6 個固定變數、透過 `switch nargout` 把 varargout 組出來。Python port 要把這個結構打平成明確 parameters。

### 輸入（varargin，最多 4 個）

| 位置 | 名稱 | 預設 | 說明 |
|---|---|---|---|
| 1 | `A` | — | m-order n-dim tensor（`A.ndim == m`）**或** unfolding matrix `(n, n^(m-1))`。`input_check` 用 `length(size(A))` 判斷、自動 dispatch |
| 2 | `plot_res` | `0` | `1` 則在演算法結束後 `loglog(1:nit, res(1:nit))` 畫圖；本 port **建議全砍**（或做成 optional matplotlib）|
| 3 | `tolerance` | `1e-12` | 外層 eigenvalue iteration 停止門檻 |
| 4 | `option` | `struct('linear_solver','exact','initial_vector',[],'maxit',100)` | 三欄 struct：solver 類型、初始向量、最大 iter 數 |

**option 三欄**：
- `linear_solver`: `'exact'`（預設）或 `'inexact'` — 切換內層 Multi 呼叫的 tolerance 策略 + 外層更新公式（見 §二、§五陷阱 #6）
- `initial_vector`: `[]`（預設 → `rand(n,1)`）或使用者指定的 column vector
- `maxit`: 外層 iter 硬上限（預設 100，語意上比 Multi 的 100 buffer 大小更寬鬆）

### 輸出（varargout，1-6 個，由 `switch nargout` 分派）

| nargout | 返回順序 |
|---|---|
| 0 或 1 | `lambda`（最大特徵值） |
| 2 | `[x, lambda]`（特徵向量 + 特徵值） |
| 3 | `[x, lambda, res]`（加收斂曲線、length = outer nit） |
| 4 | `[x, lambda, res, nit]`（加外層迭代次數） |
| 5 | `[x, lambda, res, nit, innit]`（加內層 Newton 總次數） |
| 6 | `[x, lambda, res, nit, innit, hal]`（加 halving 總次數） |

**Python port 建議**：全部固定輸出 `(x, lambda_, res, nit, innit, hal)` 即 nargout=6 版本、不做 varargout 分派。`lambda` 是 Python keyword、用 `lambda_` 或 `mu` 命名。

---

## 二、迭代主結構

### 初始化（line 33-44）

```
[AA, m, n, plot_res, tolerance, option] = input_check(...)
                                          ├─ 若 A 是 tensor (m-D) → ten2mat(A, 1)
                                          └─ 若 A 是 matrix (2-D) → 原樣
x         = option.initial_vector                        line 35
x         = x / norm(x)                                  line 36   normalize
temp      = tpv(AA, x, m) ./ (x.^(m-1))                  line 37   element-wise local eigenvalue
lambda_U  = max(temp)                                    line 38   上界初估
II        = sp_tendiag(ones(n,1), m)                     line 39   identity tensor unfolding
maxit     = option.maxit                                 line 40
res       = ones(maxit, 1)                               line 41   pre-alloc, pre-fill = 1
nit       = 1                                            line 42
res(1)    = |max(temp) - min(temp)| / lambda_U           line 43
hal       = 0                                            line 44
innit     = 0                                            line 44
```

**關鍵細節**：
- `temp = tpv(AA, x, m) ./ (x.^(m-1))` 逐項是「A 對 x 作用後、除以 x^(m-1) 的**每個分量的局部 eigenvalue 估計**」。收斂時所有分量相等 → max-min = 0 → res → 0。
- `res` pre-fill 值為 1（對比 Multi 的 `na+nb`）。理由：`res(1) = |max-min|/lambda_U` 是**相對差**、恆 ≤ 1，所以 pre-fill 1 不會被後續實際值超過。可在 port 加 assert 防呆。
- `nit = 1` 是 MATLAB 1-based init（同 Multi）— Python 0-based 起算要改 `nit = 0`。
- `II` 是 identity tensor 的 mode-1 unfolding，用在 `lambda_U*II - AA` 當 Multi 的係數（shift-invert 結構）。

### 外層 eigenvalue iteration（line 46-79）

```
while min(res) > tolerance && nit < maxit
    nit = nit + 1
    
    if option.linear_solver == 'exact':
        ─ EXACT branch ──────────────────────────────────────────
        inner_tol  = 1e-10                                    line 53   STATIC
        [y, chit, hal_inn] = Multi(lambda_U*II - AA, x, m, inner_tol)   line 54
        hal       = hal + sum(hal_inn)                        line 55
        innit     = innit + chit                              line 56
        temp      = x ./ y                                    line 58   ⚠️ ratio, not tpv
        lambda_U  = lambda_U - min(temp)^(m-1)                line 59   INCREMENTAL
        res(nit)  = |max(temp)^(m-1) - min(temp)^(m-1)| / lambda_U      line 60
        x         = y / norm(y)                               line 61   normalize AFTER lambda update
    
    elif option.linear_solver == 'inexact':
        ─ INEXACT branch ───────────────────────────────────────
        inner_tol = max(1e-10, min(res) * min(x)^(m-1) / nit) line 67   DYNAMIC
        [y, chit, hal_inn] = Multi(lambda_U*II - AA, x, m, inner_tol)   line 68
        hal       = hal + sum(hal_inn)                        line 69
        innit     = innit + chit                              line 70
        x         = y / norm(y)                               line 72   normalize FIRST
        temp      = tpv(AA, x, m) ./ (x.^(m-1))               line 73   ⚠️ tpv, not ratio
        lambda_U  = max(temp)                                 line 74   RECOMPUTE from new x
        res(nit)  = |max(temp) - min(temp)| / lambda_U        line 75
end
```

### Exact vs Inexact 差異矩陣（port 時逐欄對照）

| 差異點 | Exact | Inexact |
|---|---|---|
| inner_tol | `1e-10` 寫死 | `max(1e-10, min(res)*min(x)^(m-1)/nit)` |
| temp 定義 | `x ./ y`（新舊 x 的**比值**）| `tpv(AA, x_new, m) ./ (x_new.^(m-1))`（對新 x 重算 tpv）|
| lambda_U 更新 | `lambda_U -= min(temp)^(m-1)`（**增量**、單調遞減）| `lambda_U = max(temp)`（**整個重算**）|
| res 公式 | `\|max^(m-1) - min^(m-1)\|/lambda_U` | `\|max - min\|/lambda_U` |
| x 正規化時機 | 先更新 lambda_U、**最後**再 `x = y/\|y\|` | **先** `x = y/\|y\|`、再用新 x 算 temp/lambda_U |

**這兩個分支數學上求解同一個問題、但迭代路徑完全不同**。port 要兩個都做、parity 要兩個 case 分別驗。

### 收斂判準

| 位置 | 條件 | 語意 |
|---|---|---|
| 外層 while（line 46） | `min(res) > tolerance` | 最小相對殘差未達門檻 |
| 外層 while（line 46） | `nit < maxit` | 硬上限（預設 100）|
| Multi 內層（已 port） | `min(res) > tol*(na*\|u\|+nb)` / `nit < 100` / `theta < 1e-14` | 見 Multi.m line 22, 46 |

**不變量**：`x` 保持單位範數（每次 normalize）；`lambda_U` 在 exact 分支是 monotone non-increasing 的上界（前提：`min(temp) >= 0`，否則其實會變大）。

### 輸出分派（line 81-117）

`switch nargout` 把內部 `x / lambda_U / res(1:nit) / nit / innit / hal` 攤成 1-6 個輸出。Port 忽略、固定返回全部。

---

## 三、呼叫了哪些函式 — 已 port 狀態

| MATLAB 呼叫（HONI.m 內） | MATLAB 定義位置 | 已 port 到 Python | 狀態 |
|---|---|---|---|
| `Multi(AA, b, m, tol)` | `matlab_ref/hni/Multi.m`（外部）| `tensor_utils.py::multi` | ✅ Day 2 完成，Q5 parity 通過到 machine epsilon |
| `tpv(AA, x, m)` | HONI.m line 123-127（local）| `tensor_utils.py::tpv` | ✅ Layer 1 完成 |
| `tenpow(x, p)`（via `tpv`）| HONI.m line 129-139（local）| `tensor_utils.py::tenpow` | ✅ Layer 1 完成 |
| `sp_tendiag(d, m)` | HONI.m line 142-149（local）| `tensor_utils.py::sp_tendiag` | ✅ Layer 1 完成 |
| `ten2mat(A, k)`（只在 `input_check` 用到）| HONI.m line 152-165（local）| `tensor_utils.py::ten2mat` | ✅ Layer 2 完成（column-major 主檢查點）|
| `idx_create(n, type)`（`ten2mat` 內部用）| HONI.m line 167-177（local）| N/A — Python port 用 `np.moveaxis` 取代 `eval(express)`，不需要 idx_create | — |
| `input_check(...)` | HONI.m line 182-231（local）| **未 port，Python port 本身會用 keyword args 取代**，不需要獨立函式 | 新寫 |
| `max`, `min`, `abs`, `norm` | MATLAB built-in | `np.max / np.min / np.abs / np.linalg.norm` | — |
| `strcmp` | MATLAB built-in | Python `==` 字串比較 | — |
| `ones(n,1)`, `rand(n,1)` | MATLAB built-in | `np.ones(n) / np.random.default_rng(...).random(n)` | — |
| `sparse × dense - sparse × dense`（即 `lambda_U*II - AA`）| MATLAB 混合運算 | scipy.sparse 支援 `scalar * sparse - sparse`，但要注意輸出格式 | **新注意點**，見 §五陷阱 #5.8 |
| `loglog(...)` | MATLAB plot | matplotlib（optional、建議 port 不實作）| — |

**重點**：HONI 所有 dependency **全部已 port**。階段 B 只剩 HONI 自身外層結構 + input_check 邏輯改寫成 Python。**不會引入新的 Layer 1/2 陷阱**。

---

## 四、五大陷阱逐行分析

### 陷阱 #1：邊界處理
**N/A** — HONI 不是 filter / convolution、無邊界概念。

### 陷阱 #2：濾波器 / 核截斷寬度
**N/A** — 無 kernel。

### 陷阱 #3：1-based vs 0-based indexing

| 行號 | MATLAB 原碼 | 是否觸發 | 處置 |
|---|---|---|---|
| 41 | `res = ones(maxit,1); nit=1; res(1) = ...` | ⚠️ **是**，同 Multi 的初始化 pattern | Python 用 `nit = 0`、`res = np.ones(maxit)`、`res[0] = ...` |
| 46 | `while min(res) > ... && nit < maxit` | 低 | `nit < maxit - 1` Python 0-based（MATLAB nit < 100 ≡ Python nit < 99；見 Multi 的結論） |
| 47 | `nit = nit + 1` | 低 | 同步遞增 |
| 60, 75 | `res(nit, 1) = ...` | 中 | MATLAB 2-D index on column vector → Python 1-D `res[nit]` |
| 67 | `inner_tol = max(1e-10, min(res)*min(x)^(m-1)/nit)` | ⭐ **中-高**。這裡 `nit` 是**字面 MATLAB 1-based 值**（nit 在第一次 inexact iter 時是 2、Python 0-based nit 在此時是 1）| **兩種做法選一**：<br>(a) Python 保留 1-based 變數傳入公式：用 `nit_mat = nit + 1` 代入<br>(b) 公式就寫 `/(nit + 1)` 明示 offset |
| 82-117 | `varargout` / `res(1:nit)` | 中 | Python 寫 `res[:nit + 1]`（MATLAB 1-based `1:nit` ≡ Python 0-based `0:nit+1`）|

**結論**：同 Multi 的決定，Python 全用 0-based、`nit = 0` 起算、第 0 格存初始化狀態。**唯一要特別小心的是 line 67 的 `/nit`**：這個 nit 是字面數值（不只是 index），Python 用 `/(nit + 1)` 才能與 MATLAB bit-identical。

### 陷阱 #4：Column-major vs Row-major

**HONI 主體不觸發**。Line 33-119 沒有 `reshape`、沒有 `flatten`、沒有 linear index → n-dim unravel 的操作。全部是 matrix/vector 逐項運算。

**間接觸發**（已處理）：
- `input_check` 在 `A.ndim >= 3` 時呼叫 `ten2mat(A, 1)` — 這是 mode-1 unfolding，內部有 `reshape(..., order='F')`。本 port 會用 Python port 的 `ten2mat`（已 parity 通過、column-major 守門員測試 `test_ten2mat_basic` 在 Day 1 已存）。
- `sp_tendiag(ones(n,1), m)` 產生 II — 已 port、reshape 已 column-major 處理。

**結論**：`ten2mat` 和 `sp_tendiag` 已在 Layer 2 排好雷，**HONI 自身 port 不需要再注意 column-major**。

### 陷阱 #5：迭代控制流 + sparse 數值（Multi 陷阱 #5 在 HONI 的延伸）

| 子項 | 行號 | 內容 | 風險 |
|---|---|---|---|
| **5.1 `lambda_U*II - AA` 的 sparse/dense 分派** | 54, 68 | `II` 是 sparse csr（`sp_tendiag` 返回）；`AA` 可能 sparse 或 dense | ⭐ **中**。MATLAB 自動處理混合運算；Python 端 `scalar * csr_matrix - ndarray` 要注意：<br>- 若 `AA` 是 sparse：`lambda_U * II - AA` 都是 sparse，結果 sparse。OK<br>- 若 `AA` 是 dense：`lambda_U * II - AA`，`scalar * csr - dense` 會將 csr 轉 dense（scipy 可能 warning）。建議 port 頭 **先統一 AA 成 sparse 或 dense**（依輸入型別）、再做減法 |
| **5.2 `x ./ y` element-wise（exact 分支 line 58）** | 58 | Python 1-D `x / y`。若 `y_i == 0` blow up。演算法假設 y > 0（Multi 的正解承諾），但若 Multi 收斂不好 y 有 0 或負項就 nan/inf | 中 |
| **5.3 `tpv(AA,x,m) ./ (x.^(m-1))`（line 37, 73）** | 37, 73 | element-wise。若 `x_i == 0` blow up | 低-中（`x = x/\|x\|` normalize 過，|x_i| 小但通常 > 0） |
| **5.4 `lambda_U = lambda_U - min(temp)^(m-1)`（exact line 59）** | 59 | **符號**：`min(temp)` 可能負（early iters）。`^(m-1)` 為偶數時 → 非負、`lambda_U` 單調遞減。m 為奇數時 → `^(m-1)` 保符號、`lambda_U` 可能上下擺動 | 中。Python `x ** (m - 1)` 保持 sign（與 MATLAB 一致）。但要確認 m 為 int（不是 float），否則負數取非整數次方會產生 complex。**Port 要 `m = int(m)` assert** |
| **5.5 `inner_tol` 公式（inexact line 67）** | 67 | `max(1e-10, min(res)*min(x)^(m-1)/nit)` 裡的 `nit` 是字面 1-based，**見陷阱 #3 的處置** | ⭐ **高**（若忘記 +1 offset、每個 inexact iter 的 inner_tol 都不對、Multi 收斂到不同位置、parity 從第一個 outer iter 就發散）|
| **5.6 Pre-alloc `res = ones(maxit,1)` 策略** | 41 | 同 Multi 的 pre-fill，但 maxit 預設 100，語意「最多 100 外層 iter」 | 低。Python `np.ones(maxit)` 等價。加 assert `res[:nit+1] <= 1.0 + 1e-10` 防呆 |
| **5.7 `innit += chit` 的 nit-offset parity** | 56, 70 | MATLAB `chit` = Multi 返回的 nit（**MATLAB 1-based**，值 = K+1 where K = Multi 外層 Newton iter 數）。Python port 的 Multi 返回 `nit_py = K`。若 Python HONI 寫 `innit += nit_py` 則 innit 跟 MATLAB 差 `outer_nit * 1`（每 outer iter 少數 1） | ⭐ **高**。**Port 要寫** `innit += chit_py + 1` 或 `innit += int(chit_py) + 1`，這樣 innit scalar 跟 MATLAB bit-identical |
| **5.8 `hal += sum(hal_inn)`** | 55, 69 | `hal_inn` 是 Multi 返回的 pre-alloc (100,) 向量、未用 slot 為 0。`sum` over full vector OK | 低。Python 完全一樣 |

**陷阱 #5 結論（HONI port checklist）**：

- [ ] `AA` 和 `II` 混合運算前先確認 sparse/dense 型別（`issparse(AA)` 分派）
- [ ] `m = int(m)` 開頭 assert
- [ ] `inner_tol` inexact 公式用 `(nit + 1)` 取代字面 `nit`
- [ ] `innit += chit_py + 1`（明寫 +1 offset、docstring 註明）
- [ ] `x = x / np.linalg.norm(x)` normalize 後 assert `np.all(x != 0)` 或類似
- [ ] `res = np.ones(maxit)`、assert `res[:nit+1] <= 1.0 + 1e-10`
- [ ] 陷阱 #3 的所有 indexing offset 逐項對照

---

## 五、HONI 專屬新陷阱 #6：exact / inexact 分支 + polymorphic input

### 6.1 exact / inexact 分支 — 兩個獨立的演算法路徑

**不是**「精度差異」、**是**「數學公式差異」（見 §二 差異矩陣）。Port 風險：

- **共用程式不能照 MATLAB 的 `if strcmp / elseif strcmp` 原樣抄**（容易複製錯欄位）。Python 建議**拆成兩個內部函式** `_honi_exact_step(...)` / `_honi_inexact_step(...)`、外層 dispatch。這樣每個 step 的公式逐字對照 MATLAB 來寫、parity 比較容易定位。
- **Test case 要兩個分支各跑一次**、`multi_reference_honi.mat` 要包兩個 case 的 history（或分兩個 .mat）。
- **Parity tolerance**：exact 分支預期 bit-identical 或 machine epsilon（同 Multi Q5）。inexact 分支因 `inner_tol` 動態調整、內層 Multi 解的精度不同，**可能在某個 iter 因 Multi 內的 halving 選擇不同而發散 1-2 ulp**。但因兩邊演算法一字對一字，預期仍在 machine epsilon 等級。

### 6.2 polymorphic input A（tensor vs unfolding）

MATLAB `input_check` line 211-219：
```matlab
if length(size(A)) == 2         % matrix
    n = size(A,1)
    m = round(log(size(A,1)*size(A,2)) / log(n))   % solve n^m = numel(A)
elseif length(size(A)) >= 2     % tensor
    n = size(A,1)
    m = length(size(A))
    A = ten2mat(A, 1)
end
```

**兩個注意點**：
1. **`length(size(A)) == 2` 分支算 m**：要 `numel(A) = n^m`。對標準 unfolding `(n, n^(m-1))`，`n * n^(m-1) = n^m` 成立、m 算出來正確。但若使用者傳非標準 shape（例如 `(n, n)` 當 matrix），算出 m=2 — 合理、HONI 也可解矩陣特徵值。
2. **`length(size(A)) >= 2` 分支**：MATLAB `size(tensor)` 會給 `[n, n, ..., n]`，`length` = m。Python `A.ndim` 等價。Python port：
   ```python
   if A.ndim == 2:
       n = A.shape[0]
       m = int(round(np.log(A.shape[0] * A.shape[1]) / np.log(n)))
       AA = A
   elif A.ndim >= 3:
       n = A.shape[0]
       m = A.ndim
       AA = ten2mat(A, k=0)  # mode-1 unfolding (Python 0-based)
   ```
   **注意**：MATLAB `ten2mat(A, 1)` 的 `1` 是 1-based mode index；Python 0-based `ten2mat(A, k=0)`。**必須是 k=0、不是 k=1**。

### 6.3 `option.initial_vector = []` default 的 rand 問題

MATLAB line 224-226：若 `option.initial_vector` 為空，`option.initial_vector = rand(n, 1)`。這是個**不確定性**：每次 call 結果不同。

對 parity：**MATLAB reference 要顯式 `rng(42); init = rand(n,1)` 後把 init 存進 .mat，Python parity test load 這個 init、確保兩邊用同一個向量**。不能依賴 Python 端自己 rand。

### 6.4 `plot_res = 1` 的副作用

MATLAB line 82-86 在演算法結束後 `loglog(...)` 畫圖。Python port 建議：
- **預設 `plot_res=False`**、完全不畫
- 若使用者明確要圖、才 import matplotlib 並在函式尾端 plot

這不影響 parity（plot 不改動資料）、但 port 要決定是否支援這功能。

---

## 六、Per-iteration parity — 要存的 history 欄位

依照 Multi port 的 history 設計（Day 2 已驗證的 5 欄 pattern），HONI 有**兩層 history**：外層（HONI 自身每 outer iter）+ 內層（每次呼叫 Multi 產生的 Multi history）。

### 外層 history（長度 `outer_nit + 1`，index 0 存初始化狀態）

| 欄位 | 形狀 | 意義 | `[0]` 位置存什麼 |
|---|---|---|---|
| `x_history` | `(n, outer_nit + 1)` | 每 outer iter 結束後的 normalized x | `init_x / \|init_x\|`（line 36 結果）|
| `lambda_history` | `(outer_nit + 1,)` | 每 outer iter 結束後的 lambda_U | `max(tpv(AA, x_init, m) ./ x_init^(m-1))`（line 38）|
| `res_history` | `(outer_nit + 1,)` | 每 outer iter 的相對殘差 | `\|max(temp) - min(temp)\| / lambda_U`（line 43）|
| `y_history` | `(n, outer_nit + 1)` | 每 outer iter Multi 返回的 y（未 normalize）| `NaN` vector（iter 0 無 Multi call）|
| `inner_tol_history` | `(outer_nit + 1,)` | 每 outer iter 使用的 inner_tol | `NaN`（iter 0 無 Multi call）|
| `chit_history` | `(outer_nit + 1,)` | 每 outer iter 的 Multi inner nit（MATLAB 1-based、即 `nit_py + 1`） | `0`（iter 0 無 Multi call）|
| `hal_per_outer_history` | `(outer_nit + 1,)` | 每 outer iter 的 halving 總次數 `sum(hal_inn)` | `0` |
| `innit_history`（accumulator） | `(outer_nit + 1,)` | 累積 innit | `0` |
| `hal_accum_history`（accumulator）| `(outer_nit + 1,)` | 累積 hal | `0` |

**為什麼存 accumulator history 而不只是最後的 scalar**：parity 失敗時可以看到是哪個 outer iter 開始 innit 或 hal 累加不一致。

### 內層 history（每次 Multi call 各一份、預設**不存**）

**建議預設不存**、只存外層 summary（chit_history / hal_per_outer_history / y_history）。理由：
- Multi 在 Day 2 已做 Q5 parity、單元級別 bit-identical 驗證過
- HONI 每外層 iter 都呼叫一次 Multi、全部 inner history 會膨脹到 `(outer_nit × n × inner_nit)`、大 case 爆掉
- 若某個 outer iter 的 y 對不上、可以**單獨重跑那個 outer iter** 並 `record_history=True` 呼叫 Multi、拿到 inner history 去定位

**Option**（若使用者要）：加 `record_inner_history=False` flag，True 時存 list of 每次 Multi 的 history dict。

### 長度慣例

所有 history 長度 = `outer_nit_py + 1` = `outer_nit_mat`（因 nit_py = nit_mat - 1、Python 多存 index 0 = init slot）。MATLAB reference 腳本的 pre-alloc buffer 大小至少 `maxit + 1`、結尾截斷到 `1:nit` （MATLAB）= `0:nit+1` （Python）。

### MATLAB reference 腳本草案（階段 B 會寫、此處只記 spec）

- 檔名：`matlab_ref/hni/generate_honi_reference.m`
- rng(42) 固定 seed、跑兩個 test case：`honi_reference_exact.mat` 和 `honi_reference_inexact.mat`
- 每個 .mat 包含：
  - 輸入：`AA`、`m`、`n`、`tol`、`maxit`、`initial_vector`、`linear_solver`
  - scalar 輸出：`lambda`、`nit`、`innit`、`hal`
  - 外層 history：`x_history`、`lambda_history`、`res_history`、`y_history`、`inner_tol_history`、`chit_history`、`hal_per_outer_history`、`innit_history`、`hal_accum_history`
- 改寫 `HONI.m` 的副本 `HONI_with_history.m`（不動 canonical）、加 history outputs、跑完 save

---

## 七、Port 前 Open Questions（需要使用者決策）

1. **分支 port 策略**：兩個分支同時 port（一個 Python 函式 + `linear_solver` kwarg）、還是拆兩個獨立函式 `honi_exact` / `honi_inexact`？
   - 建議：**一個函式 + kwarg**，內部 dispatch 到 `_exact_step` / `_inexact_step` 私有函式。理由：對外 API 跟 MATLAB 同構（`option.linear_solver`）、對內程式拆乾淨避免欄位混淆。

2. **varargout / nargout 要不要模擬**：
   - 建議：**固定返回 `(x, lambda_, res, nit, innit, hal)` 全 6 項**，不做 nargout 分派。理由：Python 沒有 nargout 概念、callers 自己 unpack 需要的；多出的輸出對 parity 和 debug 都有用。
   - `lambda` 是 Python keyword、用 `lambda_` 或 `mu` — 建議 `mu`（跟 MATLAB docstring 裡 `mu = HONI(A)` 一致）。

3. **Inner Multi history 存法**：
   - (A) 預設不存、只存外層 summary（chit/hal_per_outer/y）
   - (B) 預設存全部內層 history、dict-of-lists
   - (C) `record_inner_history=False` flag、使用者按需開啟
   - 建議：**(A) 預設不存 + (C) flag 按需**。Multi 單元 parity 已驗證、HONI 出錯 99% 在 HONI 自己的更新公式、不在 Multi；若萬一 Multi 在 HONI 情境下出錯、可用 flag 開啟 inner history 重跑目標 outer iter 定位。

4. **test case 策略**：
   - (A) 造個 random M-tensor（同 Multi Q5、diagonally dominant、halving 少）
   - (B) 用 HNI paper 的 benchmark eigenvalue 問題
   - (C) 兩個都做
   - 建議：**(A) 先**，保證 port 正確、parity 通過；若之後要 demo 或 publication-level validation 再加 (B)。

5. **inexact 分支的 `nit` offset 細節**：
   - `inner_tol = max(1e-10, min(res)*min(x)^(m-1)/nit)` 裡的 nit 是字面 1-based
   - Python port：用 `(nit + 1)` 顯式、或 `nit_mat = nit + 1` 變數明名？
   - 建議：**後者更清楚**。函式內部保有 `nit`（0-based Python 變數）、並維護 `nit_mat = nit + 1` 當字面 MATLAB 等效、在需要字面值的地方用 `nit_mat`。docstring 明寫。

6. **`innit += chit + 1` 的明示 offset**：
   - Multi 返回 Python nit（0-based、即 MATLAB nit - 1）
   - HONI 的 innit scalar 要與 MATLAB bit-identical、需累加 `chit_py + 1`
   - 建議：**在 HONI port 的 docstring 和程式碼註解明寫這個 offset**，放進 checklist。

7. **polymorphic A 的分派**：
   - 建議：**match MATLAB 的行為**（`A.ndim == 2` → 當 unfolding；`A.ndim >= 3` → 當 tensor、呼叫 ten2mat）。這讓 HONI Python API 對 MATLAB 使用者自然。

8. **plot_res 是否支援**：
   - 建議：**不 port plot**（省依賴、不影響數值正確性）。若將來 Streamlit demo 需要，再另寫 wrapper。

這 8 個問題會在階段 B 第 1 步先定、然後才動 Python 碼。

---

**階段 B 的起跑線（摘要）**：HONI 主體演算法已拆解清楚、exact/inexact 差異矩陣化；所有 dependency（Multi + 5 工具）已 port + parity 通過；新風險集中在「分支公式抄錯」/「`innit` 和 `inner_tol` 的 1-based nit offset」/「sparse+dense `lambda_U*II - AA` dispatch」三處、checklist 已列。階段 B 第一步：回答 §七 open questions → 寫 `HONI_with_history.m` + `generate_honi_reference.m`（exact + inexact 兩個 case）→ 實作 `python/tensor_utils.py::honi` → 寫 `python/test_honi_parity.py` → sanity → MATLAB ref → parity。
