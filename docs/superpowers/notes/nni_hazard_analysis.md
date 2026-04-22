# NNI.m — Port 前 Hazard Analysis

**日期**：2026-04-22（Day 3 晚，Session 4 階段 A）
**對象**：`source_code/Tensor Eigenvalue Problem/2020_HNI_Revised/NNI.m`（271 行；主體 line 29-76，其餘是 local helpers + input_check）
**目的**：在動手 port 之前釐清 NNI vs HNI 的演算法差異、MATLAB 原版的 linear solver 邏輯、Python scope 擴展（新增 `gmres` 分支）、五大陷阱 + NNI 專屬陷阱、per-iter parity 欄位。下一階段（B）才寫 Python 實作 + parity。

**前置**：HNI 線 Layer 1/2/3 已全綠（tenpow / tpv / sp_tendiag / ten2mat / sp_Jaco_Ax + Multi + HONI、含 exact/inexact 雙分支 + 三 Tier parity 框架）。5 個 Layer 1+2 工具 + Multi + HONI 共 7 個函式在 `python/tensor_utils.py`、per-iter parity 三件組在 `python/poc_iteration/parity_utils.py`。

**Scope 擴展（使用者 Day 3 指定）**：Python port 不只逐行 port MATLAB 2020_Revised 版、要加**linear solver 選單**：
- `linear_solver='spsolve'`（預設）：對應 MATLAB `'GE'`（backslash）、用來與 MATLAB parity
- `linear_solver='gmres'`：Python-only、`scipy.sparse.linalg.gmres`、適合大 sparse tensor
- Parity test 只跑 `'spsolve'`、bit-identical 目標
- `'gmres'` 獨立 sanity test、確認「兩 solver 給出近似 eigenpair」（不求 bit-identical）

---

## 一、函式簽章

MATLAB 原型（line 1）：
```matlab
function varargout = NNI( varargin )
```

**變長介面**：透過 `input_check`（line 220-269）把 varargin 拆成 6 個固定變數、透過 `switch nargout` 把 varargout 組出來。Python port 打平成明確 parameters。

### 輸入（varargin，最多 4 個）

| 位置 | 名稱 | 預設 | 說明 |
|---|---|---|---|
| 1 | `A` | — | m-order n-dim tensor（`A.ndim == m`）**或** unfolding matrix `(n, n^(m-1))`。`input_check` 用 `length(size(A))` 判斷、自動 dispatch（跟 HONI 一字不差） |
| 2 | `plot_res` | `0` | `1` 則 `loglog(1:nit, res(1:nit))`。Port 忽略 |
| 3 | `tolerence` | `1e-12` | 外層 Newton 停止門檻（注意 MATLAB 原拼字錯誤 `tolerence` 不是 `tolerance`、docstring 沿用；Python port 用正確拼字 `tol`） |
| 4 | `option` | `struct('linear_solver','GE','initial_vector',[],'maxit',100)` | 三欄 struct |

**option 三欄**：
- `linear_solver`: `'GE'`（預設、backslash `mldivide`）**或** `'GTH'`（Gauss-Tower Householder、M-matrix 專用、用 `geM.m` 的 Alfa-Xue-Ye LU 分解）
- `initial_vector`: `[]`（預設 → `rand(n,1)`）或使用者指定
- `maxit`: 外層 Newton 硬上限（預設 100）

### 輸出（varargout、1-5 個、`switch nargout`）

| nargout | 返回順序 |
|---|---|
| 0 或 1 | `lambda` |
| 2 | `[x, lambda]` |
| 3 | `[x, lambda, res]`（length = outer nit） |
| 4 | `[x, lambda, res, nit]` |
| 5 | `[x, lambda, res, nit, chit]`（加 halving 總次數、**在 canonical NNI.m 永遠為 0**、見 §五） |

**Python port 建議簽章**（綜合 scope 擴展）：
```python
def nni(
    A, m, tol=1e-12, *,
    linear_solver="spsolve",   # "spsolve" (MATLAB 'GE') / "gmres" (Python-only)
    maxit=100,
    initial_vector=None,
    gmres_opts=None,            # dict: maxiter / rtol / atol / restart / M (preconditioner)
    record_history=False,
    plot_res=False,             # 保留 kwarg、實作忽略（同 HONI）
    matlab_compat=False,
):
    """..."""
    return (lambda_, x, outer_nit, res_history, lambda_U_history, lambda_L_history, chit)
    # 若 record_history=True、加一個 history dict 在尾端（見 §六）
```

**與 HONI 簽章對比**：
- HONI 有 `linear_solver in {'exact', 'inexact'}`（演算法分支、**數學公式不同**）
- NNI 的 `linear_solver in {'spsolve', 'gmres'}`（**同一數學公式、只是線性求解器不同**）
- 名稱類似、語意截然不同、docstring 要明示避免混淆

**`lambda` 是 Python keyword**：用 `lambda_` 或 `mu`（MATLAB docstring 用 `mu = NNI(A)`、建議 `mu` 或 `lambda_`）。本 port 建議 `lambda_`（跟 HONI 一致）。

**m 不在簽章裡？** — MATLAB 原版 m 是 `input_check` 內部推算（見 line 249-257：若 A 是 matrix 用 `log/log` 推、若 A 是 tensor 用 `length(size(A))`）。Python 建議：**把 m 放簽章首位必填**（同 HONI 慣例），使用者自己確定 m、避免 log-inference 的浮點數 round 風險。

---

## 二、迭代主結構

### 2.1 NNI vs HNI/HONI 演算法差異（關鍵）

**HNI/HONI**（雙層迭代）：
- 外層更新 `lambda_U`、內層解多線性 `(lambda_U*II - AA) · u^(m-1) = x^(m-1)`（→ 呼叫 `Multi` 做內層 Newton+halving）
- 每個外層 iter 會觸發 Multi 的 Newton 迴圈（通常 5-15 步）、每步 Multi 做**一次** sparse linear solve
- Shift-invert 結構、收斂尾段 near-singular → y_history 量級指數放大（Day 3 memory `feedback_honi_multi_fragility_propagation.md`）

**NNI**（單層迭代）：
- 直接對 `F(x) = AA·x^(m-1)` 做 Newton 型迭代、**每個外層 iter 只解一次 sparse linear system** `(-M) · w = x^(m-1)`、無內層
- Jacobian `M = sp_Jaco_Ax(AA,x,m) - (m-1)*lambda_U*diag(x.^(m-2))`
- 線性求解器的角色：**MATLAB 就是 backslash**、Python port 擴展成 `spsolve` / `gmres` 選單

**兩者數學上解同一個問題**（非負張量最大 H-eigenvalue + eigenvector）、但演算法路徑完全不同：

| 維度 | NNI | HONI |
|---|---|---|
| 迭代層數 | **單層** | 雙層（外層 λ + 內層 Multi） |
| 每 iter 線性 solve 次數 | 1 次（`M\b`） | 多次（Multi 內層 Newton 每步一次） |
| 線性系統大小 | (n × n) sparse | (n × n) sparse（Multi 內部） |
| Line search | canonical NNI.m **無**（halving 註解掉、theta=1） | Multi 內層有 `/3` 三等分 halving |
| λ 更新 | `lambda_U = max(temp)`（step_length 內重算） | exact: `-=min(temp)^(m-1)` 增量 / inexact: `max(temp)` 重算 |
| res 公式 | `(lambda_U - lambda_L) / lambda_U` | `\|max(temp) - min(temp)\| / lambda_U` |
| Shift-invert 結構 | 有（M 的定義含 `-(m-1)*lambda_U*diag(x^(m-2))` 項） | 有（`lambda_U*II - AA`） |
| Fragility 預期 | ⚠️ 同型但較弱（see §五） | 有（見 Day 3 memory） |

**注意**：兩邊的 `res` 公式數學上**等價**（`(lambda_U - lambda_L) / lambda_U = |max - min| / max = 1 - lambda_L/lambda_U`）、只是變數命名不同。

### 2.2 初始化（line 29-41）

```
[AA, m, n, plot_res, tolerence, option] = input_check(...)
                                         ├─ 若 A 是 tensor (m-D) → ten2mat(A, 1)
                                         └─ 若 A 是 matrix (2-D) → 原樣

x         = option.initial_vector                  line 31
x         = x / norm(x)                            line 32  normalize
temp      = tpv(AA, x, m) ./ (x.^(m-1))            line 33  element-wise local eigenvalue
lambda_U  = max(temp)                              line 34
lambda_L  = min(temp)                              line 35

maxit     = option.maxit                           line 37
res       = ones(maxit, 1)                         line 38  pre-alloc, pre-fill 1
lam       = ones(maxit, 1)                         line 38  ⚠️ pre-alloc but never written (dead)
res(1)    = (lambda_U - lambda_L) / lambda_U       line 39
lam(1)    = lambda_U                               line 40
nit       = 1                                      line 41  MATLAB 1-based init
chit      = 0                                      line 41  halving total
```

**關鍵細節**：
- `temp = tpv(AA, x, m) ./ (x.^(m-1))` 跟 HONI.m 一字不差（每個分量的局部 eigenvalue 估計）
- `res` pre-fill 1.0（理由同 HONI：`res = (λ_U - λ_L)/λ_U ≤ 1` 當 λ_L ≥ 0）
- **`lam(maxit,1)` pre-alloc 是 dead code**：line 64 `% lam(nit) = lambda_U;` 註解、整隻 `lam` 除了 `lam(1) = lambda_U`（line 40）之外從未被寫入或讀取。Python port **完全忽略 `lam` buffer**。
- `nit = 1` MATLAB 1-based 起算 → Python 用 `nit = 0`

### 2.3 外層 Newton 迴圈（line 42-75）

```
while min(res) > tolerence && nit < maxit
    nit = nit + 1                                                line 43
    
    B   = sp_Jaco_Ax(AA, x, m)                                   line 44  Jacobian of F(x) = AA·x^(m-1)
    % B = sp_Jaco_Ax_sym(AA, x, m)                                line 45  ⚠️ commented, semi-symmetric case
    M   = B - (m-1) * lambda_U * diag(x.^(m-2))                  line 46  shifted Jacobian
    
    if strcmp(option.linear_solver, 'GE')                        line 48
        w = (-M) \ x.^(m-1)                                      line 49  ← MATLAB backslash
    elseif strcmp(option.linear_solver, 'GTH')                   line 50
        u     = x                                                line 51  (copy; never mutated)
        P     = diag(diag(M)) - M                                line 52  M 的離對角部分（變號）
        [L,U] = geM(P, u, M*u)                                   line 53  Alfa-Xue-Ye LU on M-matrix
        w     = -U \ (L \ (x.^(m-1)))                            line 54
    end
    
    y            = w / norm(w)                                   line 56  normalize
    lambda_U_old = lambda_U                                      line 58  ⚠️ dead: 只被 line 66 的 commented guard 用
    
    [x, lambda_U, lambda_L, hit] = step_length(AA, m, x, y, w, lambda_U)   line 60
    chit = chit + hit                                            line 61  halving 總次數（NNI.m canonical: hit≡0）
    
    res(nit) = (lambda_U - lambda_L) / lambda_U                  line 63
    % lam(nit) = lambda_U                                         line 64  ⚠️ commented, dead
    
    % if nit > 1                                                  line 65-74  ⚠️ entire block commented
    %     if lambda_U - lambda_U_old > 1e-15 || isnan(res(nit))
    %         x = (x - y*theta)/(m-2)
    %         ...rollback nit-1 and break with fprintf...
    %     end
    % end
end
```

### 2.4 `step_length()` 詳解（line 148-189）

**canonical NNI.m 版本**（halving 區塊**整個註解**、只有 theta=1 一次）：

```matlab
function [x_new, lambda_U_new, lambda_L_new, hit] = step_length(AA,m,x,y,w,lambda_U)
    % if m == 3  ... (line 149-165 整塊 Eta-based 特殊公式 commented)
        tol_theta    = 1e-8
        hit          = 0
        theta        = 1
        x_new        = (m-2)*x + 1*y                    % ← 核心更新公式（hyperbolic step）
        x_new        = x_new / norm(x_new)
        temp         = tpv(AA, x_new, m) ./ (x_new.^(m-1))
        lambda_U_new = max(temp)
        lambda_L_new = min(temp)
        % while lambda_U_new - lambda_U > 1e-13        ← halving loop（註解掉）
        %     theta /= 2
        %     x_new = (m-2)*x + theta*y
        %     ...
        %     hit = hit+1
        %     if theta < tol_theta: break
        % end
    % end
end
```

**`NNI_ha.m` 的唯一差異**：line 174-186 的 halving 迴圈**沒有註解**、實際啟用、且 `tol_theta = 1e-12`（比 NNI.m 的 1e-8 嚴格）。

→ **Test_Heig2.m 的 benchmark 實際呼叫的是 `NNI_ha`、不是 `NNI.m`**（line 51: `[EV3, EW3, res3, nit3, hav3] = NNI_ha(Q, ...)`）。

**使用者 Day 3 指定 canonical 為 `NNI.m`**、不是 `NNI_ha.m`。所以 port 依 NNI.m（無 halving、`hit ≡ 0`、`chit ≡ 0`）。如未來要重現 Test_Heig2 benchmark、再把 halving 開關當 Python 選項補上（建議 `halving=False`/`halving=True` kwarg）。見 §七 Open Questions #9。

### 2.5 關鍵觀察（port 時逐點核對）

1. **Newton 更新公式 `x_new = (m-2)*x + y`** — 不是標準 Newton Raphson（那會是 `x + w`、或 `x + θw`），是**張量特徵值問題特殊的 hyperbolic step**。
   - `m = 2`：`x_new = 0*x + y = y`（純 Newton 方向、x 不貢獻）
   - `m = 3`：`x_new = x + y`（gradient-like）
   - `m >= 4`：`(m-2)*x` dominates、y 是擾動
2. **`M = B - (m-1)*lambda_U*diag(x^(m-2))`**：這是 NNI 版的 shift（對比 HONI 的 `lambda_U*II - AA`）。
   - `m = 2`：`x^(m-2) = ones`、`diag(ones) = I` → `M = B - lambda_U*I`（純 shifted-inverse、退化到 power-like 方法）
   - `m >= 3`：`diag(x^(m-2))` 對角元素依 x 變化、若某 `x_i ≈ 0` 則 `M_{ii}` 可能 near-singular（§五專屬陷阱）
3. **收斂判準**：`min(res) > tolerence && nit < maxit`（跟 HONI 結構一樣）。注意 `min(res)` 是整條 buffer 的 min（含 pre-fill 1）、要靠 pre-fill ≥ 實際殘差才正確。
4. **符號**：`w = (-M) \ x.^(m-1)` 的 `(-M)` 前面的負號很關鍵、逐字照抄（若忘寫或誤寫成 `M\...` 會讓 `y = w/||w||` 符號翻轉、整個收斂路徑錯）。

### 2.6 收斂判準

| 位置 | 條件 | 語意 |
|---|---|---|
| 外層 while（line 42） | `min(res) > tolerence` | 最小相對殘差未達門檻 |
| 外層 while（line 42） | `nit < maxit` | 硬上限（預設 100） |
| step_length 內（commented） | `lambda_U_new - lambda_U > 1e-13` 觸發 halving | **canonical 未啟用、見 §五.5** |

**不變量**：`x` 保持 `||x|| = 1`（每次 step_length 正規化）；`lambda_U` 理論上 monotone non-increasing（前提：初始 `lambda_U = max(temp) ≥ true eigenvalue`、Newton 步驟保持上界）。

---

## 三、呼叫了哪些函式 — 已 port 狀態

| MATLAB 呼叫（NNI.m 內） | 定義位置 | 已 port 到 Python | 狀態 |
|---|---|---|---|
| `tpv(AA, x, m)` | NNI.m line 110-114（local） | `tensor_utils.py::tpv` | ✅ Layer 1 完成 |
| `tenpow(x, p)`（via tpv / sp_Jaco_Ax） | NNI.m line 136-146（local） | `tensor_utils.py::tenpow` | ✅ Layer 1 完成 |
| `sp_Jaco_Ax(AA, x, m)` | NNI.m line 117-125（local） | `tensor_utils.py::sp_Jaco_Ax` | ✅ Layer 2 完成 |
| `sp_Jaco_Ax_sym(AA, x, m)` | NNI.m line 127-133（local） | **未 port、且 canonical 註解掉 line 45**、不需要 | — |
| `ten2mat(A, k)`（via input_check） | NNI.m line 191-204（local） | `tensor_utils.py::ten2mat` | ✅ Layer 2 完成（column-major 主檢查點） |
| `idx_create(n, type)`（via ten2mat） | NNI.m line 206-216（local） | 不需要 — Python `np.moveaxis` + `reshape(order='F')` 取代 `eval(express)` | — |
| `step_length(AA,m,x,y,w,lambda_U)` | NNI.m line 148-189（local） | **未 port、需新寫** | 新 port target #1 |
| `input_check(...)` | NNI.m line 220-269（local） | 不需要、Python 用 keyword args 取代 | — |
| `geM(P, u, M*u)`（via GTH 分支） | `2020_HNI_Revised/geM.m` 54 行（獨立） | **未 port** — M-matrix 專用、**本 port scope 不含 GTH**、故不需要 | — |
| `diag(x.^(m-2))` | MATLAB built-in | scipy `scipy.sparse.diags(x**(m-2))` | 新用法、見 §四陷阱 #5 |
| `(-M) \ x.^(m-1)` | MATLAB backslash | `scipy.sparse.linalg.spsolve(-M.tocsc(), x**(m-1))` | 新用法、見 §四陷阱 #5 |
| `max`, `min`, `abs`, `norm` | MATLAB built-in | `np.max / np.min / np.abs / np.linalg.norm` | — |
| `strcmp` | MATLAB built-in | Python `==` 字串比較 | — |
| `ones(n,1)`, `rand(n,1)` | MATLAB built-in | `np.ones(n) / np.random.default_rng().random(n)` | — |
| `loglog(...)` | MATLAB plot | matplotlib（**不 port**、同 HONI） | — |

**總結**：
- **需新 port 的只有 `step_length`**（約 10 行活碼、結構簡單）
- `geM.m` 是 GTH 分支才需要、本 port scope 不含 → 免 port
- **所有重要 dependency（tpv/tenpow/sp_Jaco_Ax/ten2mat）已 port + parity 通過** → NNI port 不會引入新的 Layer 1/2 陷阱

---

## 四、五大陷阱逐行分析

### 陷阱 #1：邊界處理
**N/A** — NNI 不是 filter、無邊界概念。

### 陷阱 #2：濾波器 / 核截斷寬度
**N/A** — 無 kernel。

### 陷阱 #3：1-based vs 0-based indexing

| 行號 | MATLAB 原碼 | 是否觸發 | 處置 |
|---|---|---|---|
| 38-40 | `res = ones(maxit,1); nit=1; res(1) = ...; lam(1) = lambda_U` | ⚠️ **是**（同 Multi/HONI 的 pattern） | Python：`nit = 0`、`res = np.ones(maxit)`、`res[0] = ...`；`lam` 整個忽略（dead code） |
| 42 | `while min(res) > tolerence && nit < maxit` | 中 | Python `nit < maxit - 1`（同 Multi/HONI 的 0-based 決定） |
| 43 | `nit = nit + 1` | 低 | 同步 |
| 63 | `res(nit) = (lambda_U - lambda_L) / lambda_U` | 低 | `res[nit]` |
| 83-105 | `switch nargout` + `res(1:nit)` | 中 | Python 固定返回全部、`res[:nit + 1]`（MATLAB `1:nit` ≡ Python `0:nit+1`） |

**結論**：同 Multi/HONI 慣例，Python 全用 0-based、`nit = 0` 起算、第 `[0]` 格存初始化狀態。**不特別的新點**。

### 陷阱 #4：Column-major vs Row-major

**NNI 主體幾乎不觸發**。主迴圈 line 42-75 沒有 `reshape`、沒有 linear index、全部是 matrix/vector 逐項運算。

**間接觸發**（已處理）：
- `input_check` 若 `A.ndim >= 3` 呼叫 `ten2mat(A, 1)` — 已 port、column-major 處理完成（Layer 2 主檢查點）
- `sp_Jaco_Ax` 內部有 sparse `kron`（layout-free 但邏輯順序依 column-major tensor unfolding 慣例）— 已 port

**新注意點**：
- **`diag(x.^(m-2))`** 在 Python 要用 `scipy.sparse.diags(x**(m-2))` 建 n×n sparse diagonal。layout 中立、不觸發 column-major。
- **`(-M)\x.^(m-1)`** 是 matrix-vector 線性 solve、layout 中立。

**結論**：NNI 自身 port **不需要再注意 column-major**（所有 reshape 都已在 Layer 1/2 處理）。

### 陷阱 #5：迭代控制流 + sparse 數值（HONI/Multi 陷阱 #5 在 NNI 的延伸）

| 子項 | 行號 | 內容 | 風險 |
|---|---|---|---|
| **5.1 `M = B - (m-1)*lambda_U*diag(x.^(m-2))`** | 46 | `B` sparse（`sp_Jaco_Ax` 返回 csr）；`diag(x.^(m-2))` MATLAB 是 **dense** n×n diagonal matrix；減法後 `M` 是 **dense**（MATLAB 允許 sparse-dense 相減、結果 dense） | ⭐ **中-高**。Python 要顯式走 sparse path：<br>- `D = scipy.sparse.diags(x**(m-2))`（sparse csr n×n）<br>- `M = B - (m-1)*lambda_U*D`（兩 sparse 相減 → sparse）<br>- **不要**走 `np.diag(x**(m-2))` + sparse-dense（會把 M 變 dense、大 n 時爆記憶體） |
| **5.2 `x.^(m-2)` 當 m=2** | 46 | MATLAB `x.^0 = ones(n,1)`、numpy `x**0 = np.ones(n)` — 一致 | 低。但若 `x` 有 `0` 分量、`0^0 = 1`（MATLAB 與 numpy 都這樣）、無異常。 |
| **5.3 `x.^(m-2)` 當 `x_i ≈ 0`、m ≥ 3** | 46 | `0^(m-2) = 0` → `diag` 對角第 i 項 0 → M 在那列非常 ill-conditioned / singular | ⭐ **高**。**NNI 特有**的 fragility 觸發點、詳見 §五.2 |
| **5.4 `w = (-M) \ x.^(m-1)` 符號** | 49 | `-M`（M 變號）、**不是** `M`。Python 要 `spsolve((-M).tocsc(), x**(m-1))` 或 `-spsolve(M.tocsc(), x**(m-1))` | ⭐ **高**：若 sign 弄錯、`y = w/||w||` 符號翻轉、下一輪 `x_new = (m-2)*x + y` 往相反方向走、parity 從 iter 2 開始爆 |
| **5.5 `spsolve` CSR → CSC** | 49 | scipy `spsolve` 偏好 CSC、CSR 會 warn | 低。`(-M).tocsc()` 顯式轉換、同 Multi port 的慣例 |
| **5.6 `gmres` 分支處理（scope 擴展、Python-only）** | — | 使用 `scipy.sparse.linalg.gmres(-M, x**(m-1), **gmres_opts)`、return tuple `(w, info)`；`info=0` 成功、`info>0` 到 maxiter 未收斂、`info<0` illegal input | ⭐ **中**。需要 runtime 檢查 `info`、warn 或 raise（見 §七 Open Q #10） |
| **5.7 Pre-alloc `res = ones(maxit,1)`** | 38 | 同 Multi/HONI 的 pre-fill 策略、理由：`res ≤ 1` 當 `lambda_L ≥ 0`（clean M-tensor） | 低-中。若 `lambda_L < 0`（初期 iter 可能）、`res = 1 - lambda_L/lambda_U > 1`、pre-fill 失效。見 §五.4 |
| **5.8 `dtype` 保存** | 33, 49 | `x.^(m-1)` 若 `x` 是 int 且 m=2 就 ok、但若 caller 傳 int 陣列、numpy 行為不同：`np.array([1,2], dtype=int) ** 1 = int array`、無 `dtype` 隱式轉 float。HONI/Multi 的處理：`b = np.asarray(b, dtype=np.float64)` 入口立即轉 | 低。Port 開頭 `x = np.asarray(x, dtype=np.float64).ravel()`、同 Multi |
| **5.9 `||`/`&&` short-circuit 順序** | 42 | `min(res) > tolerence && nit < maxit` 逐字保留、不要倒過來（同 Multi line 22、HONI line 46 的慣例） | 低 |
| **5.10 不要 mutate caller 的 `initial_vector`** | 31-32 | MATLAB line 31 `x = option.initial_vector`（copy-on-write）、line 32 `x = x/norm(x)` 覆寫 x 但不影響 option。Python 若 caller 傳 ndarray、`x = initial_vector` 是**view**、`x = x/norm(x)` 產生**新 array**（不 mutate）。OK | 低。Port 慣例：`x = np.asarray(initial_vector, dtype=np.float64).ravel() / norm` 一氣呵成 |

**陷阱 #5 結論（NNI port checklist）**：

- [ ] `M = B - (m-1)*lambda_U * scipy.sparse.diags(x**(m-2))` — sparse 減法、**不走 `np.diag`**
- [ ] `(-M).tocsc()` 給 `spsolve`、保持符號和格式
- [ ] `gmres` 分支接 `(w, info)` tuple、檢查 info
- [ ] 開頭 `x = np.asarray(...).ravel() / norm`
- [ ] `res = np.ones(maxit)`、assert `res[:nit+1] <= 1.0 + tol` 在 exit
- [ ] `nit = 0` 起算、其他 indexing 同 Multi/HONI 慣例
- [ ] `lam` buffer 完全忽略（dead code）

**NNI 專屬手眼同步點（對應 §五專屬陷阱）**：

- [ ] **假設 input 是正 M-tensor**、`x > 0` 沿 iteration 保持、**不做主動非負投影**（不 `np.maximum(x, 0)`、不 `np.abs(x)`）；docstring 明寫此前提（§五.1）
- [ ] **`diag(x^(m-2))` 用 `scipy.sparse.diags(x**(m-2))`、不用 `np.diag`**；避免大 n 時 dense n×n 爆記憶體；`M` 整路走 sparse（§五.2、§四.5.1）
- [ ] **`step_length` 從 MATLAB line 169 重寫**（`θ=1; x_new = (m-2)*x + y; x_new /= norm(x_new); recompute temp, lambda_U, lambda_L`）；MATLAB line 174-186 的 halving `while` 迴圈**整個註解**、Python port 等價實作無 halving、`hit ≡ 0`、`chit ≡ 0`（§五.5）
- [ ] **`res` pre-fill = 1.0 的 assertion 在收尾前加**（`assert np.all(res[:nit+1] <= 1.0 + 1e-10)`）；對應 Q6 決定；若 fail 表示 `lambda_L < 0`、AA 不是 clean M-tensor、測試 case 要檢查（§五.3）

---

## 五、NNI 專屬新陷阱（HONI 沒遇過的）

### 5.1 非負投影 — **原 canonical 完全不投影、僅正規化**

**重要釐清**：NNI 的名稱 Nonnegative Newton Iteration 意味著**目標解是非負特徵對**、**不是每 iter 都強制投影到非負**。

MATLAB NNI.m 對 `x` 的唯一變換：
- line 32: `x = x / norm(x)`（L2 正規化）
- line 56: `y = w / norm(w)`（L2 正規化 Newton 方向）
- line 169 in step_length: `x_new = (m-2)*x + 1*y; x_new = x_new / norm(x_new)`

**沒有任何地方** `max(0, x)` / `x(x<0) = 0` / `abs(x)` — 演算法靠 Perron-Frobenius 類性質保證 `x` 在收斂路徑上保持非負（當 AA 是非負 M-tensor、初值 `x_0 > 0`）。

**Port 應對**：
- **不加任何投影**（會破壞 parity）
- **不加任何保護性 clipping**（e.g. 不寫 `x = np.maximum(x, 0)`、不寫 `x = np.abs(x)`）
- 若某個 iter 某 `x_i < 0`、port 照原樣傳入下一 iter — 這是 MATLAB 原碼行為
- 若收斂卡住（res 不降）、是數學問題（AA 不是 M-tensor、或 x_0 不夠非負）、不是 port bug

**未來擴展**（非本 port scope）：可加 `project_nonneg=False` kwarg、True 時每 iter 做 `x = np.abs(x)`（某些 practical 使用場景會要）。現階段 **不要做**、會讓 parity 斷掉。

### 5.2 `diag(x^(m-2))` 在 x 近零分量時 ill-conditioned

當 `x_i → 0` 且 `m ≥ 3`：
- `x_i^(m-2) → 0`
- `diag(x.^(m-2))` 第 i 個對角元素 → 0
- `M = B - (m-1)*lambda_U*diag(x^(m-2))` 第 i 列的 shift 項消失
- 若 `B` 第 i 列也退化（e.g. `B_{ii}` 小）、**M 在第 i 列 near-singular** → `spsolve` 解出的 `w_i` 可能巨大 → `y = w/||w||` 主要由那分量主導 → 下一 iter 的 `x_new = (m-2)*x + y` 被 y 主導 → x 偏離真 eigenvector

**HNI/HONI 不觸發這個**：HONI 的 `lambda_U*II - AA` 是均勻 shift（所有對角同樣 shift `lambda_U`）、不依 x；而 NNI 的 shift 逐分量不同、依 `x^(m-2)`。

**Fragility 量化（預期、需實測）**：
- Clean M-tensor + 適當 x_0：`x` 所有分量 ~ 1/sqrt(n)、`x^(m-2)` 所有分量 ~ n^((2-m)/2)、均勻分佈、M 良 conditioned → w / y 不爆、parity 通過 strict tol
- Non-M-tensor 或 x_0 有 0 分量：可能遇到局部 ill-conditioning、w 某分量量級放大、parity 在 w/y 上可能需 rtol

**建議應對**：
- 仿 HONI 三 Tier 框架、w_history 和 y_history 預備 rtol fallback
- parity test 先用 healthy test case（見 §七 Open Q #7）、若通過 strict abs tol、皆大歡喜
- 若某 iter 的 w/y rel diff > 1e-8、切成 Tier 2（rtol 1e-2）、document in memory

### 5.3 停止判準 `(lambda_U - lambda_L) / lambda_U` 的符號風險

`lambda_U = max(temp)`、`lambda_L = min(temp)` where `temp = tpv(AA, x, m) ./ (x.^(m-1))`。

**若 `x` 有任何分量的 `x_i^(m-1)` 和 `(AA·x^(m-1))_i` 符號不同**：
- `temp_i < 0`
- 若這是全局 min、`lambda_L < 0`
- 若 `lambda_U > 0`（通常）：`res = (lambda_U - lambda_L) / lambda_U = 1 + |lambda_L|/lambda_U > 1`
- **`res[nit]` 超過 pre-fill 值 1** → pre-alloc 策略的 `min(res)` 判準失效（`min` 會回 pre-fill `1`、不是最新值）

**應對**：
- Port 函式尾端加 sanity：`assert np.all(res[:nit + 1] <= 1.0 + 1e-10)` 檢查（**但這會 fail 在 non-M-tensor、是 expected**）
- 更 robust：改成 `assert` 只 warn、不 raise；或 docstring 明寫「本 port 假設輸入是 M-tensor、`res ≤ 1`；若非此情況、pre-fill 策略失效、行為等同 MATLAB（演算法仍跑、`min(res)` 的語意退化）」
- 建議：**照 MATLAB 逐字 port（不加保護）、docstring 明示此假設**、同時在 parity test 選 clean M-tensor 避免觸發

### 5.4 `lambda_U` 非 monotone non-increasing 的 debug 陷阱

**理論上** NNI 每 iter 讓 `lambda_U` 單調遞減到真 eigenvalue（這是 canonical 的 halving 設計動機、雖然 NNI.m 把 halving 註解掉了）。

**實際上** canonical NNI.m 無 halving、若 theta=1 的 x_new 造成 `lambda_U_new > lambda_U`（overshoot、line 174 的 commented guard 本來要 catch）：
- `lambda_U_new` 會上升
- 下一 iter 的 M 用更大的 lambda_U、shift 更大、w 方向改變
- 若 AA 是 clean M-tensor + x_0 接近真 eigenvector、不會觸發
- 若不是、**可能 diverge** 或 **振盪不收斂**

**對 parity 的影響**：Python 和 MATLAB 會一起 diverge / 一起振盪（同 iteration sequence）、所以 parity 仍成立。**不是 port bug、是演算法 fragility**。

**對 port 的影響**：不需處理、但 docstring 要寫「canonical NNI.m 無 halving、若輸入非 healthy M-tensor 可能 lambda_U 上升 / diverge、行為與 MATLAB 原碼一致；需 halving 版請用未來的 `NNI_ha` mode」。

### 5.5 `NNI.m` 的 `step_length` halving 完全註解（`chit ≡ 0`）

canonical NNI.m 的 `step_length`（line 148-189）結構：
- `tol_theta = 1e-8` 定義（活的）
- `hit = 0`、`theta = 1`（活的）
- `x_new = (m-2)*x + 1*y; normalize; recompute temp, lambda_U, lambda_L`（活的）
- **halving `while` 迴圈整個註解**（line 174-186）
- `end` 在 `% end` 之下、以 `% end` 結尾（line 188）

**結果**：
- `hit` 永遠 = 0
- `chit = chit + hit` 永遠不增、最終 `chit = 0`
- 外部看到的 `chit` 輸出永遠 0

**對 port 的影響**：
- `chit_history` 每 iter 寫 0、scalar `chit` 最終 0
- Python port 保留 `chit` 在返回 tuple（match MATLAB 簽章）、但 **docstring 明寫「canonical NNI.m 永遠 `chit = 0`、除非未來啟用 halving」**
- parity test 要對 chit = 0 做 `assert`

**`NNI_ha.m` 的差別（對照記錄、不 port）**：
- halving 迴圈活的、用 `theta /= 2`（不是 `/3`、跟 Multi 不同）
- `tol_theta = 1e-12`（比 NNI.m 嚴格）
- 觸發條件：`lambda_U_new - lambda_U > 1e-13`（即 lambda_U 反而上升就 halve）
- 這個邏輯跟 Multi 的 halving 語意相反：Multi halve 是為了 `res` 減少、NNI_ha halve 是為了 `lambda_U` 遞減

### 5.6 被使用者忽略的 commented guard block（line 65-74）

```matlab
%         if nit > 1
%             if  lambda_U - lambda_U_old > 1e-15 || isnan(res(nit))
%                 x        = (x - y*theta)/(m-2);      %%% inverse update
%                 temp     = tpv(AA,x,m)./(x.^(m-1));
%                 lambda_U = max( temp );
%                 nit      = nit - 1;                   %%% rollback
%                 fprintf('NNI stopped at iteration %d without converging...');
%                 break
%             end
%         end    
```

這是**發散保護 + rollback**機制、在 canonical 版本被註解。若未來想實作：
- `lambda_U - lambda_U_old > 1e-15` 觸發（lambda_U 反增）
- 或 `isnan(res)` 觸發（數值爆炸）
- 把 x 用 `(x - y*theta)/(m-2)` **反推回上一 iter**（還原更新）
- `nit -= 1`（計數回退）、break

**Port 建議**：不實作（canonical 未啟用）。但 docstring 備註「原碼註解有 divergence guard、本 port 未實作、若演算法 diverge 會直接跑到 maxit、行為與 canonical MATLAB 一致」。

### 5.7 `main_Heig.m` / `Test_Heig*.m` 的 driver 混亂

- `main_Heig.m`（line 22）呼叫 `NNI_two`、但 **`NNI_two.m` 在 2020_HNI_Revised 資料夾不存在**（inventory Section B.2 記錄、是 Revised 版繼承自 Heig 版的 stale reference）
- `Test_Heig.m` 混合呼叫 `HNI` / `IHNI`（這兩個在 Revised 也不存在、只在 Heig 版）
- `Test_Heig2.m`（inventory 說它是 canonical benchmark 腳本、line 51）實際呼叫 `NNI_ha`、**不是 `NNI.m`**
- `Test_Heig3.m` 和 `Test_Heig4.m` 結構類似、需要實際讀確認

**對 port 的影響**：
- **不要用 `main_Heig.m` 當 driver 參考**（它壞掉、且呼叫的 NNI_two 不是使用者指定的 canonical）
- 如果未來要重現 Test_Heig2 benchmark、要先 port `NNI_ha` 版本（halving 啟用、`theta/=2`、`tol_theta=1e-12`）、或用 `halving=True` kwarg 切換
- 本 port scope **只做 NNI.m canonical**、driver / benchmark 整合推到 Layer 4 階段

---

## 六、Per-iteration parity — 要存的 history 欄位

依照 HONI port 的 history 設計（Day 2 末完成的 9 欄 pattern）、NNI 是**單層迴圈**、history 比 HONI 簡單。

### 外層 history（長度 `outer_nit + 1`、index 0 存初始化狀態）

| 欄位 | 形狀 | 意義 | `[0]` 位置存什麼 |
|---|---|---|---|
| `x_history` | `(n, outer_nit + 1)` | 每 iter 結束後的 normalized x | `initial / \|initial\|`（line 32） |
| `lambda_U_history` | `(outer_nit + 1,)` | 每 iter 結束後的 lambda_U | `max(tpv/x^(m-1))`（line 34） |
| `lambda_L_history` | `(outer_nit + 1,)` | 每 iter 結束後的 lambda_L | `min(tpv/x^(m-1))`（line 35） |
| `res_history` | `(outer_nit + 1,)` | 相對殘差 | `(lambda_U - lambda_L) / lambda_U`（line 39） |
| `w_history` | `(n, outer_nit + 1)` | 每 iter Newton direction `M\b`（未 normalize）| `NaN`（iter 0 無 solve） |
| `y_history` | `(n, outer_nit + 1)` | 每 iter normalized Newton direction `w/\|w\|` | `NaN`（iter 0 無 solve） |
| `chit_history` | `(outer_nit + 1,)` | 累積 halving 總次數 | `0`（canonical NNI.m 永遠 0） |
| `hit_per_outer_history` | `(outer_nit + 1,)` | 每 iter 的 halving 次數（canonical 永遠 0） | `0` |

**欄位數（8）< HONI（9）**、少了 `inner_tol_history` / `innit_history` / `hal_accum_history`（這些是 HONI 內層 Multi 產物、NNI 無內層）。

### 沒有「內層 history」（對比 HONI）

HONI 有 inner Multi history option（`record_inner_history=True`）— NNI **無內層**、此 kwarg 無用、**不加到 NNI 簽章**。

### MATLAB reference 腳本草案（階段 B 會寫、此處只記 spec）

- 檔名：`matlab_ref/hni/generate_nni_reference.m`（也可放 `matlab_ref/nni/`、使用者決定）
- 改寫 `NNI.m` 副本 `NNI_with_history.m`（不動 canonical）、加 history outputs
- `rng(42)` 固定、一個 test case、linear_solver='GE'（對應 Python 'spsolve'）
- 輸出：`nni_reference.mat`（單 case、不像 HONI 有 exact/inexact 兩 case、因為 NNI.m 的 `GTH` 分支本 port 不支援）
- Save: 輸入 (`AA`, `m`, `n`, `tol`, `maxit`, `initial_vector`, `linear_solver`) + scalars (`lambda`, `nit`, `chit`) + history 8 欄

### 預期 fragility（參考 HONI 經驗）

**低 fragility 預期**（比 HONI 容易 parity）：
- 單層迴圈、無 Multi 內層 Newton、無 halving 的自然累積誤差
- 若 AA 是 clean M-tensor、x_0 healthy：**strict abs 1e-10 應全欄位通過**（machine epsilon）

**可能的 fragility 點**（如果出現、用三 Tier 框架處理）：
- **w_history / y_history** 在收斂尾段、M 的條件數爆（`x` 某分量 → 0、`diag(x^(m-2))` 對角有零）→ `spsolve` (SuperLU) vs MATLAB `mldivide` (LU) pivot 差異放大
- 類似 HONI 的 y_history fragility、但**更 local**（NNI 的 ill-conditioning 是 x_i → 0 造成的 row-wise 局部；HONI 的是 lambda_U → eigenvalue 造成的 global shift 消失）
- 若發生、預期比 HONI 輕（單層、無內層累積）

**三 Tier 應用建議**（預備、不先設）：
- Tier 1 STRICT abs 1e-10：x_history、lambda_U_history、lambda_L_history、res、chit_history
- Tier 1 STRICT rel 1e-8：w_history, y_history（mid slots 1..K-2、若 strict abs fail 再降成 rtol）
- Tier 2 APPROX rel 1e-2：w_history[:, K-1] / y_history[:, K-1]（若 last-iter fragile）
- Tier 3 INFO：chit scalars（canonical NNI.m 兩邊永遠 0、trivial）

**若實測全部 strict abs 通過**、直接用單 Tier。若不、升級到 HONI 三 Tier pattern。

---

## 七、Port 前 Open Questions（需要使用者決策）

### Q1. `linear_solver` kwarg 設計 — spsolve vs gmres 細節

**已定**（使用者 Day 3 指定）：`linear_solver in {'spsolve', 'gmres'}`、`'spsolve'` 預設、parity test 只測 spsolve。

**需決定**：
- (a) `gmres` 參數透過 `gmres_opts=dict(maxiter=1000, rtol=1e-10, restart=50, M=None)` 一個 dict kwarg 傳入？還是**頂層 kwargs**（`gmres_maxiter=1000, gmres_rtol=1e-10, gmres_restart=50`）？
- (b) `gmres` 預設 `maxiter` = ? 建議 `min(1000, 10*n)`（小 n 不浪費、大 n 夠用）
- (c) 預設 `rtol` = ? 建議 `1e-10`（比外層 tol=1e-12 寬鬆一個量級、讓外層 converge 不被內層誤差綁架）
- (d) Preconditioner `M` 預設 `None`（無 preconditioner）、使用者要用可傳 `scipy.sparse.linalg.LinearOperator`

**建議**：
- (a) **dict kwarg** `gmres_opts=None`、`None` 時用合理預設。理由：未來加 `bicgstab`/`qmr` 時不用再加 `bicgstab_maxiter` 等頂層 kwargs、保持簽章整潔
- (b) 預設 `maxiter=1000`、常見大 sparse 問題足夠
- (c) 預設 `rtol=1e-10`、`atol=0`
- (d) `restart=50` GMRES-specific restart length、`M=None`

### Q2. `gmres` sanity test tolerance — 「近似 eigenpair 比對」怎麼訂？

**問題**：`'gmres'` 分支不求 bit-identical、但要確認「給出合理 eigenpair」。比對 `lambda_spsolve` vs `lambda_gmres`、`x_spsolve` vs `x_gmres` 的誤差上界。

**建議**：
- `|lambda_spsolve - lambda_gmres| / |lambda_spsolve| < 1e-6`（相對誤差、考慮 gmres rtol=1e-10 累積到外層 ~1e-6 是合理）
- `||x_spsolve - x_gmres|| / ||x_spsolve|| < 1e-5`（考慮 eigenvector 本身比 eigenvalue 敏感一個量級）
- 若 gmres convergence rate 差、放寬到 `lambda rel 1e-4` / `x rel 1e-3`（document 原因）

**最終 tolerance 等實測 gmres 實際表現後 tune**。階段 B 實作時先 build、再跑 sanity、再訂 tolerance。

### Q3. `polymorphic A` 處理 — tensor vs unfolding

**建議**：照 HONI pattern（`A.ndim == 2` 當 unfolding、`A.ndim >= 3` 當 tensor、內部 `ten2mat(A, k=0)`）。**直接沿用 HONI 的 input dispatch code**、不改。

### Q4. MATLAB `option` struct 到 Python kwargs 映射

| MATLAB | Python kwarg |
|---|---|
| `option.linear_solver = 'GE'` | `linear_solver='spsolve'`（對應） |
| `option.linear_solver = 'GTH'` | **不支援、raise ValueError**（需 geM.m、本 port scope 不含） |
| `option.initial_vector = [...]` | `initial_vector=np.array([...])` |
| `option.initial_vector = []` (default) | `initial_vector=None` → 內部 `rng.random(n)` |
| `option.maxit = 100` (default) | `maxit=100` |

**建議**：簡單明瞭、不需拍板。實作時若使用者傳 `linear_solver='GE'` 自動 map 到 `'spsolve'`？建議**不 map**、raise `ValueError("use 'spsolve' for Python port; 'GE' is MATLAB-only")` 避免混淆。

### Q5. `m = 2` 邊界情況

NNI 在 m=2 退化為矩陣特徵值問題：
- `x^(m-2) = x^0 = ones` → `diag(ones) = I` → `M = B - lambda_U*I`
- `x_new = (m-2)*x + y = 0*x + y = y`（純 Newton 方向）
- 本質上是 **shifted inverse power method** for matrix eigenvalue

**問題**：Python port 要不要對 m=2 special-case？
**建議**：**不 special-case、逐字 port**（MATLAB 也沒 special-case）。m=2 時演算法自動退化、語意正確。docstring 註明「m=2 時 NNI 退化為 shifted inverse power method on matrix eigenvalue」。

### Q6. `pre-alloc res = ones(maxit)` upper-bound assumption

**問題**：`res = (lambda_U - lambda_L)/lambda_U ≤ 1` 只在 `lambda_L ≥ 0` 時成立、non-M-tensor 可能違反。

**建議**：
- Port 尾端保留 `assert np.all(res[:nit + 1] <= 1.0 + 1e-10)` 如同 Multi/HONI
- docstring 明示此假設、**並 document「parity test 必須用 clean M-tensor、否則 assert 會 fail」**
- 若未來要放寬、改用 `res = np.full(maxit, np.inf)`、while 判準不受 pre-fill 干擾（但會跟 MATLAB 行為分家、不 port parity）

**此 case 建議選前者、維持 MATLAB parity。**

### Q7. Parity test case 選擇

**問題**：該用什麼 AA / x_0 / m / n 當 reference？

**選項**：
- (a) 手工設計 healthy M-tensor（仿 Multi Q5 / HONI Q5 pattern、小 n、seeded、穩定收斂）
- (b) 用 MATLAB 2020_Revised 自帶的 tensor generator（`tengen_4D.m`、`tengen_HD_weakly.m` 等）— 但這些是 random-seeded、`rng(42)` 固定一個種子
- (c) Test_Heig2 benchmark 的某個 ω 值（但那個用 NNI_ha 不是 NNI.m、且較大 n=50、parity 會慢）

**建議**：(a) 和 (b) 混合：
- 主 parity case：**手工 M-tensor**（`sp_tendiag(diag_vec, m) + perturbation`、仿 Multi Q5、m=3、n=10、rng(42) 初值）、確保無 halving 觸發（NNI.m 無 halving 程式、這點天然滿足）、確保 `x` 收斂路徑無近零分量（避開 §五.2 fragility）
- 次 parity case（optional）：**port `tengen_4D.m` + 用它生成的 AA**、驗「Python 同 generator 結果」 — 但這會引入 `tengen_4D` 的 column-major / indexing 陷阱、delta scope、可延到 Layer 4

**此 Open Q 待使用者拍板**（像 Multi 的 Q5 決策）。

### Q8. `NNI_ha`（halving 版）要不要一併 port？

**問題**：Test_Heig2 實際用的是 NNI_ha、若未來要重現 benchmark、需要 halving 版本。

**選項**：
- (a) **本 port scope 只做 NNI.m canonical**（使用者已定）、halving 以後再加
- (b) 本 port 加 `halving=False`/`halving=True` kwarg、True 時啟用 NNI_ha 的 `while` 迴圈（`theta /= 2`、`tol_theta = 1e-12`）
- (c) 把 `NNI_ha` 也 port 成獨立函式 `nni_ha()`、兩個並存

**建議**：**(a)**、最小 scope、parity 最乾淨。`halving=True` 當未來延伸（Layer 4 benchmark integration 時、再做 NNI_ha port + parity）。

### Q9. `0` vs `1`-based 的 `nit` 和返回慣例

**建議**：完全沿用 Multi/HONI 的慣例（Python 0-based、`nit = 0` 起算、第 [0] 格存 init state、返回 nit = 外層 iter 完成次數、MATLAB `nit_mat = nit_py + 1`）。**無新決策**、只列出給 audit 時 double-check。

### Q10. `gmres` convergence failure 處理

**問題**：`scipy.sparse.linalg.gmres` 回傳 `(w, info)`、`info > 0` 表示未收斂到指定 tol、`info < 0` 輸入非法。

**選項**：
- (a) `info != 0` 就 `warnings.warn`、繼續跑外層（讓外層 res 自然決定收斂）
- (b) `info != 0` 就 raise RuntimeError、停止
- (c) `info != 0` 記到 `history['gmres_info_history']`（sanity test 用）、否則照樣走

**建議**：**(a) + (c) 組合**：warn user + 記 info 到 history（sanity 時看是否所有 iter 都 converged、或某些 iter 未收斂）。不 raise、讓外層 tolerance 決定。

### Q11. Sparse vs dense AA 的 `M` 格式

**問題**：MATLAB `B - (m-1)*lambda_U*diag(x.^(m-2))` 如果 B 是 sparse、`diag(x.^(m-2))` MATLAB 默認 dense（`diag()` 作用在 vector 時生 n×n dense）。相減結果 MATLAB 是 **dense**。

Python 選擇：
- (a) `M = B - (m-1)*lambda_U * scipy.sparse.diags(x**(m-2))`（全 sparse、M sparse）→ parity 可能不 bit-identical（MATLAB 原碼 M 是 dense、dense LU 跟 sparse LU 的 pivot 順序可能不同）
- (b) `M = B.toarray() - (m-1)*lambda_U * np.diag(x**(m-2))`（全 dense、M dense、`np.linalg.solve` 走）→ 較接近 MATLAB 行為、但 n 大時爆記憶體
- (c) 兩個都實作、`dense_matrix=False`（default）走 sparse spsolve、`dense_matrix=True` 走 dense solve

**建議**：**(a) sparse + `spsolve`**。理由：
- HONI port 已驗證 sparse `spsolve` vs MATLAB `mldivide` 在 non-singular 區段 bit-identical（Multi Q5 parity 通過）
- n 大時 dense 不可行、sparse 是長期正確選擇
- 若 parity 失敗、再考慮 (c) 或在 memory feedback 記「MATLAB dense vs Python sparse linear solve 的 pivot 差異」

**實測後再決定是否升級**。先假設 (a) 可行。

### Q12. `innit` 輸出 — NNI 是否需要？

HONI 有 `innit`（內層 Multi Newton 總次數），NNI 無內層、`innit` 概念不存在。

MATLAB NNI.m 的 nargout=5 返回 `[x, lambda, res, nit, chit]` — 就 5 個、**沒有 innit**。

**建議**：Python port 簽章返回 `(lambda_, x, outer_nit, res, lambda_U_history, lambda_L_history, chit)` 或 `(..., chit, history)`；**不加 innit**（與 MATLAB 對齊）。docstring 明示「NNI 單層、無 inner iteration 計數」。

---

## 八、階段 B 的起跑線（摘要）

NNI 主體演算法已拆解清楚、跟 HONI 的差異矩陣化（單層 vs 雙層、shift 結構不同）；所有 dependency（tpv / tenpow / sp_Jaco_Ax / ten2mat）已 port + parity 通過、Layer 3 的 Multi + HONI 不需要呼叫；**唯一新 port 的是 `step_length`**（~10 行活碼）。

新風險集中在：
1. **`diag(x^(m-2))` 在 x_i → 0 的 row-wise ill-conditioning**（NNI 專屬、HONI 無此 pattern）
2. **`(-M) \ ...` 的符號**（高風險 typo 陷阱）
3. **sparse vs dense `M` 的 pivot 差異**（可能觸發 w/y rtol fallback、如 HONI 的三 Tier）
4. **scope 擴展：`gmres` 分支新設計**（`gmres_opts` 參數、info 處理、sanity tolerance）

checklist 已列（§四陷阱 #5.1-5.10 + §五專屬 5.1-5.7 + §七 Open Q 12 題）。

**階段 B 第一步**：回答 §七 Open Questions → 寫 `NNI_with_history.m` + `generate_nni_reference.m`（spsolve 單 case）→ 實作 `python/tensor_utils.py::nni`（spsolve + gmres 雙分支、step_length helper）→ 寫 `python/test_tensor_utils.py::test_nni_basic` 等 sanity → 寫 `python/test_nni_parity.py`（spsolve 對 MATLAB）→ 寫 `python/test_nni_gmres_sanity.py`（gmres vs spsolve 近似比對）→ MATLAB 端跑 reference → Python parity + sanity → 若全過收尾、否則走 HONI 的三 Tier pattern。

**預計工作量**：~4-6 小時（比 HONI ~8 小時短、因為單層 + 依賴全備）。
