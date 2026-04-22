# Multi.m — Port 前 Hazard Analysis

**日期**：2026-04-22
**對象**：`matlab_ref/hni/Multi.m`（89 行，ASCII、LF、無中文註解）
**目的**：在動手 port 之前把演算法結構、四大陷阱 + 一個新陷阱、per-iteration parity 欄位全部定位好。下一步（階段 B）才寫 Python 實作 + 比對。

---

## 一、函式簽章

MATLAB 原型：
```matlab
function [u, nit, hal] = Multi(AA, b, m, tol)
```

### 輸入

| 參數 | 型別（MATLAB） | 形狀 | 意義 |
|---|---|---|---|
| `AA` | double / sparse matrix | `(n, n^(m-1))` | m-order n-dim M-tensor 的 mode-1 unfolding（“stretching matrix”） |
| `b` | double column vector | `(n, 1)` | 多線性系統 `A·u^(m-1) = b` 的右邊；**line 15 會被就地覆寫為 `b.^(m-1)`**（見陷阱 #5 的狀態管理） |
| `m` | int（scalar） | — | Tensor order（`m >= 2` 才有意義；`m = 2` 退化成線性系統 + 線性 solver） |
| `tol` | double（scalar） | — | 外層停止門檻，相對收斂用（見 line 22） |

### 輸出

| 參數 | 型別 | 形狀 | 意義 |
|---|---|---|---|
| `u` | double column | `(n, 1)` | 多線性系統的正解 |
| `nit` | int | — | 外層 Newton 總迭代次數（停止時的 `nit` 值） |
| `hal` | double column | `(100, 1)` | 每一外層 iter 的「三等分 halving」內層次數；`hal(k) = k 次 Newton 這一步執行了幾次 theta = theta/3`。Pre-allocate 長度 100 的向量、後面用不到的位置留 0 |

**注意**：docstring（line 12）的 `hit` 是 typo — 回傳變數是 `hal`，`hit` 只是內迴圈的局部計數器。Port 時以 code 為準。

---

## 二、迭代主結構

### 初始化（line 14-21）

```
u       ← b / ||b||                          (line 14)   單位化初始解
b       ← b .^ (m-1)                         (line 15)   右邊做 element-wise 次方，就地覆寫
na      ← sqrt(||AA||_∞ · ||AA||_1)          (line 16)   MATLAB 的算子 1-norm 和 ∞-norm 幾何平均
nb      ← ||b||                              (line 17)   覆寫後 b 的 2-norm
temp    ← tpv(AA, u, m)     = A · u^(m-1)    (line 18)   當前殘差向量 = temp - b
res     ← (na+nb) · 1_100                    (line 19)   ⚠️ 預填整個 100 長度 buffer 為 na+nb
res(1)  ← ||temp - b||                       (line 20)   第一個真殘差寫進 slot 1
hal     ← 0_100                              (line 21)   halving 計數 buffer（預填 0）
nit     ← 1                                  (line 21)
```

**`res` 預填為 `na+nb` 的用意**：line 22 的 while condition 用 `min(res)`（整條 buffer 做 min），這只有在**未使用的 slots 值 ≥ 實際算出的殘差**時才等於 `min(res(1:nit))`。MATLAB 作者選 `na+nb` 當 upper bound 因為 `||temp - b|| ≤ ||temp|| + ||b|| ≤ ||AA||_？·... + nb`，實際上 `na + nb` 往往比初始殘差大，所以 `min` 不會被雜訊觸發。**Port 時兩種選擇都要注意**：
- （A）嚴格復刻：預填 `(na+nb)*np.ones(100)`、用 `np.min(res)`
- （B）語意替換：用 `np.min(res[:nit])`、無 buffer

兩者在**數學結果上應一致**（只要 `na+nb` 真的比實際殘差都大），但若要 bit-identical parity，應選 (A)。選 (B) 要先驗證 min 等價、之後再 parity。**建議：階段 B 選 (A)，對得上就收工；若對不上再回頭查**。

### 外層 Newton 迴圈（line 22-52）

```
while  min(res) > tol · (na · ||u|| + nb)  AND  nit < 100         (line 22)
    nit ← nit + 1                                                 (line 24)

    # --- 1. Jacobian + 線性求解 -----------------------------
    M   ← sp_Jaco_Ax(AA, u, m) / (m-1)                            (line 27)
    v   ← M \ b                                                   (line 28)

    # --- 2. 試算 θ = 1 的 Newton step ------------------------
    θ          ← 1                                                (line 31)
    tol_θ      ← 1e-14                                            (line 32)
    u_old, v_old ← u, v                                           (line 33)
    u          ← (1 - θ/(m-1))·u_old + θ·v_old/(m-1)              (line 34, 注意 u_old/v_old)
    temp       ← tpv(AA, u, m)                                    (line 35)
    res(nit)   ← ||temp - b||                                     (line 36)
    hit        ← 0                                                (line 37)

    # --- 3. 三等分 halving line search ----------------------
    while  res(nit) - res(nit-1) > 0   OR   min(temp) < 0         (line 39)
        θ        ← θ / 3            ⚠️ 除以 3，不是 2!              (line 40)
        u        ← (1 - θ/(m-1))·u_old + θ·v_old/(m-1)            (line 41)
        temp     ← tpv(AA, u, m)                                  (line 42)
        res(nit) ← ||temp - b||                                   (line 43-44)
        hit      ← hit + 1                                        (line 45)
        if θ < tol_θ                                              (line 46)
            fprintf('Can''t find a suitible step length ...')     (line 47; typo 保留)
            break                                                 (line 48)
        end
    end
    hal(nit) ← hit                                                (line 51)
end
```

### 收斂判準（3 層）

| 位置 | 條件 | 語意 |
|---|---|---|
| 外層 while（line 22） | `min(res) > tol·(na·‖u‖ + nb)` | 相對殘差未達門檻 |
| 外層 while（line 22） | `nit < 100` | 硬上限，防止無限迴圈 |
| 內層 while（line 39） | `res(nit) - res(nit-1) > 0` | 此步殘差**上升** — Newton step 不下降 |
| 內層 while（line 39） | `min(temp) < 0` | `A·u^(m-1)` 有負分量 — 違反「正解」要求 |
| 內層 break（line 46） | `θ < 1e-14` | halving 已到浮點底、放棄此步 |

**演算法關鍵不變量**：解多線性系統 `A·u^(m-1) = b` 要求 u > 0，所以每步都用 halving 保證 `temp = A·u^(m-1)` 各分量為正。如果一步降不下去就把 θ 切 3 等分、直到殘差下降且 temp ≥ 0。

---

## 三、呼叫了哪些函式 — 已 port 狀態

| MATLAB 呼叫（Multi.m 內） | MATLAB 定義位置 | 已 port 到 Python | 狀態 |
|---|---|---|---|
| `tpv(AA, u, m)` | Multi.m line 60-64（內嵌定義） | `tensor_utils.py::tpv` | ✅ Layer 1 完成，`max_err ~1e-16` |
| `sp_Jaco_Ax(AA, u, m)` | Multi.m line 79-87（內嵌定義） | `tensor_utils.py::sp_Jaco_Ax` | ✅ Layer 2 完成，`max_err ~1e-15` |
| `tenpow(x, p)`（`tpv` / `sp_Jaco_Ax` 間接呼叫） | Multi.m line 66-76 | `tensor_utils.py::tenpow` | ✅ Layer 1 完成，bit-identical |
| `norm(AA, 'inf')` / `norm(AA, 1)` | MATLAB built-in | `np.linalg.norm` (dense) / `scipy.sparse.linalg.norm` (sparse) | — |
| `norm(b)` | MATLAB built-in（2-norm） | `np.linalg.norm(b)` | — |
| `b.^(m-1)` | MATLAB element-wise | `b ** (m - 1)` | — |
| `ones(100,1)` / `zeros(100,1)` | MATLAB built-in | `np.ones(100)` / `np.zeros(100)` | — |
| `M \ b` | MATLAB 左除 | `scipy.sparse.linalg.spsolve(M, b)` | — |
| `min(...)` | MATLAB min | `np.min(...)` 或 `.min()` | — |

**重點**：Multi.m 用到的**張量運算全部已 port 完畢**。階段 B 只剩下 Newton + halving 控制流 + 三種 norm + 一個 sparse solve，**沒有新的 column-major 陷阱**。

---

## 四、四大陷阱逐行分析

### 陷阱 #1：邊界處理
**N/A** — Multi 不是 filter / convolution，沒有邊界概念。

### 陷阱 #2：濾波器 / 核截斷寬度
**N/A** — 沒有 kernel。

### 陷阱 #3：1-based vs 0-based indexing

| 行號 | MATLAB 原碼 | 是否觸發 | 處置 |
|---|---|---|---|
| 19-21 | `res = (na+nb)*ones(100,1); res(1) = …; hal = zeros(100,1); nit = 1;` | ⚠️ **是**。MATLAB `res(1)` 是第一格；Python `res[0]` 才是。`nit = 1` 的語意是「MATLAB 第 1 格已填好」— Python 要改成 `nit = 0`（表示「Python 第 0 格已填好」），**或**保留 `nit = 1` 但每次 index 都用 `res[nit - 1]`（保留 MATLAB 閱讀對應）。**建議：用 `nit = 0`，改寫 loop/index 成 Python 慣例**（與 POC `sqrt2_newton` 的 `x_history[k]` 一致）。 |
| 22 | `while min(res) > ... && nit < 100` | 低 | 上限 100 是大小，不是 index，Python 直接抄。 |
| 24 | `nit = nit + 1` | 低 | Python 同步遞增。 |
| 36 | `res(nit) = norm(temp - b)` | 中 | MATLAB `res(nit)` → Python `res[nit]`（若選 `nit = 0` 起算）或 `res[nit - 1]`（若保留 1-based）。 |
| 39 | `res(nit) - res(nit-1) > 0` | 中 | 兩個 index 都要同步減 1（若改 0-based）或保持一致（若留 1-based）。**同一個函式不能混用**。 |
| 51 | `hal(nit) = hit` | 中 | 同上。 |

**結論**：indexing 是這支函式最容易手殘的地方。**決策**：統一用 Python 0-based（`nit = 0` 起算、`res[nit]` 而非 `res[nit-1]`），不保留 MATLAB 1-based 語意。理由：
- Multi.m 裡的 index 算術單純（`nit` 和 `nit-1` 而已），沒有像 `sp_Jaco_Ax` 的 `i-1` / `p-i` 那種公式對應風險
- 外面 `u_history` / `res_history` 用 0-based 更接近 POC `sqrt2_newton` 的 pattern（`x_history[0]` 為初值）
- 唯一要留的 1-based 借鑑：**第 0 格存「初始化完成後的狀態」**（`u_history[0] = u_init`、`res_history[0] = ||tpv(AA, u_init, m) - b||`），`nit = 0` 表示「已初始化、還沒進外層 loop」。對應 MATLAB 的 `nit = 1`、`res(1) = …`。

### 陷阱 #4：Column-major vs Row-major

**本函式不觸發**。Multi.m 沒有 `reshape`，全部 column-major 風險**都在已 port 完的 `tpv` / `sp_Jaco_Ax` / `ten2mat` 內部處理好了**。

唯一需要注意的**間接風險**：`b` 作為輸入向量。MATLAB 習慣 column vector `(n, 1)`；Python 端遵循 `tensor_utils.py` 的慣例用 1-D `(n,)`。輸入 contract 要明示「b 必須 1-D」、不接受 2-D column。

### 陷阱 #5（本次新增）：halving line search 的計數與狀態管理 + sparse solve 的 dtype/order

這條是 README C4 沒單獨列出的新陷阱，針對 Multi.m 的迭代控制流細節。

| 子項 | 行號 | 內容 | 風險 |
|---|---|---|---|
| **5.1 三等分 `/3` 寫成 `/2`** | 40 | `theta = theta/3` | ⭐ **高**。註解和 docstring 都講 “one-third procedure”，這跟一般「halving line search」的 `/2` 不同。`/2` 也能收斂但路徑不同、parity 會發散。**port 時手眼要同步核對**。 |
| **5.2 halving 用 `u_old, v_old` 而非 `u, v`** | 41 | `u = (1-theta/(m-1))*u_old + theta * v_old/(m-1)` | ⭐ **高**。line 34 的 Newton step 用的是 `u`（舊 u）跟 `v`；**halving 進到內迴圈之後所有的 recompute 都用 line 33 凍結的 `u_old` / `v_old`，不是當前 `u` / `v`**。若誤寫成「以當前 u / v 當基礎重算」，相當於每次 halving 都從上一次 halving 結果再降，數學上不同、parity 絕對發散。line 33 的「state snapshot」是關鍵。 |
| **5.3 `res(nit)` 在外層被寫兩次（line 36、line 44）** | 36, 44 | 外層先寫一次（θ=1 試算），halving 每次再覆寫 | 中。正確：`res_history[nit]` 最終存的是 halving 結束後的值（若有 halving）或 θ=1 的值（若 halving 未觸發）。**中間的 θ=1 試算值不保留**。要確認 history 欄位的語意。 |
| **5.4 `min(temp) < 0` 的 short-circuit 順序** | 39 | `res(nit) - res(nit-1) > 0 \|\| min(temp) < 0` | 低-中。MATLAB `\|\|` 是 short-circuit OR：若殘差已下降但 `min(temp) < 0` 仍要 halving。Python `or` 同為 short-circuit。兩邊一致，但**運算順序不能換**（不能寫 `min(temp) < 0 or res[nit] > res[nit-1]`，否則在 `min(temp) < 0` 為 True 時不 evaluate 殘差比較；數學結果雖然相同，但執行路徑不同、不利 debug）。**建議：逐字照抄順序**。 |
| **5.5 Pre-allocate res/hal buffer 100** | 19, 21 | `(na+nb)*ones(100,1)` / `zeros(100,1)` | 中。選 (A) 嚴格復刻 vs (B) dynamic list — 見「二、初始化」的分析。**決策：階段 B 先用 (A)**，搭配 `res = np.full(100, na + nb)` 和 `hal = np.zeros(100)`，對 Python 慣例有點怪但 parity 最保險。 |
| **5.6 硬上限 100 次迭代** | 22 | `nit < 100` | 中。如果剛好在第 100 次收斂，MATLAB 會跑第 100 步；port 要保留同樣邊界條件（`nit < 100`，不是 `<= 100`）。 |
| **5.7 `M = sp_Jaco_Ax(...) / (m-1)` 的 sparse 格式** | 27 | csr / csc 對 spsolve 的影響 | 低-中。`sp_Jaco_Ax` 回傳 `csr_matrix`（已驗證）；`csr / scalar` 仍是 csr。`scipy.sparse.linalg.spsolve(M_csr, b)` 會自動轉 csc，可能有 warning 但結果正確。建議：傳進 spsolve 前先 `.tocsc()` 顯式轉換，避免 warning。 |
| **5.8 `v = M\b` 的 dtype** | 28 | float64 v.s. float32 | 低。AA 是 float64（MATLAB default）、b 是 float64、M 繼承 float64、v 也是 float64。Python 端只要不中途 cast 就對齊。但**`b ** (m-1)` 若 b 是 integer array，Python 會返回 integer**（m=3 時 `b**2` 仍 int）— 這是一個 silent bug 源頭。解方：port 的最開頭做 `b = np.asarray(b, dtype=np.float64)`。 |
| **5.9 norm(AA, 'inf') / norm(AA, 1) 的 sparse 分派** | 16 | MATLAB `norm` 自動分派 dense/sparse | 中。Python 端 **dense 用 `np.linalg.norm(AA, ord=np.inf)` / `ord=1`；sparse 用 `scipy.sparse.linalg.norm(AA, ord=np.inf)`**。若寫成 `np.linalg.norm(AA_sparse, ...)` 會 raise。建議 port 的 helper：`def _matrix_norm(AA, ord): ...` 統一派發。 |
| **5.10 `norm(AA, 'inf')` 對非方陣的意義** | 16 | AA 是 `(n, n^(m-1))` 長方陣 | 確認。MATLAB 的 `norm(A, 'inf')` 對非方陣就是「最大 row 絕對值和」；numpy / scipy 同義。不是運算子範數的特殊定義（方陣的 spectral radius 是 `norm(A)` 不是 `norm(A, inf)`）。OK。 |

**結論**：陷阱 #5 的風險**集中在控制流細節**，而不是數學計算。階段 B 的 code review checklist 必然包含：
- [ ] `theta /= 3` 不是 `/= 2`
- [ ] 內層 halving 用 `u_old, v_old`（已 snapshot）
- [ ] `b = np.asarray(b, dtype=np.float64)` 在函式開頭
- [ ] `M = sp_Jaco_Ax(...).tocsc() / (m - 1)` 或解前顯式 `.tocsc()`
- [ ] Pre-allocate buffer 100，語意比照 MATLAB
- [ ] `or` 子句順序不換

---

## 五、Per-iteration parity — 要存的 history 欄位

依照 POC `sqrt2_newton` 的樣板（`x_history[0] = x0`、`x_history[k+1]` = 第 k+1 iter 後的值），Multi 的 history 設計：

| 欄位名 | 形狀 | 意義 | `[0]` 位置存什麼 | 為什麼需要 |
|---|---|---|---|---|
| `u_history` | `(n, nit + 1)` 或 list of `(n,)` | 每個外層 iter 結束後的 `u` | 初始化後（line 14）的 `u = b / ‖b‖` | 主 state；收斂路徑的核心。**用 2-D `(n, nit+1)` 方便在 Streamlit 畫 heatmap；但逐 iter 比對時可能更偏好 1-D list of arrays**。先用 2-D，MATLAB reference 好存 |
| `res_history` | `(nit + 1,)` | 每個外層 iter 結束後的 `‖temp - b‖`（halving 結束後的定版） | line 20 的 `‖tpv(AA, u_init, m) - b‖` | scalar，最方便定位發散點 |
| `theta_history` | `(nit + 1,)` | 每個 iter 最終用到的 θ（halving 結束後） | `NaN` 或 `1.0`（iter 0 無 Newton step） | ⭐ **關鍵**。halving 發生了幾次決定 θ 的值；若 Python 和 MATLAB 的 θ 發散，表示 halving 進入次數不同、殘差路徑接下來會整個岔開 |
| `hal_history` | `(nit + 1,)` | 每個 iter 的 halving 次數（= `hit`） | `0` | 對應 MATLAB `hal(nit)`。跟 `theta_history` 互補：`theta = 1/3^hal` 的期望關係成立才是 consistent |
| `v_history` | `(n, nit + 1)` | 每個 iter 的 `v = M \ b`（halving 不動 v） | `NaN` vector（iter 0 無 solve） | ⭐ **診斷用**。若 v_history 從某 iter 開始發散、但該 iter 的 u_old 在 MATLAB/Python 仍一致，表示是 sparse solve（`M\b` vs `spsolve`）精度差異造成；不是 halving 邏輯錯 |
| `M_history`（optional） | list of `csr_matrix` | 每個 iter 的 `M = sp_Jaco_Ax(AA, u_old, m)/(m-1)` | — | 開發期用；正式 parity 不需要（`sp_Jaco_Ax` 本身的 parity 已在 Layer 2 驗過） |
| `temp_history`（optional） | `(n, nit + 1)` | 最終 `temp = tpv(AA, u, m)`（halving 結束後） | line 18 的 `tpv(AA, u_init, m)` | 若 `min(temp) < 0` 是 halving 觸發原因，看 temp 比看 res 多一層資訊；但多數情況 `res_history` + `hal_history` 已足 |

**建議**：**階段 B 先存** `u_history`、`res_history`、`theta_history`、`hal_history`、`v_history` 這 5 欄。`M_history` 和 `temp_history` 等 parity 跑出發散後再加（避免一開始就 over-engineering）。

**長度慣例**：所有 history 長度一致為 `nit_final + 1`。MATLAB reference 用 pre-alloc 長度 100、最後 `u_history = u_history(:, 1:nit_final + 1)` 截斷；Python 端在函式返回前同樣截。

### MATLAB reference 生成腳本草案（階段 B 會實作、這裡只記 spec）

- 檔名：`matlab_ref/hni/generate_multi_reference.m`
- 輸入 seed 與規模：`rng(42); m = 3; n = 20;`（對齊 `main_Heig.m` 的規模；但 Multi 要自己設 `AA` 和 `b`，不借 `main_Heig` 的 Q）
- 步驟：
  1. 隨機產生 M-tensor 的 unfolding `AA`（非零子對角分量需要讓系統有正解 — 可以先從 `sp_tendiag(d, m) + sparse perturbation` 起手）
  2. `b = abs(rand(n, 1))`（保證正）
  3. 改寫 `Multi.m` 的副本（或新檔）：加 `history` output（`[u, nit, hal, u_hist, res_hist, theta_hist, hal_hist, v_hist]`）
  4. `save('multi_reference.mat', 'AA', 'b', 'm', 'tol', 'u', 'nit', 'hal', 'u_hist', 'res_hist', 'theta_hist', 'hal_hist', 'v_hist')`
- 注意：不能直接改 `Multi.m` 原檔（保持 canonical 乾淨），應 copy 成 `Multi_with_history.m`

---

## 六、Port 前 Open Questions（需要使用者決策）

1. **buffer 策略**：pre-alloc 100 vs dynamic list — **建議 (A) pre-alloc**，與 MATLAB 逐字對應。
2. **indexing base**：完全 0-based vs 保留 MATLAB 1-based — **建議 0-based**，理由見陷阱 #3 結論。
3. **sparse format for `M`**：spsolve 前是否顯式 `.tocsc()` — **建議明寫 `.tocsc()`**，避免 warning、無數值差異。
4. **`b` mutation**：MATLAB 原碼 line 15 就地覆寫 `b`。Python port 要不要 mutate 呼叫端傳入的 array？**建議不 mutate** — 在函式內部用 `b_raised = b ** (m - 1)`，之後全部用 `b_raised`（不 shadow 原名，避免 bug）。
5. **MATLAB reference tensor 選擇**：是要找「已知有正解」的 M-tensor（避免 halving 收不斂）還是用隨機？**建議先用已知穩定的問題**（例如從 `sp_tendiag` 造 diagonally dominant M-tensor + 小 perturbation），先確認整條 parity pipeline 通；若要測 halving 觸發路徑再換難一點的 test case。

這 5 個問題會在階段 B 第 1 步先定，然後才動 Python 碼。

---

## 七、總結：階段 B 的起跑線

- 演算法結構已理清、每條陷阱已定位、history 欄位已定 spec
- 呼叫的 5 個張量工具（tenpow / tpv / sp_tendiag / ten2mat / sp_Jaco_Ax）**全部已 port 完成且 parity 通過**
- 本次 port **沒有新的 column-major 風險**（主戰場在 Layer 2 已 cleared）
- 新增陷阱 #5（halving + state + sparse solve）是控制流細節 — checklist 化處理
- 階段 B 第一步：回答 open questions → 寫 MATLAB reference 生成腳本 → 實作 `python/multi.py` → parity test
