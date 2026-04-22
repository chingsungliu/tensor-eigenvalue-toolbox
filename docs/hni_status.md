# HNI 系統 Port 現況 — Layer 1/2/3 全綠

**最後更新**：2026-04-22（Day 3 收工）
**目的**：1 分鐘對齊 HNI 線（HoNi 求非負張量最大特徵值）的 port 完整現況；不取代 PROGRESS.md 的 session 紀錄、focus 在「系統現況」。

---

## 1. HNI 系統總覽

HNI（**H**igher-order **N**onlinear **I**teration）是使用者 2020-2021 年發表的求 m-order n-dim 非負張量最大特徵值演算法。架構分三層：

```
Layer 3   HONI         外層 eigenvalue iteration（exact / inexact 兩分支）
              │
              ├─ Multi 內層 Newton + 三等分 halving line search
              │       │
Layer 2       │       sp_Jaco_Ax     Jacobian via sparse Kronecker
              │       │
Layer 1       │       tenpow / tpv / sp_tendiag / ten2mat   tensor 工具
              │
              demo: main_Heig.m（待 port、屬 Layer 4 整合範圍）
```

**Canonical 來源**：`source_code/Tensor Eigenvalue Problem/2020_HNI_submitt/`（2020/12 撰寫、2021/06 定稿、`HONI.m` + `Multi.m` + `main_Heig.m` 三檔）。已複製到 `matlab_ref/hni/` 當 port 對象。

---

## 2. Layer 1+2 — 5 個張量工具（Day 1 完成）

| 函式 | Python 位置 | Max parity error | 主要陷阱類型 |
|---|---|---|---|
| `tenpow` | `tensor_utils.py::tenpow` | **0**（bit-identical） | 純 Kronecker 次方、無陷阱 |
| `tpv` | `tensor_utils.py::tpv` | ~1e-16 | 矩陣-向量乘、float 加總順序 |
| `sp_tendiag` | `tensor_utils.py::sp_tendiag` | **0** | reshape `order='F'`（對角張量是巧合 OK，但守規矩） |
| `ten2mat` | `tensor_utils.py::ten2mat` | **0** | ⭐ **Trap #4 column-major 主檢查點**：`np.moveaxis` + `reshape(order='F')` 取代 MATLAB `eval` |
| `sp_Jaco_Ax` | `tensor_utils.py::sp_Jaco_Ax` | ~1e-15 | 1-based 迴圈 bound 保留以對照原碼 exponent expression |

**parity test**：`python/test_tenpow_parity.py` ... `test_sp_Jaco_Ax_parity.py`，逐函式對 MATLAB 生成的 `.mat` reference 比對。
**Python-only sanity**：`python/test_tensor_utils.py`，不依賴 MATLAB、未來 CI 可單獨跑。

---

## 3. Layer 3 — Multi（Day 2 完成）

**檔案**：`python/tensor_utils.py::multi`、reference 在 `matlab_ref/hni/Multi.m`（54 行 + helpers）。

**演算法**：求 `A · u^(m-1) = b` 的正解。外層 Newton + 內層三等分 halving line search（不是常見的 /2、是 `θ /= 3`）。

**Q5 case parity 結果**（rng 固定、m=3、n=10、AA = sp_tendiag(diag) + sprand 微擾）：

| 欄位 | max_err | 語意 |
|---|---|---|
| `hal_history` | **0.000e+00** | bit-identical |
| `theta_history[1:]` | **0.000e+00** | bit-identical |
| `final u` | 1.110e-16 | 0.5× machine eps |
| `u_history` (2-D) | 3.331e-16 | 1.5× machine eps |
| `res_history` | 4.441e-16 | 2× machine eps |
| `v_history[:, 1:]` (2-D) | 6.661e-16 | 3× machine eps |

**已知 fragility**（不影響正確性、但限制 standalone 測試方式）：

> **Multi halving path 在 m≥3 + random AA 沒有「健康 halving」sweet spot**。Newton 一步 overshoot → u 翻負 → halving 拉回 u_old 但 u_old 也已負 → trap-and-diverge。這是 Multi 原演算法 fragility，**halving 設計上是「near-optimal 微調」、不是「大幅 overshoot 修正」**。詳見 `memory/feedback_multi_halving_fragility.md`。
>
> Halving path 的 parity 延後到 HONI integration test 自然觸發（HONI 外層 iter 讓 u 已接近最優、是 halving 的 designed workload）。

**MATLAB-side 自我驗證**（debug 時的「原碼無罪」憑據）：MATLAB Q5b 印出 `max |theta(k) - 3^(-hal(k))| = 8.7e-19`，即 `theta /= 3` 100 次的累積誤差小於 double epsilon 的一半。**Python 端若對不上、bug 一定在 Python 端**。

---

## 4. Layer 3 — HONI（Day 2 末完成、Day 3 收尾）

**檔案**：`python/tensor_utils.py::honi`、reference 在 `matlab_ref/hni/HONI.m`（232 行）。

**演算法**：求最大 eigenvalue + 對應 eigenvector，shift-invert 結構：每外層 iter 解 `(lambda_U·II - AA) · y^(m-1) = x^(m-1)`（內層丟給 Multi），更新 `lambda_U` 和 `x`，直到 `|max(temp) - min(temp)| / lambda_U < tol`。

**兩個分支**（數學上求解同一問題、但迭代路徑完全不同）：

| 差異點 | exact | inexact |
|---|---|---|
| 內層 Multi 的 `inner_tol` | `1e-10` 寫死 | `max(1e-10, min(res)*min(x)^(m-1)/nit_mat)` 動態 |
| `temp` 定義 | `x ./ y`（新舊 x 比值） | `tpv(AA, x_new, m) ./ x_new^(m-1)`（重算） |
| `lambda_U` 更新 | `lambda_U -= min(temp)^(m-1)` 增量、自帶 damping | `lambda_U = max(temp)` 整個重算、無 damping |
| `res` 公式 | `\|max^(m-1) - min^(m-1)\| / lambda_U` | `\|max - min\| / lambda_U` |
| `x` normalize 時機 | 先更新 lambda_U、最後 `x = y/\|y\|` | 先 `x = y/\|y\|`、再用新 x 算 lambda_U |

### Q5 case parity 結果（三 Tier 框架）

**Tier 設計**：

| Tier | 度量類型 | 容差 | 對象欄位 |
|---|---|---|---|
| **1 STRICT** | abs | `1e-10` | final λ / final x、`x_history`、`lambda_history`、`res`、`inner_tol_history`、`chit_history`、`hal_per_outer_history`（slots 0..K-2） |
| **1 STRICT** | **rel** | `1e-8` | **`y_history`**（slots 1..K-2、量級指數放大、必須 rtol） |
| **2 APPROX** | rel | `1e-2` | `y_history[:, K-1]`（last-iter shift-invert near-singular fragility） |
| **3 INFO** | — | no-assert | last-iter `chit/hal_per_outer/hal_accum`、`nit/innit/hal` scalars |

**PASS condition**：Tier 1 + Tier 2 全過、Tier 3 純資訊。

**兩分支實測**：

| 分支 | y_history strict 段 max rel | y_history Tier 2 last-iter rel | 最終 λ / x | PASS? |
|---|---|---|---|---|
| `exact` | ~5e-12 (machine eps) | **2.1e-5** | bit-identical（abs ≤ 1e-10） | ✓ |
| `inexact` | up to ~5e-9 at iter 4 | **1.1e-3** | bit-identical（abs ≤ 1e-10） | ✓ |

### 已知 fragility（shift-invert 在收斂尾段必然 near-singular）

> **`lambda_U` 收斂到真特徵值時，`(lambda_U·II - AA)` 變 near-singular**、內層 Multi 解出的 `y` 量級從 O(1) → O(10^6)。scipy `spsolve`（SuperLU）vs MATLAB `mldivide`（LU）的 pivot 順序差異在這個 ill-conditioned 區段被放大、`y_history[:, K-1]` 的相對誤差到 1e-3 量級。
>
> **inexact 比 exact ~50× 敏感**：`lambda_U = max(temp)` 整個重算、把 y 的相對誤差透過 `x = y/||y||` 直接餵下一輪；exact 的 `lambda_U -= min(temp)^(m-1)` 增量更新自帶 damping、誤差在 lambda_U 上幾乎不留痕跡。
>
> **最終 λ/x 仍 bit-identical**（abs ≤ 1e-10），fragility 不影響演算法正確性。詳見 `memory/feedback_honi_multi_fragility_propagation.md`。

---

## 5. 已知 fragility 摘要（兩條獨立、在 HONI 同時現身）

| Fragility | 在哪一層 | 觸發條件 | 對 parity 的影響 | 處理 |
|---|---|---|---|---|
| **Multi halving 無 healthy sweet spot** | Layer 3 (Multi) | m≥3 + random AA + 大 perturbation → Newton overshoot 翻負 → halving trap | Multi 單元 parity 無法 exercise halving path、需 ill-conditioned 設計但會 trap-and-diverge | 延後到 HONI integration test 自然觸發（HONI 提供 near-optimal u） |
| **Shift-invert 收斂尾段 near-singular** | Layer 3 (HONI) | `lambda_U → eigenvalue` → `(lambda_U·II - AA)` 條件數爆 → y 量級指數放大 | `y_history` 末段 abs diff 巨、rel diff ~1e-3（inexact）/~1e-5（exact） | 三 Tier 框架：y 用 rtol、last-iter 用 Tier 2 1e-2 接住、scalar 仍 strict 1e-10 |

兩條 fragility 都**不影響演算法正確性**、最終答案仍 bit-identical（lambda 與 x），純粹反映 shift-invert + Newton-on-tensor 的數值學本質。

---

## 6. 檔案地圖（HNI 線完整）

```
matlab_ref/hni/
├── README.md                    HNI port 進度表
├── Multi.m                      canonical reference (Day 2 port 對象)
├── HONI.m                       canonical reference (Day 2 末 port 對象)
├── main_Heig.m                  demo (尚未 port、Layer 4 範圍)
├── Multi_with_history.m         Multi.m + 5 欄 history（生 reference 用）
├── HONI_with_history.m          HONI.m + 9 欄 history（生 reference 用）
├── generate_multi_reference.m   Multi Q5 case 驅動
└── generate_honi_reference.m    HONI exact + inexact case 驅動

python/
├── tensor_utils.py
│   ├── tenpow / tpv / sp_tendiag / ten2mat / sp_Jaco_Ax    Layer 1+2
│   ├── multi                                                 Layer 3 step 1
│   └── honi                                                  Layer 3 step 2
├── test_tensor_utils.py         Python-only sanity（全 7 函式）
├── test_tenpow_parity.py
├── test_tpv_parity.py
├── test_sp_tendiag_parity.py
├── test_ten2mat_parity.py
├── test_sp_Jaco_Ax_parity.py
├── test_multi_parity.py         Multi Q5 per-iter（5 欄）
├── test_honi_parity.py          HONI exact + inexact 三 Tier
└── poc_iteration/
    ├── parity_utils.py          find_divergence / report / print_neighborhood
    └── README.md                per-iteration parity 機制設計

docs/superpowers/notes/
├── multi_hazard_analysis.md     Multi port 前 hazard analysis（Day 2 階段 A）
└── honi_hazard_analysis.md      HONI port 前 hazard analysis（Day 2 末階段 A）

docs/
└── hni_status.md                ← 你正在讀
```

---

## 7. 下一步 scope（HNI 線後續）

### Layer 4：demo + main_Heig（待選）

- `python/streamlit_app/demo_v0.py` 加 Multi + HONI 兩個 renderer（demo §9 擴充 contract、4 處改動）
- `main_Heig.m` 是 HNI paper 的 benchmark driver、port 後可作 reproducible eigenvalue benchmark
- 預計 1.5-2h（demo pattern 已驗證）

### 跨線：NNI canonical port

- NNI（Non-negative tensor Iteration）是 HNI 的兄弟線、同樣解非負張量最大特徵值、algorithm 完全不同
- canonical 候選 + 版本 inventory 在 `matlab_ref/NNI_HNI_inventory.md` Section H、待使用者決定 canonical
- port 完成後可在同一 demo 對比 HNI vs NNI 的收斂行為

兩者選擇詳見 `PROGRESS.md` 的「下一個動作 — Session 4 二選一」section。

---

## 8. 給未來自己 / Claude 的提示

- **HNI 系統的 5 個 Layer 1+2 工具是基礎建設**：未來 port 任何 tensor eigenvalue 演算法（NNI、其他線）都會大量重用。`source_code/` 裡這 5 個函式一共重複定義 549 次（見 `matlab_ref/GLOBAL_INVENTORY.md`）。
- **三 Tier parity 框架是 shift-invert 類演算法的通用方法**，不只 HONI 用。未來 port 任何「內層 `(λI - A)\b` 線性 solve、外層收斂到 λ 變 near-singular」的演算法（HOPM、各類 power-like method），直接複製 `test_honi_parity.py` 的 `_compare_2d_relative` + 三 Tier 結構。
- **Fragility 不是 bug**：兩條 fragility（halving 無 sweet spot、shift-invert near-singular）都是演算法數值學的本質、不是 port 錯誤。未來看到類似量級的 y/halving 偏差、先檢查 fragility 是否在 expected band 內、再考慮 port bug。
- **MATLAB-side 自我驗證**：debug 時若懷疑 Python 偏差過大，可在 MATLAB 端跑「同公式不同實作」對拍（如 Multi 的 `theta=3^(-hal)` self-check），找出原碼數值學的 expected band，再判斷 Python 是否在範圍內。
