# 演算法系統 Port 現況 — Multi / HONI / NNI 三個核心演算法全綠

**最後更新**：2026-04-22（Day 4 收工）
**目的**：1 分鐘對齊三個核心演算法的 port 完整現況；不取代 PROGRESS.md 的 session 紀錄、focus 在「系統現況」。

**重構說明**：本檔由 Day 3 的 `hni_status.md` 重構而來、Day 4 加入 NNI 後調整為「三演算法並列、NNI 為主」結構（NNI 是使用者研究主軸、HNI 是另一個角度、Multi 是共用基礎）。

---

## 0. 三演算法總覽

| 演算法 | 角色 | 研究定位 | 完成日 |
|---|---|---|---|
| **NNI** | **使用者主演算法** | Nonnegative Newton Iteration — 單層 Newton + 結構化 hyperbolic step、求非負張量最大 H-eigenvalue | **Day 4** |
| HONI | 對比 / 另一角度 | H-eigenvalue Optimization via Newton Iteration — 雙層 shift-invert iteration | Day 2 末 |
| Multi | 共用基礎 | Multilinear Newton — 正解多線性系統 `A·u^(m-1) = b` 的 solver（HONI 的內層、單獨 port 為可獨立呼叫的模組） | Day 2 |

三者**共用** Layer 1+2 的 5 個 tensor 工具（`tenpow / tpv / sp_tendiag / ten2mat / sp_Jaco_Ax`）、Day 1 完成。

```
                Layer 1+2 (5 tensor utils, Day 1)
              ┌───────────┬──────────┬──────────┐
              │           │          │          │
   Layer 3   Multi      HONI       NNI        demo (Layer 4, Session 5)
             (Day 2)   (Day 2 末)  (Day 4)
             │
             └─ HONI 內層 caller
```

---

## 1. NNI —「使用者主演算法」

**Canonical 來源**：`source_code/Tensor Eigenvalue Problem/2020_HNI_Revised/NNI.m`（271 行、`HNI` 系列 2020 年整理定稿期、canonical 無 halving 版、`chit ≡ 0`）。

**演算法**：求 m-order n-dim 非負 M-tensor 的最大 H-eigenvalue + 對應非負 eigenvector。**單層 Newton**、每 iter 解一個 sparse 線性系統 `(-M_shifted) · y = x^(m-1)`、更新 eigenvector via hyperbolic step `x_new = (m-2)*x + y; normalize`。

**Python scope 擴展**（使用者 Day 3 指定）：`linear_solver` 選單
- `'spsolve'`（預設）：對應 MATLAB backslash、parity target
- `'gmres'`：Python-only、`scipy.sparse.linalg.gmres`、適合大 sparse tensor

### Q7 case parity 結果（3-Tier 框架）

test case 完全複製 Multi Q5 / HONI Q4 參數（rng=42, n=20, m=3, d∈[1,11], pert 0.01）：

| Tier | 度量類型 | 容差 | 實測結果 |
|---|---|---|---|
| **1 STRICT**（overlap slots [0:22]）| abs | 1e-10 | `x/w/y` ~1e-15 / `λ_U/λ_L/res` ≤ 1e-11（全綠） |
| **1 STRICT**（y_history） | **rel** | 1e-10 | max rel 9.5e-15（machine eps）|
| **2 APPROX**（final outputs）| abs / rel | 1e-12 / 1e-8 | `λ_U` diff **3.5e-13**；`x` rel **4.9e-17** |
| **3 INFO**（no-assert） | — | — | MATLAB 多 5 iter 於 noise floor 震盪、Python 早停 |

**PASS condition**：Tier 1 + Tier 2 全過。實測全過。

### 已知 fragility（NNI 專屬、不影響正確性）

> **Rayleigh quotient noise floor at `min(x_i) → 0`**：`temp = tpv(AA, x, m) / x^(m-1)` 的 element-wise 除法在 `min(x_i) ~ 1e-5` 時把 tpv 的 1e-16 誤差放大到 ~1e-12。公式 `noise_floor ≈ machine_eps / min(x_i)^(m-1)`。當 `tol` 和 noise floor 同量級時、停止條件在 floating-point 抽籤（Python iter 21 `res=1.1e-14` 過 tol、MATLAB iter 21 `res=1.3e-12` 未過繼續到 iter 26）。這是 Rayleigh-quotient 類演算法通用現象、**不是 port bug**。詳見 `memory/feedback_nni_rayleigh_quotient_noise_floor.md`。

### Port 檔案

- `python/tensor_utils.py::nni` — 338 行（docstring 130 + 實作 210）、`spsolve` + `gmres` 雙分支、polymorphic A、8+1 欄 history（+ `gmres_info_history`）
- `python/test_tensor_utils.py::test_nni_basic` — 10 sub-check sanity（含 gmres 分支 3 處覆蓋）
- `python/test_nni_parity.py` — 3-Tier parity 框架
- `matlab_ref/nni/NNI_with_history.m` — canonical GE 路徑 + step_length no-halving + 7 欄 history
- `matlab_ref/nni/generate_nni_reference.m` — Q7 driver
- `docs/superpowers/notes/nni_hazard_analysis.md` — 653 行階段 A 產出（commit `ab454ea`）

---

## 2. HONI —「HNI 線另一個角度」

**Canonical 來源**：`source_code/Tensor Eigenvalue Problem/2020_HNI_submitt/HONI.m`（232 行、投稿版最小子集）。

**演算法**：同樣求最大 H-eigenvalue、但用**雙層 shift-invert**結構：外層更新 `lambda_U`、內層解多線性 `(lambda_U*II - AA) · y^(m-1) = x^(m-1)` 呼叫 Multi 做 Newton iteration。**exact** / **inexact** 兩分支（inner tolerance 策略 + lambda 更新公式不同）。

### parity 結果（3-Tier 框架、Day 2 末建立）

| 分支 | y_history strict 段 max rel | y_history Tier 2 last-iter rel | 最終 λ / x |
|---|---|---|---|
| `exact` | ~5e-12 (machine eps) | **2.1e-5** | bit-identical（abs ≤ 1e-10）|
| `inexact` | up to ~5e-9 at iter 4 | **1.1e-3** | bit-identical（abs ≤ 1e-10）|

### 已知 fragility（HONI 專屬）

> **Shift-invert global near-singular**：`lambda_U → eigenvalue` 時 `(lambda_U*II - AA)` 條件數爆、內層 Multi 解的 `y` 量級從 O(1) → O(10^6)。`inexact` 分支比 `exact` ~50× 敏感（重算 vs 增量 damping 差異）。最終 `λ/x` 仍 bit-identical。詳見 `memory/feedback_honi_multi_fragility_propagation.md`。

### Port 檔案

- `python/tensor_utils.py::honi` — `exact` + `inexact` 兩分支、9 欄 history
- `python/test_honi_parity.py` — 3-Tier parity 框架（**NNI 的 tier 框架基於此延伸**）
- `matlab_ref/hni/HONI_with_history.m` + `generate_honi_reference.m`
- `docs/superpowers/notes/honi_hazard_analysis.md`

---

## 3. Multi —「共用基礎 / HONI 內層」

**Canonical 來源**：`source_code/Tensor Eigenvalue Problem/2020_HNI_submitt/Multi.m`（54 行、2021/06 定稿）。

**演算法**：求 `A · u^(m-1) = b` 的正解。外層 Newton + 內層**三等分 halving line search**（`θ /= 3`、不是 `/2`）。

### Q5 case parity 結果

| 欄位 | max_err | 語意 |
|---|---|---|
| `hal_history` | **0** | bit-identical |
| `theta_history[1:]` | **0** | bit-identical |
| `final u` | 1.11e-16 | 0.5× machine eps |
| `u_history` (2D) | 3.33e-16 | 1.5× machine eps |
| `res_history` | 4.44e-16 | 2× machine eps |
| `v_history[:, 1:]` (2D) | 6.66e-16 | 3× machine eps |

### 已知 fragility（Multi 專屬）

> **Halving path 在 m≥3 + random AA 沒有「健康 halving」sweet spot**：halving 設計為 near-optimal 微調、不是大幅 overshoot 修正。Multi 單元 parity 無法 exercise halving path（會 trap-and-diverge）、延後到 HONI integration test 自然觸發（HONI 提供 near-optimal u、是 halving 設計的 workload）。詳見 `memory/feedback_multi_halving_fragility.md`。

### Port 檔案

- `python/tensor_utils.py::multi` — 5 欄 history
- `python/test_multi_parity.py`
- `matlab_ref/hni/Multi_with_history.m` + `generate_multi_reference.m`
- `docs/superpowers/notes/multi_hazard_analysis.md`

---

## 4. Layer 1+2 共用工具（Day 1 完成、全 bit-identical 或 machine eps）

| 函式 | Python 位置 | Max parity error | 主要陷阱類型 |
|---|---|---|---|
| `tenpow` | `tensor_utils.py::tenpow` | **0** | 純 Kronecker 次方 |
| `tpv` | `tensor_utils.py::tpv` | ~1e-16 | 矩陣-向量乘 |
| `sp_tendiag` | `tensor_utils.py::sp_tendiag` | **0** | reshape `order='F'` |
| `ten2mat` | `tensor_utils.py::ten2mat` | **0** | ⭐ **Trap #4 column-major 主檢查點** |
| `sp_Jaco_Ax` | `tensor_utils.py::sp_Jaco_Ax` | ~1e-15 | 1-based 迴圈保留以對照 MATLAB exponent |

---

## 5. Parity 總表（三演算法一目了然）

| 演算法 | parity 框架 | 最終 λ / x | 中段 fields 通過 | 特有欄位 | 測試 file |
|---|---|---|---|---|---|
| **NNI** | 3-Tier（overlap + final APPROX + extra-iter INFO）| **λ abs 3.5e-13, x rel 4.9e-17** | x/w/y machine eps，λ_U/λ_L/res ≤ 1e-11 | `gmres_info_history`（gmres 分支）| `test_nni_parity.py` |
| **HONI** | 3-Tier（STRICT abs + STRICT rel + APPROX rel）| λ abs ≤ 1e-10, x ≤ 1e-10 | y_history rel ≤ 1e-8 中段；last-iter rel ≤ 1e-2 | `inner_tol_history / chit_history / innit_history` | `test_honi_parity.py` |
| **Multi** | 單 Tier（全 abs 1e-10）| machine eps | u/res/theta/hal/v 全 machine eps | `theta_history / hal_history` | `test_multi_parity.py` |

---

## 6. 三種 fragility 模式摘要（port 時的「原碼無罪」憑據）

| Fragility | 觸發者 | 表現 | 對應 memory | 未來 port 類似演算法時的 tier 策略 |
|---|---|---|---|---|
| **halving sweet spot** | Multi | m≥3 random AA 下 halving trap-and-diverge、無 healthy 區間 | `feedback_multi_halving_fragility.md` | 延後到 integration test 觸發、別硬 exercise |
| **shift-invert global** | HONI | `lambda_U → eigenvalue` 時 `(λ·II - AA)` 近奇異、`y` 量級 O(10^6) | `feedback_honi_multi_fragility_propagation.md` | y-like 欄位用 rtol、last-iter 進 APPROX tier |
| **Rayleigh quotient local** | NNI | `min(x_i) → 0` 時 `tpv/x^(m-1)` 除法放大 rounding | `feedback_nni_rayleigh_quotient_noise_floor.md` | overlap 比對 + final APPROX + extra-iter INFO |

**通則**：未來 port 新演算法前、先判斷屬於哪種 fragility 模式（或全新模式）、對應的 tier 策略直接複用。

---

## 7. 檔案地圖（三演算法完整生態）

```
matlab_ref/hni/                              HNI / Multi / HONI 的 canonical + history
├── README.md                                HNI 線 port 進度表
├── Multi.m / Multi_with_history.m           canonical + history 版
├── HONI.m / HONI_with_history.m             (同上)
├── main_Heig.m                              demo driver (Layer 4 範圍)
└── generate_{multi,honi}_reference.m        .mat 生成器

matlab_ref/nni/                              NNI 專用 (Day 4 新增)
├── NNI_with_history.m                       canonical GE 路徑 + step_length no-halving
└── generate_nni_reference.m                 Q7 driver

python/
├── tensor_utils.py                          Layer 1+2 (5 tools) + Multi + HONI + NNI
├── test_tensor_utils.py                     全體 Python-only sanity tests
├── test_{multi,honi,nni}_parity.py          三個 parity test
└── poc_iteration/
    ├── parity_utils.py                      find_divergence / report / print_neighborhood
    └── README.md

docs/
├── algorithms_status.md                     ← 本檔（Day 4 從 hni_status.md 重構）
└── superpowers/notes/
    ├── multi_hazard_analysis.md
    ├── honi_hazard_analysis.md
    └── nni_hazard_analysis.md                Session 4 階段 A 產出 (653 行)
```

---

## 8. 下一步 scope

### Session 5 選項 A：Layer 4 demo 整合

- 擴 `python/streamlit_app/demo_v0.py` 加 Multi + HONI + NNI 三個 renderer
- 特別價值：**同一 AA 跑 HNI vs NNI 對比 tile**（收斂行為視覺化、研究主軸展示）
- 工作量 2-3h（demo pattern 已驗證）

### Session 5 選項 B：NNI_ha port

- `NNI.m` canonical 是無 halving 版、但 **`Test_Heig2.m` benchmark 實際呼叫 `NNI_ha`**（halving 啟用版）
- port NNI_ha 可重現 2020 paper 的 HNI vs NNI benchmark 實驗
- 工作量 2-4h（hazard analysis 已覆蓋、只是 `halving=True` 分支 + `θ/=2` + `tol_theta=1e-12`）

### 未來方向（Session 6+）

- **Layer 4 完整**：`main_Heig.m` port（HNI paper benchmark driver、可 reproducible eigenvalue benchmark）
- **其他類別 port**：Optimization/QP（ISM）、Nonlinear Schrödinger/BEC、M-matrix Iteration、Generalized Eigenvalue（GINI）— 見 `matlab_ref/GLOBAL_INVENTORY.md` Section C
- **CI 整合**：把 Python-only sanity tests 接 GitHub Actions

---

## 9. 給未來自己 / Claude 的提示

- **三個 fragility 模式**是 port 迭代類演算法的通用 checklist — 遇到 parity 不乾淨時、先對照這三種模式判斷屬於哪一類、再選 tier 策略
- **Layer 1+2 的 5 個工具是基礎建設**：未來 port 任何 tensor eigenvalue 演算法都會大量重用（`source_code/` 這 5 個函式重複定義 549 次、見 `matlab_ref/GLOBAL_INVENTORY.md`）
- **parity tier 框架已有三版**（Multi 單 Tier、HONI 3-Tier shift-invert、NNI 3-Tier overlap）、複製最匹配的模式即可
- **Fragility 不是 bug**：三條 fragility 都是演算法數值學的本質、不是 port 錯誤。debug 前先對照 memory 判斷是否在 expected band 內
- **MATLAB-side 自我驗證**：debug 時若懷疑 Python 偏差過大、先在 MATLAB 跑「同公式不同實作」對拍（如 Multi 的 `theta=3^(-hal)` self-check）、找出原碼數值學的 expected band、再判斷 Python 是否在範圍內
