# Per-iteration Parity POC（Layer 3 迭代對帳機制驗證）

## 為什麼有這個 POC

Phase D Layer 3 要 port 的 `Multi` 和 `HONI` 是**迭代演算法**。跟前兩層的 5 個純函式不同：

- **純函式**（tenpow / tpv / sp_tendiag / ten2mat / sp_Jaco_Ax）：同樣輸入進去、一次計算、出來結果。parity 就是「比這一個輸出」。Layer 1-2 已經完成。
- **迭代演算法**（Multi、HONI、NNI 都算）：同樣輸入進去、跑 N 次迴圈、每步內部狀態 `(u_k, res_k, θ_k, ...)` 都可能偏離 MATLAB 路徑。**兩邊最終都可能「收斂」到相同解，但中途路徑完全不同**。這種 bug 用單點 parity（只比最終輸出）**完全抓不到**。

所以 Layer 3 前必須先設計好「**逐 iteration 比對 + 分岔點報告**」的機制。這個 POC 用最小可驗證的迭代演算法（Newton's method 求 √2）把機制跑一遍，確保：

1. MATLAB reference 存下每步 state 的格式可以直接被 Python 讀取
2. 比對程式不只報 `max_err`，還能定位「**第一個超過 tolerance 的 iteration**」
3. 展示分岔點附近前後 2 步的值，讓 debug 有上下文

機制通過後，Multi / HONI / NNI 就照這個樣板 port。

## 實際學到什麼

### 一、MATLAB reference 存檔格式

用**兩個 1-D array**，長度 = `n_iter + 1`（第 0 項是初始狀態 x_0）：

```matlab
x_history(1) = x0;                        % index 1 = iter 0
res_history(1) = x0^2 - 2;

for k = 1:n_iter
    x = x - (x^2 - 2) / (2 * x);
    x_history(k + 1) = x;                 % index k+1 = iter k
    res_history(k + 1) = x^2 - 2;
end

save('sqrt2_newton_reference.mat', 'x_history', 'res_history');
```

**重點**：**`initial state` 要納入 history**（第 0 個元素）。這樣 parity test 可以從「iteration 0 的初始值是不是一樣」開始查，若初始值已經差就是 Python caller 傳錯 seed/x_0、不是演算法問題。

Layer 3 的 `Multi` 會存 `u_history`、`res_history`、`theta_history`、`hal_history`；`HONI` 還多 `lambda_history`、`innit_history`。全部照這個 pattern。

### 二、Python parity test 的 `find_divergence` API

核心函式（見 `python/poc_iteration/test_sqrt2_newton_parity.py`）：

```python
def find_divergence(matlab_seq, python_seq, tolerance, name=""):
    """回傳 dict，含：
    - passed: bool
    - max_err, max_err_iter
    - first_bad_iter: 第一個超過 tolerance 的 iter（None if passed）
    - matlab_val, python_val, diff_at_first_bad
    """
```

關鍵：**不只回 `max_err`，也回 `first_bad_iter`**。這是整個 POC 要驗證的設計重點。

```python
def report(result):
    """PASS 輸出一行 max_err；FAIL 輸出 first_bad_iter + MATLAB/Python 值 + diff。"""

def print_neighborhood(matlab, python, center, radius=2):
    """在 center 前後各 radius 步列值，讓使用者看「分岔前後路徑長什麼樣」。"""
```

失敗時的輸出範例（不是實際發生，僅示意）：
```
[x_history] FAIL  first divergence at iteration 3
        MATLAB value = 1.500000000000000e+00
        Python value = 1.500000000000001e+00
        diff at that iter = 1.000e-15  (tolerance 1e-10)

--- x_history 分岔鄰近 (iteration 3 前後各 2 步) ---
  iter              MATLAB              Python         diff
     1  1.500000000000000  1.500000000000000  0.000e+00
     2  1.416666666666666  1.416666666666666  0.000e+00
     3  1.500000000000000  1.500000000000001  1.000e-15  <-- DIVERGE
     4  ...                ...                ...
     5  ...                ...                ...
```

### 三、Tolerance 設定

POC 用 **`TOLERANCE = 1e-10`**。實際對 sqrt(2) Newton 的結果：
- MATLAB 和 Python 每一步 `max_err = 0.000e+00`（bit-identical）
- 因為兩邊都是 IEEE 754 float64、從同一個初始值、同樣的基本四則運算
- 所以分岔「不存在」— 通過門檻比實際誤差寬 `10^10` 倍都沒問題

**Layer 3 的 tolerance 建議**：也用 1e-10。迭代中途若有 matrix solve（`M \ b`）或 kron 鏈產生 ~1e-14 量級的 floating-point noise，可能某個 iter 的差會超過 0、但應該還在 1e-10 以內。若真的某步超過 1e-10 就是**真正的分岔**、需要 debug。

### 四、本 POC 本身的結果

10 次 Newton iteration 從 `x_0 = 1.0` 收斂到：
```
x_10   = 1.41421356237309492343   （MATLAB 和 Python 完全一致）
target = 1.41421356237309514547   （真 sqrt(2)，差 1 ulp ≈ 2.2e-16）
res_10 = -4.441e-16               （Newton 在 float64 的最後極限）
```

所以 `sqrt2_newton` **沒有真的解出 sqrt(2)**（差 1 ulp），但 MATLAB 和 Python 的迭代路徑**完全一致**（bit-exact）— 這才是本 POC 要驗證的。

## 這個 POC 在 Layer 3 Port 時的用法

**Port Multi / HONI / NNI 時複製這個樣板**：

1. 寫 MATLAB reference（例如 `matlab_ref/hni/generate_Multi_reference.m`），讓原函式在每步把 `u, res, theta, hal` 存進 history array，最後 `save` 成 `.mat`
2. Python 端的 `multi.py` 也每步存 history（可以加個 `record_history=False` 參數，預設不存、parity test 時打開）
3. Parity test 呼叫 `find_divergence` 對每個 history field 各做一次、用 `report` 印摘要、有失敗用 `print_neighborhood` 顯示上下文
4. Tolerance 1e-10，期望結果 ~1e-14 到 1e-16（Layer 1 的 tpv 通過到 1e-16 可以當 benchmark）

## 延伸：若將來做 NNI / HONI 時分岔真的發生

這個 POC 的目標就是在**還沒有真實分岔發生前**把框架建好。真實 port 時如果分岔了：

1. `find_divergence` 報 first_bad_iter
2. `print_neighborhood` 顯示分岔前後值
3. Debug 切到「為什麼 iter k 會差」：檢查 Python 端在 iter k 計算 u_k 時用到的所有中間 sparse kron 結果是否跟 MATLAB 一致
4. 常見原因：Kronecker 順序、sparse matrix 類型不一致（CSR vs CSC 可能影響 matmul 精度）、halving procedure 步長不同
5. 不用擔心後面的 iteration — first_bad_iter 是根源，後面都是受害者
