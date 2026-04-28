"""HNI / NNI 共用的張量工具函式。

本模組的公用函式遵循以下 port 慣例：

- **向量 shape**：輸入/輸出一律 1-D numpy array `(n,)`，不使用 `(n,1)` column
  vector。呼叫端如有 2-D 意圖請用明確的 `reshape(-1, 1)` 或 `np.outer` 表達。
- **Indexing**：Python 介面採 0-based（Python 自然慣例）。MATLAB 的 1-based
  語義只在 `matlab_compat=True` 路徑內部還原。
- **Column-major 對齊**：任何 `reshape` 呼叫只要在 parity 路徑上，一律顯式
  `order='F'`，匹配 MATLAB 的欄優先儲存。
"""
import warnings

import numpy as np
from scipy.sparse import (
    csr_matrix,
    diags as sp_diags,
    eye as sp_eye,
    issparse,
    kron as sp_kron,
)
from scipy.sparse.linalg import gmres, norm as sparse_norm, spsolve


def tenpow(x, p, matlab_compat=False):
    """計算向量 x 的 p 階 Kronecker 次方：x ⊗ x ⊗ ... ⊗ x（p 個 x 相 kron）。

    對應 MATLAB reference：`matlab_ref/hni/HONI.m` 第 129-139 行（同樣的定義在
    `Multi.m` 第 66-76 行也重複一次；MATLAB 原碼的參數名也是 `p`，本 port
    刻意保留同名）。

    Parameters
    ----------
    x : array-like, 1-D
        輸入向量，shape `(n,)`。2-D 輸入會 raise ValueError（強制 1-D 慣例以
        早點抓到呼叫端 bug）。
    p : int
        Kronecker power count — 總共要把幾個 x 做 Kronecker product。
        - `p = 0`：回傳 `np.array([1.0])`（長度 1 的 neutral element，對應
          MATLAB 原碼的 scalar `1`；用 1-D shape 以符合本模組的 shape 慣例)
        - `p = 1`：回傳 x（copy）
        - `p >= 2`：回傳 `x ⊗ x ⊗ ... ⊗ x`（p 個 x 左結合 Kronecker product）
        必須 `>= 0`；負值會 raise ValueError。

        註：在 HNI 的呼叫點會看到 `tenpow(x, m - 1)`，那個外層的 `m` 是 tensor
        order，傳進來的值 `m - 1` 才是本函式的 `p` 參數（Kronecker power）。
        刻意用 `p` 而不是 `m`，跟 MATLAB 原碼一致、並避免跨層同名。
    matlab_compat : bool, default False
        **對 tenpow 無作用**（no-op）。保留以維持 port 模組的統一 API。原因：
        numpy `np.kron` 對 1-D input 的輸出元素順序跟 MATLAB `kron` 逐元素
        一致，不需要切換任何行為。

    Returns
    -------
    np.ndarray, 1-D
        `p >= 1` 時 shape 為 `(n**p,)`；`p == 0` 時 shape 為 `(1,)`。
        dtype 繼承自輸入 `x`。

    四大陷阱在本函式的觸發分析
    -------------------------
    Trap #1 邊界：N/A（不是 filter）
    Trap #2 截斷：N/A
    Trap #3 Indexing：N/A — 迴圈次數在 MATLAB `1:p-1` 和 Python `range(p-1)`
        都是 `p-1` 次，元素訪問沒有 off-by-one 風險。
    Trap #4 Column-major：N/A — `np.kron` 對兩個 1-D array 輸出的元素順序
        跟 MATLAB `kron` 一致（都是 `a[0]*b, a[1]*b, ...` 的外積展平）。
        本函式不涉及任何 reshape，因此 column/row-major 無從觸發。

    Examples
    --------
    >>> tenpow(np.array([1.0, 2.0]), 2)
    array([1., 2., 2., 4.])
    >>> tenpow(np.array([1.0, 2.0]), 3)
    array([1., 2., 2., 4., 2., 4., 4., 8.])
    >>> tenpow(np.array([7.0]), 0)
    array([1.])
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")

    if p == 0:
        return np.array([1.0], dtype=x.dtype)

    result = x.copy()
    for _ in range(p - 1):
        result = np.kron(x, result)
    return result


def tpv(AA, x, m, matlab_compat=False):
    """計算 m-order tensor 的 tensor-vector product A·x^(m-1)。

    對應 MATLAB reference：`matlab_ref/hni/HONI.m` 第 123-127 行（同樣的定義在
    `Multi.m` 第 60-64 行重複一次）。

    Parameters
    ----------
    AA : array-like or scipy.sparse matrix, 2-D
        m-order n-dim tensor 的 mode-1 unfolding，shape `(n, n^(m-1))`。dense
        (numpy.ndarray) 或 sparse (scipy.sparse.*_matrix) 都可以。
    x : array-like, 1-D
        向量，shape `(n,)`。
    m : int
        Tensor order；會用來算 Kronecker 次方 `x^(m-1)`。必須 `m >= 1`。
        注意這個 `m` 是 tensor order（跟 HNI 的 m 同義），**不是** tenpow
        的 `p`;本函式內部做 `tenpow(x, m - 1)`，把 tensor order 換成
        Kronecker 次方傳給 tenpow。
    matlab_compat : bool, default False
        **對 tpv 本身是 no-op** — 矩陣-向量乘法在 MATLAB 和 numpy 產生等價
        的浮點加總順序。底層會把 flag 一併傳進 tenpow。保留以維持 API 一致。

    Returns
    -------
    np.ndarray, 1-D
        Shape `(n,)`。即使 AA 是 sparse matrix，回傳的也是 1-D dense ndarray。

    四大陷阱在本函式的觸發分析
    -------------------------
    Trap #3 Indexing：N/A — 沒有逐元素索引。
    Trap #4 Column-major：**本函式不觸發，但 AA 參數會**。矩陣-向量乘法本身
        是 layout-free 的（結果跟儲存順序無關），但如果呼叫端自己把 m-order
        tensor 攤成 AA 時 reshape 順序錯，`AA @ x^(m-1)` 雖然 matmul 數值
        上正確，但在原 tensor 語義下就是錯的。這是「trap 在 caller 不在
        callee」的案例 — tpv 自己沒事，完全依賴上游給對的 AA。
    """
    if not issparse(AA):
        AA = np.asarray(AA)
    if AA.ndim != 2:
        raise ValueError(f"AA must be 2-D, got shape {AA.shape}")

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    x_m = tenpow(x, m - 1, matlab_compat=matlab_compat)
    y = AA @ x_m
    # sparse×1-D 的 return 型別依 scipy 版本可能為 np.matrix 或 2-D ndarray；
    # 一律 ravel 回 1-D 以維持 shape 慣例。
    return np.asarray(y).ravel()


def sp_tendiag(d, m, matlab_compat=False):
    """構造 m-order n-dim 對角張量的 mode-1 unfolding（scipy.sparse.csr_matrix,
    shape `(n, n^(m-1))`）。

    對應 MATLAB reference：`matlab_ref/hni/HONI.m` 第 142-149 行。

    「對角張量」指對於 m-order n-dim tensor T，只有 T(i, i, ..., i) = d[i] 非零
    （i = 0..n-1），其他位置全部為 0。本函式直接回傳它的 mode-1 unfolding。

    Parameters
    ----------
    d : array-like, 1-D
        對角元素向量，shape `(n,)`。
    m : int
        Tensor order。必須 `m >= 1`。`m = 1` 邊界時輸出 shape 為 `(n, 1)`，
        即 d 作為 column（因為 n^0 = 1）。
    matlab_compat : bool, default False
        **對輸出結果無作用** — 本函式永遠使用 `reshape(..., order='F')` 匹配
        MATLAB column-major。保留 flag 以維持 API 一致性。

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape `(n, n^(m-1))`，有 n 個非零元素，位於 `D[i, i * stride]`，其中
        `stride = (n^(m-1) - 1) / (n - 1)`（n=1 時 stride=0；m=1 時 stride=0）。

    四大陷阱在本函式的觸發分析
    -------------------------
    Trap #3 Indexing：MATLAB `linspace(1, n^m, n)` 給 1-based 整數索引，Python
        用 `np.linspace(0, n**m - 1, n).astype(int)` 對應到 0-based。
    Trap #4 Column-major：⭐ **會觸發，但已顯式處理。** MATLAB 的
        `reshape(D, n, n^(m-1))` 是 column-major。本函式在 scatter 之後用
        `reshape(..., order='F')` 還原相同的 reshape。

        **註**：對於「對角張量」這個特例，column-major 和 row-major 數學上
        湊巧會把 `d[i]` 放到同一個 (row, col) 位置（因為對角元素在 1-D 中是
        等距排列的特殊結構）。就算寫成 `order='C'` parity 也會過 — 但這是
        巧合，不是設計規則，也不適用於別的 reshape 場景。本函式堅持寫
        `order='F'` 以維持「reshape 一律 F」的模組公約。
    """
    d = np.asarray(d)
    if d.ndim != 1:
        raise ValueError(f"d must be 1-D, got shape {d.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    n = len(d)
    total = n ** m
    num_cols = n ** (m - 1)

    # Scatter d into a dense 1-D vector at the diagonal linear indices.
    # MATLAB 1-based `linspace(1, n^m, n)` 對應 Python 0-based 如下：
    D_vec = np.zeros(total, dtype=d.dtype)
    positions = np.linspace(0, total - 1, n).astype(int)
    D_vec[positions] = d

    # Reshape to (n, n^(m-1)) **column-major**，匹配 MATLAB 的 reshape 語義。
    D_dense = D_vec.reshape(n, num_cols, order="F")

    return csr_matrix(D_dense)


def ten2mat(A, k=0, matlab_compat=False):
    """將 m-order tensor A 做 **mode-k 展開**（mode-k unfolding / matricization），
    回傳 shape 為 `(A.shape[k], prod(其他維度))` 的 2-D 矩陣。

    對應 MATLAB reference：`matlab_ref/hni/HONI.m` 第 152-177 行。MATLAB 原碼
    用 `eval()` 動態組出 `reshape(A(:,...,i,...,:), 1, n^(m-1))` 的字串再逐 row
    執行。**本 port 禁止使用 `eval()` 或 `exec()`**，改用 `np.moveaxis` 把
    第 k 個 axis 搬到最前面，然後 `reshape(..., order='F')` 以匹配 MATLAB 的
    column-major 展開。兩種做法數學上等價、結果逐位元一致，但本版是純 numpy
    原生運算，安全且可讀。

    Parameters
    ----------
    A : array-like
        m-order tensor，shape `(n_0, n_1, ..., n_{m-1})`。`m` 必須 `>= 1`。
        dense ndarray;不支援 sparse（MATLAB 原碼也不支援）。
    k : int, default 0
        展開的 mode index (0-based)，對應 MATLAB 原碼的 `k - 1`。必須
        `0 <= k < A.ndim`。HNI 實際使用時 k 永遠是 0（mode-1 unfolding），
        但本函式支援任意 mode 以維持與 MATLAB 介面的對應。
    matlab_compat : bool, default False
        **對輸出結果無作用** — 本函式永遠使用 `order='F'` 做 reshape，是
        Trap #4 column-major 的主戰場。`order='F'` 是演算法正確性的必要
        條件（見 Paper derivation 以下），不是可切換選項。保留 flag 以
        維持 API 一致性。

    Returns
    -------
    np.ndarray, 2-D
        Shape `(n_k, prod(其他維度))`。對於 n-dim m-order 正方張量（所有
        mode 大小都是 n），shape 為 `(n, n^(m-1))`。dtype 繼承自 A。

    四大陷阱在本函式的觸發分析
    -------------------------
    Trap #3 Indexing：MATLAB 的 `k` 是 1-based，Python 的 `k` 是 0-based。
        呼叫 MATLAB `ten2mat(A, 1)` 的對應是 Python `ten2mat(A, 0)`。
    Trap #4 Column-major：⭐⭐ **整個 HNI port 的 column-major 主戰場**。
        見下方 Paper derivation，數學上 row-major (`order='C'`) 和
        column-major (`order='F'`) 對任意非對角 tensor pattern 都會產生
        **不同的輸出陣列**，不像 `sp_tendiag` 那樣有巧合。任何未來改動
        本函式的 `order='F'` 都會立刻被 sanity test（見
        `test_tensor_utils.py::test_ten2mat_basic`）和 parity test 抓到。

    Paper derivation (order='F' 是必要而非選擇)
    --------------------------------------------
    Take `T = np.arange(1, 9).reshape(2, 2, 2)`. With numpy default C-order
    fill, entries are:
        T[0,0,0]=1  T[0,0,1]=2  T[0,1,0]=3  T[0,1,1]=4
        T[1,0,0]=5  T[1,0,1]=6  T[1,1,0]=7  T[1,1,1]=8

    Mode-0 unfolding (k=0) defines `B[i, j] = T[i, j_1, j_2]` where `j`
    unravels `(j_1, j_2)` in column-major over shape `(2, 2)`:
        j=0 → (0,0)   j=1 → (1,0)   j=2 → (0,1)   j=3 → (1,1)

    So the correct B is:
        B[0, :] = [T[0,0,0], T[0,1,0], T[0,0,1], T[0,1,1]] = [1, 3, 2, 4]
        B[1, :] = [T[1,0,0], T[1,1,0], T[1,0,1], T[1,1,1]] = [5, 7, 6, 8]
        B_correct = [[1, 3, 2, 4], [5, 7, 6, 8]]

    Implementation: `T.reshape(2, 4, order='F')` 產生正好這個結果 ✓

    If `order='C'` is accidentally written instead:
        B_wrong = [[1, 2, 3, 4], [5, 6, 7, 8]]
    欄 1 和欄 2 在兩個結果之間被交換，max |B_F - B_C| = 1 for this input.
    對 random 輸入來說差距是輸入動態範圍的量級（不是 machine epsilon），
    因此 parity test 會立即以巨大誤差報告失敗。

    此推導證明了 `sp_tendiag` 那種「F-order 和 C-order 巧合給出同樣 (row, col)
    位置」的現象**不會推廣到一般 tensor**，`ten2mat` 真正需要 `order='F'`。

    See also
    --------
    `test_tensor_utils.py::test_ten2mat_basic` 包含對比測試，直接比較
    `ten2mat(T)` 和 `T.reshape(..., order='C')`，證明兩者不同 — 未來若有人
    把 `order='F'` 改成 `'C'`，這個對比測試會立刻抓到。
    """
    A = np.asarray(A)
    if A.ndim < 1:
        raise ValueError(f"A must be at least 1-D, got ndim={A.ndim}")
    if k < 0 or k >= A.ndim:
        raise ValueError(f"k must be in [0, {A.ndim}), got {k}")

    # Move the k-th axis to position 0.
    # 對應 MATLAB `eval(express)` 裡的 `A(:,...,i,:,...,:)`（把 mode k 抽出）。
    A_moved = np.moveaxis(A, k, 0)

    n_k = A_moved.shape[0]
    other = int(np.prod(A_moved.shape[1:])) if A_moved.ndim > 1 else 1

    # Column-major (Fortran) reshape — 見 Paper derivation 為何這一步必須 'F'。
    return A_moved.reshape(n_k, other, order="F")


def sp_Jaco_Ax(AA, x, m, matlab_compat=False):
    """計算 F(x) = AA · x^(m-1) 的 Jacobian `F'(x)`，回傳 scipy.sparse csr_matrix
    of shape `(n, n)`。

    對應 MATLAB reference：`matlab_ref/hni/Multi.m` 第 79-87 行。

    數學公式（鏈鎖法則於 x^⊗p 的導數）：
        F'(x) = AA · Σ_{i=1..p} (x^⊗(i-1) ⊗ I_n ⊗ x^⊗(p-i))
    其中 p = m - 1。

    常見極限情況的退化：
    - `m = 1` (p=0)：空求和，F(x) = AA · 1 是常數，Jacobian = 0 matrix
    - `m = 2` (p=1)：單項 `AA · (1 ⊗ I ⊗ 1) = AA · I = AA`（若 AA 為方陣）
    - `m = 3` (p=2)：`AA · (kron(I, x) + kron(x, I))`，兩項

    Parameters
    ----------
    AA : array-like or scipy.sparse matrix, 2-D
        m-order n-dim tensor 的 mode-1 unfolding，shape `(n, n^(m-1))`。
    x : array-like, 1-D
        向量，shape `(n,)`。
    m : int
        Tensor order。必須 `m >= 1`。
    matlab_compat : bool, default False
        **對輸出無作用**（no-op）。保留以維持 API 一致性。本函式的核心
        運算是 `scipy.sparse.kron` + sparse matmul — numpy 和 MATLAB 對
        1-D Kronecker 的元素順序一致（見 tenpow 的 Trap #4 分析），sparse
        matmul 也 layout-free，因此無須切換。

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape `(n, n)` 的 Jacobian。即使 AA 是 dense ndarray，本函式也會
        內部轉成 sparse 做運算、輸出 sparse（對應 MATLAB 原版的隱含 sparse
        輸出：因為 I 是 sparse、kron 結果 sparse、再跟 AA 乘 scipy 保留
        sparse 格式）。

    四大陷阱在本函式的觸發分析
    -------------------------
    Trap #3 Indexing：⚠️ MATLAB `for i = 1:p` 的 `i-1` 和 `p-i` 是 exponent，
        在 `[0, p-1]` 範圍。Python 我們保留相同的 1-based loop bound
        `range(1, p + 1)`，讓 `i - 1` 和 `p - i` 這兩個 expression 跟
        MATLAB 一字不差（不做 0-based 重編號）— 這樣程式碼長得像 MATLAB
        原版，審校時可以逐行對照。
    Trap #4 Column-major：**不觸發**。本函式純粹是 Kronecker algebra + sparse
        matmul，沒有任何 `reshape`。唯一的 `reshape(-1, 1)` 是把 1-D `tenpow`
        結果轉成 column vector 餵給 `scipy.sparse.kron`，這個 reshape 是 1-D
        → 2-D column、layout 中立。

        注意：這裡的 mode 順序 (x^⊗(i-1) ⊗ I ⊗ x^⊗(p-i)) 直接對應張量
        unfolding 的 column-major 慣例 — 這個慣例已經烘焙進公式本身，不是
        我們選的。逐字 port 公式即正確。
    """
    if not issparse(AA):
        AA = np.asarray(AA)
    if AA.ndim != 2:
        raise ValueError(f"AA must be 2-D, got shape {AA.shape}")

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    n = len(x)
    p = m - 1

    # m=1 邊界：F(x) 是常數，Jacobian 是零矩陣 (n, n)
    if p == 0:
        return csr_matrix((n, n), dtype=x.dtype)

    # 內部統一 sparse CSR 做運算，對應 MATLAB 的 `dense*sparse = sparse` 行為
    AA_sp = AA.tocsr() if issparse(AA) else csr_matrix(AA)
    I = sp_eye(n, format="csr")

    J = csr_matrix((n, n), dtype=x.dtype)
    for i in range(1, p + 1):
        # i 保留 MATLAB 1-based 迴圈範圍，讓 `i - 1` 和 `p - i` 這兩個
        # exponent expression 跟 MATLAB 原版一字不差、好審校。
        left = tenpow(x, i - 1, matlab_compat=matlab_compat).reshape(-1, 1)  # (n^(i-1), 1)
        right = tenpow(x, p - i, matlab_compat=matlab_compat).reshape(-1, 1)  # (n^(p-i), 1)

        # kron(I (n,n), right (n^(p-i), 1)) → sparse (n^(p-i+1), n)
        inner = sp_kron(I, right, format="csr")

        # kron(left (n^(i-1), 1), inner (n^(p-i+1), n)) → sparse (n^p, n)
        outer = sp_kron(left, inner, format="csr")

        # AA_sp (n, n^p) @ outer (n^p, n) → sparse (n, n)
        term = AA_sp @ outer

        J = J + term

    return J


def multi(AA, b, m, tol, record_history=False, matlab_compat=False):
    """求解多線性系統 A·u^(m-1) = b 的正解，使用外層 Newton + 內層三等分（one-third）halving line search。

    對應 MATLAB reference：`matlab_ref/hni/Multi.m` 第 1-54 行。演算法結構與
    陷阱分析見 `docs/superpowers/notes/multi_hazard_analysis.md`。

    Parameters
    ----------
    AA : array-like or scipy.sparse matrix, 2-D
        M-tensor 的 mode-1 unfolding，shape `(n, n^(m-1))`。可為 dense 或 sparse。
    b : array-like, 1-D
        多線性系統右邊，shape `(n,)`。**不會被 mutate**：MATLAB 原碼 line 15
        做 `b = b.^(m-1)` 就地覆寫，本 port 改用 `b_raised = np.asarray(b,
        dtype=np.float64) ** (m-1)` 隔離，保護呼叫端的輸入（Q4）。
    m : int
        Tensor order，`m >= 2`。
    tol : float
        外層 Newton 相對停止門檻：`min(res) <= tol * (na*||u|| + nb)` 則收斂。
    record_history : bool, default False
        `True` 時額外回傳 per-iteration history（用於 parity test）；`False` 時
        只回傳 `(u, nit, hal)`。
    matlab_compat : bool, default False
        **對 multi 本身是 no-op**。會傳進 `tpv` / `sp_Jaco_Ax`，但那兩支也是
        no-op。保留以維持模組 API 一致。

    Returns
    -------
    若 `record_history=False`：`(u, nit, hal)`
        u : np.ndarray, 1-D, shape `(n,)`
            多線性系統的正解。
        nit : int
            外層 Newton 迭代完成次數（**Python 0-based 語意**）。`nit = 0` 表示
            初始化即收斂、沒進外層 loop；`nit = k` 表示完成 k 次外層 iter。
            **跟 MATLAB 的關係**：MATLAB `nit_mat = nit_py + 1`（MATLAB 把初始化
            算作 "第 1 次"，nit_mat 最終值 = 1 + 外層 iter 次數）。
        hal : np.ndarray, 1-D, shape `(100,)`
            Pre-allocated 長 100 的 halving 計數 buffer（`hal[k]` = 第 k 次外層
            iter 進入三等分 halving 的次數；`hal[0] = 0`；超過 `nit` 的位置為 0）。
    若 `record_history=True`：`(u, nit, hal, history)`
        history : dict，所有 array 長度 `nit + 1`，index `[0]` 存初始化狀態
            - `u_history` : `(n, nit+1)`，每外層 iter 結束後的 u；`[:, 0]` = `b/||b||`
            - `res_history` : `(nit+1,)`，每外層 iter 的最終殘差 `||A·u^(m-1) - b^(m-1)||`
            - `theta_history` : `(nit+1,)`，每外層 iter 最終使用的 θ；`[0] = NaN`
            - `hal_history` : `(nit+1,)`，每外層 iter 的 halving 次數；`[0] = 0`
            - `v_history` : `(n, nit+1)`，每外層 iter 的 Newton 方向 `v = M \\ b`；`[:, 0] = NaN`

    §四 Hazard checklist（手眼同步點、逐條手工核對）
    ----------------------------------------------
    1. `theta /= 3`（三等分 one-third）**不是** `/= 2`。見 Multi.m line 40。
    2. Halving 內迴圈用 **snapshot** `u_old` / `v_old`（line 33），不是當前 `u` / `v`。
       每次 halving 都從同一個 snapshot 重新插值、不鏈式累加。見 Multi.m line 41。
    3. `b_raised = np.asarray(b, dtype=np.float64) ** (m-1)`：保證 float64、且
       不 mutate 呼叫端的 `b`（MATLAB 原碼 line 15 是 in-place、本 port 隔離）。
    4. `M = (sp_Jaco_Ax(...) / (m-1)).tocsc()`：spsolve 前顯式轉 CSC、避免
       scipy.sparse.linalg 對 CSR 輸入的 SparseEfficiencyWarning（Q3）。
    5. Pre-allocate buffers：`res = np.full(100, na+nb)`、`hal = np.zeros(100)`。
       外層 while 條件用 `np.min(res)`（整條 buffer），靠 pre-fill 值 `na+nb`
       大於實際殘差才能避免 min 被未寫區干擾（Q1、見 hazard analysis §二）。
    6. 內層 while 條件 **不交換順序**：`res[nit] - res[nit-1] > 0 or np.min(temp) < 0`。
       MATLAB line 39 `||` 是 short-circuit OR，逐字保留可讀 + debug 時易對照。

    Indexing convention (Q2)
    ------------------------
    完全 Python 0-based、不保留 MATLAB 1-based。對應關係：
    - MATLAB `res(1)` ≡ Python `res[0]`（初始化後的殘差）
    - MATLAB `res(nit_mat)` ≡ Python `res[nit_py]` where `nit_py = nit_mat - 1`
    - 寫 parity test 時對 `nit` scalar 做 `assert nit_mat == nit_py + 1`

    Raises
    ------
    ValueError
        若 `b` 不是 1-D、`m < 2`、`tol <= 0`。
    AssertionError
        函式尾端 sanity：`res[:nit+1] <= na + nb + 1e-10`。若失敗表示
        pre-fill 策略的 upper bound 被突破，需要回頭檢查數學假設。
    """
    b_vec = np.asarray(b, dtype=np.float64)
    if b_vec.ndim != 1:
        raise ValueError(f"b must be 1-D, got shape {b_vec.shape}")
    if not isinstance(m, (int, np.integer)) or m < 2:
        raise ValueError(f"m must be integer >= 2, got {m}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    # --- 初始化 (Multi.m line 14-21) ---
    u = b_vec / np.linalg.norm(b_vec)                           # line 14
    b_raised = b_vec ** (m - 1)                                 # line 15 (no mutation of caller's b)

    if issparse(AA):
        na = float(np.sqrt(sparse_norm(AA, np.inf) * sparse_norm(AA, 1)))  # line 16
    else:
        AA_arr = np.asarray(AA)
        if AA_arr.ndim != 2:
            raise ValueError(f"AA must be 2-D, got shape {AA_arr.shape}")
        na = float(np.sqrt(
            np.linalg.norm(AA_arr, np.inf) * np.linalg.norm(AA_arr, 1)
        ))

    nb = float(np.linalg.norm(b_raised))                        # line 17
    temp = tpv(AA, u, m, matlab_compat=matlab_compat)           # line 18

    _BUF = 100
    res = np.full(_BUF, na + nb, dtype=np.float64)              # line 19
    res[0] = np.linalg.norm(temp - b_raised)                    # line 20 (MATLAB res(1))
    hal = np.zeros(_BUF, dtype=np.float64)                      # line 21
    nit = 0                                                      # line 21: MATLAB `nit=1` → Python 0-based

    if record_history:
        n = len(u)
        u_history = np.zeros((n, _BUF), dtype=np.float64)
        u_history[:, 0] = u
        v_history = np.full((n, _BUF), np.nan, dtype=np.float64)
        theta_history = np.full(_BUF, np.nan, dtype=np.float64)

    tol_theta = 1e-14                                            # line 32
    _MAX_NIT = _BUF - 1  # Python 0-based: MATLAB `nit < 100` 相當於 Python `nit < 99`

    # --- 外層 Newton 迴圈 (line 22-52) ---
    while np.min(res) > tol * (na * np.linalg.norm(u) + nb) and nit < _MAX_NIT:
        nit += 1                                                 # line 24

        # 1. Jacobian + 線性求解 (line 27-28)
        M = (sp_Jaco_Ax(AA, u, m, matlab_compat=matlab_compat) / (m - 1)).tocsc()
        v = spsolve(M, b_raised)

        # 2. Newton step θ = 1 試算 (line 31-37)
        theta = 1.0
        u_old = u.copy()                                         # line 33 snapshot
        v_old = v.copy()
        u = (1 - theta / (m - 1)) * u_old + theta * v_old / (m - 1)    # line 34
        temp = tpv(AA, u, m, matlab_compat=matlab_compat)        # line 35
        res[nit] = np.linalg.norm(temp - b_raised)               # line 36
        hit = 0                                                   # line 37

        # 3. 三等分 halving line search (line 39-50)
        #    OR 順序保持 MATLAB 寫法：先 res 比較、再 min(temp) < 0
        while res[nit] - res[nit - 1] > 0 or np.min(temp) < 0:
            theta /= 3                                           # line 40 (/3 NOT /2)
            u = (1 - theta / (m - 1)) * u_old + theta * v_old / (m - 1)  # line 41 snapshot-based
            temp = tpv(AA, u, m, matlab_compat=matlab_compat)    # line 42
            res[nit] = np.linalg.norm(temp - b_raised)           # line 43-44
            hit += 1                                              # line 45
            if theta < tol_theta:                                 # line 46
                print(
                    "Can't find a suitible step length such that inner residual decrease!"
                )
                break

        hal[nit] = hit                                           # line 51

        if record_history:
            u_history[:, nit] = u
            v_history[:, nit] = v
            theta_history[nit] = theta

    # Upper-bound sanity: pre-fill 值 (na+nb) 必須大於任何實際算出的殘差，否則
    # `min(res)` 的邏輯會被 pre-fill 值誤觸發收斂。失敗表示 hazard analysis §二
    # 選項 (A) 的 upper-bound 假設在此輸入下不成立，需檢查數學假設。
    assert np.all(res[:nit + 1] <= na + nb + 1e-10), (
        f"res upper-bound violation: max res[:{nit+1}] = {res[:nit+1].max():.6e}, "
        f"na+nb = {na + nb:.6e}"
    )

    if record_history:
        history = {
            "u_history": u_history[:, :nit + 1].copy(),
            "res_history": res[:nit + 1].copy(),
            "theta_history": theta_history[:nit + 1].copy(),
            "hal_history": hal[:nit + 1].copy(),
            "v_history": v_history[:, :nit + 1].copy(),
        }
        return u, nit, hal, history

    return u, nit, hal


def honi(
    A, m, tol=1e-12, *,
    linear_solver="exact",
    maxit=200,
    initial_vector=None,
    record_history=False,
    record_inner_history=False,
    plot_res=False,
    matlab_compat=False,
):
    """求解 m-order n-dim tensor 的最大 eigenvalue + 對應 eigenvector。
    外層 eigenvalue iteration + 內層 Multi Newton solver（shift-invert）。

    對應 MATLAB reference：`matlab_ref/hni/HONI.m`（232 行）。演算法結構與
    陷阱分析見 `docs/superpowers/notes/honi_hazard_analysis.md`。

    Parameters
    ----------
    A : array-like or scipy.sparse matrix
        Polymorphic（match MATLAB 的 ndim dispatch）：
        - 2-D `(n, n^(m-1))` — m-tensor 的 mode-1 unfolding，直接使用
        - m-D `(n, n, ..., n)` — 方張量，內部用 `ten2mat(A, k=0)` 做 mode-1
          unfolding（column-major reshape via `order='F'`）
        sparse 僅支援 2-D unfolding；tensor 必須 dense。
    m : int
        Tensor order，`m >= 2`。必須 **int 型別**（**§四 checklist #5.4**：
        負的 `min(temp)` 取非整數次方會產生 complex；此處強制 int 避免）。
    tol : float, default 1e-12
        外層 eigenvalue iteration 相對殘差停止門檻（MATLAB default 1e-12）。

    Keyword-only
    ------------
    linear_solver : {"exact", "inexact"}, default "exact"
        內層 Multi 的 tolerance 策略 + 外層 lambda_U / res 更新公式。
        兩分支數學上求解同一個問題但迭代路徑完全不同、**需各自 parity**。
        - "exact"：inner_tol = 1e-10 寫死；temp = x/y 比值；
          lambda_U 增量更新 `-= min(temp)^(m-1)`；res 用 m-1 次方公式
        - "inexact"：inner_tol 動態 `max(1e-10, min(res)*min(x)^(m-1)/nit_mat)`；
          x 先 normalize、再用新 x 算 temp = tpv/x^(m-1)、lambda_U = max(temp)
    maxit : int, default 200
        外層最大迭代數（MATLAB default 100；本 port 放寬到 200）。
    initial_vector : array-like 1-D, optional
        初始 eigenvector 估計，shape `(n,)`。`None` 時用 `np.random.default_rng()`
        隨機產生（**注意：parity test 要顯式傳入、避免 rng 不一致**）。
    record_history : bool, default False
        True 時回傳 history dict，含 9 個外層 history 欄位（x/lambda/res/y/
        inner_tol/chit/hal_per_outer/innit/hal_accum），長度 `outer_nit + 1`。
    record_inner_history : bool, default False
        True 時額外在 history dict 加 `inner_histories` key：每外層 iter 一份
        Multi 的 5 欄 history dict。**主要用在**外層 parity 失敗 first_bad_iter
        指向某 outer iter k 時、重跑開啟此 flag 定位 Multi 內部狀態。預設關閉
        因為 Multi 已單元 parity 驗證、HONI 出錯 99% 在 HONI 自身更新公式。
    plot_res : bool, default False
        **Q8 決定：不實作**（原 MATLAB `plot_res=1` 會 loglog 畫 residual 曲線）。
        kwarg 保留以維持 API 可擴充；傳 True 會 `warnings.warn` 後忽略。
    matlab_compat : bool, default False
        **對 honi 本身是 no-op**。會傳進 `tpv / sp_tendiag / ten2mat / multi`。

    Returns
    -------
    若 `record_history=False`：`(lambda_, x, outer_nit, total_inner_nit,
                               outer_res_history, lambda_history)`
        lambda_ : float
            最大 eigenvalue 估計（λ）。
        x : np.ndarray, 1-D `(n,)`
            對應 unit-norm eigenvector。
        outer_nit : int
            外層 iter 完成次數（**Python 0-based**）。MATLAB nit = outer_nit + 1。
        total_inner_nit : int
            內層 Multi Newton iter 總次數，**與 MATLAB `innit` bit-identical**。
            實作時 `total_inner_nit += chit_py + 1`（chit offset，見 §四 checklist #5.7）。
        outer_res_history : np.ndarray, 1-D
            每外層 iter 殘差（相對）。長度 `outer_nit + 1`，`[0]` 為初始化殘差
            `|max(temp)-min(temp)|/lambda_U`。
        lambda_history : np.ndarray, 1-D
            每外層 iter 結束後的 lambda_U。長度 `outer_nit + 1`，`[0]` 為初始
            估計 `max(tpv(AA, x_init, m) / x_init^(m-1))`。
    若 `record_history=True`：回傳 7-tuple `(..., history)`，額外 history 是 dict，
    9 個 key 都是長度 `outer_nit + 1` 的 array。若 `record_inner_history=True`
    另加 `inner_histories` key（list of Multi history dict）。

    §四 Hazard checklist（手眼同步點、逐條核對）
    -------------------------------------------
    1. **Polymorphic A**：`A.ndim == 2` 當 unfolding、`A.ndim >= m` 當 tensor、
       tensor 必須是方張量（`A.shape` 全 n）、呼叫 `ten2mat(A, k=0, ...)`。
    2. **`m` 強制 int**：`isinstance(m, (int, np.integer))` 開頭檢查。負
       `min(temp)^float` 會產生 complex、int 次方保持 real（§五 5.4）。
    3. **`inner_tol` inexact 公式**：用明變數 `nit_mat = nit + 1` 代入字面
       MATLAB 1-based `nit`（§五 5.5）。
    4. **`total_inner_nit` 累積**：`total_inner_nit += chit_py + 1`（Multi
       Python nit 是 0-based、MATLAB `chit` 多一；§五 5.7）。
    5. **`lambda_U*II - AA` sparse/dense dispatch**：sparse AA → 兩邊 sparse；
       dense AA → 轉 `II.toarray()` 避免 scipy 型別混淆（§五 5.1）。
    6. **exact/inexact 分支欄位拆清**：兩分支的 temp 定義、lambda_U 更新公式、
       res 公式、x normalize 時機**全部不同**。Port 建議把兩分支的 step
       寫在 `if/elif` 明確分離、逐字對照 MATLAB line 53-75。
    """
    # ---- Input validation ----
    if not isinstance(m, (int, np.integer)):
        raise ValueError(f"m must be int, got {type(m).__name__}")
    m = int(m)
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if linear_solver not in ("exact", "inexact"):
        raise ValueError(
            f"linear_solver must be 'exact' or 'inexact', got {linear_solver!r}"
        )
    if maxit < 2:
        raise ValueError(f"maxit must be >= 2, got {maxit}")

    # ---- Polymorphic A dispatch ----
    if issparse(A):
        if A.ndim != 2:
            raise ValueError(f"sparse A must be 2-D unfolding, got ndim={A.ndim}")
        n = A.shape[0]
        if A.shape[1] != n ** (m - 1):
            raise ValueError(
                f"sparse A shape {A.shape} incompatible with m={m}; "
                f"expected ({n}, {n ** (m - 1)})"
            )
        AA = A
    else:
        A_arr = np.asarray(A)
        if A_arr.ndim == 2:
            n = A_arr.shape[0]
            if A_arr.shape[1] != n ** (m - 1):
                raise ValueError(
                    f"A shape {A_arr.shape} incompatible with m={m}; "
                    f"expected ({n}, {n ** (m - 1)})"
                )
            AA = A_arr
        elif A_arr.ndim >= 3:
            n = A_arr.shape[0]
            if A_arr.ndim != m:
                raise ValueError(
                    f"tensor A.ndim={A_arr.ndim} but m={m} (need ndim == m)"
                )
            if not all(s == n for s in A_arr.shape):
                raise ValueError(
                    f"A must be square tensor (all dims == n={n}), got shape {A_arr.shape}"
                )
            AA = ten2mat(A_arr, k=0, matlab_compat=matlab_compat)
        else:
            raise ValueError(
                f"A must be 2-D unfolding or m-D tensor, got ndim={A_arr.ndim}"
            )

    # ---- Initial vector ----
    if initial_vector is None:
        rng = np.random.default_rng()
        initial_vector = rng.random(n)
    else:
        initial_vector = np.asarray(initial_vector, dtype=np.float64).ravel()
        if initial_vector.shape != (n,):
            raise ValueError(
                f"initial_vector shape {initial_vector.shape} != ({n},)"
            )

    if plot_res:
        warnings.warn(
            "plot_res is not implemented in this port (HONI hazard Q8); ignoring",
            stacklevel=2,
        )

    # ---- 初始化 (HONI.m line 35-44) ----
    x = initial_vector / np.linalg.norm(initial_vector)                 # line 35-36
    temp = tpv(AA, x, m, matlab_compat=matlab_compat) / (x ** (m - 1))  # line 37
    lambda_U = float(np.max(temp))                                       # line 38
    II = sp_tendiag(np.ones(n), m, matlab_compat=matlab_compat)          # line 39

    res = np.ones(maxit, dtype=np.float64)                               # line 41
    res[0] = abs(float(np.max(temp)) - float(np.min(temp))) / lambda_U   # line 43 (0-based)
    nit = 0                                                               # MATLAB nit=1 → 0-based
    innit = 0                                                             # inner Newton total (MATLAB 1-based)
    hal = 0                                                               # halving total

    # ---- history buffers ----
    # lambda_history always tracked (part of standard 6-item return)
    lambda_history = np.full(maxit, np.nan, dtype=np.float64)
    lambda_history[0] = lambda_U

    if record_history:
        x_history = np.zeros((n, maxit), dtype=np.float64)
        x_history[:, 0] = x
        y_history = np.full((n, maxit), np.nan, dtype=np.float64)
        inner_tol_history = np.full(maxit, np.nan, dtype=np.float64)
        chit_history = np.zeros(maxit, dtype=np.float64)
        hal_per_outer_history = np.zeros(maxit, dtype=np.float64)
        innit_history = np.zeros(maxit, dtype=np.float64)
        hal_accum_history = np.zeros(maxit, dtype=np.float64)
        inner_histories = [None]  # slot 0 = init (no Multi call)

    _MAX_NIT = maxit - 1  # Python 0-based; MATLAB `nit < maxit` ≡ Python `nit < maxit - 1`

    # ---- 外層 eigenvalue iteration (HONI.m line 46-79) ----
    while np.min(res) > tol and nit < _MAX_NIT:
        nit += 1                                                          # line 47
        nit_mat = nit + 1  # explicit MATLAB 1-based equivalent (Q5 checklist #3)

        # Shift-invert: AA_shifted = lambda_U * II - AA (line 54/68 inside Multi call)
        if issparse(AA):
            AA_shifted = lambda_U * II - AA
        else:
            AA_shifted = lambda_U * II.toarray() - AA

        # inner_tol: static (exact) or dynamic (inexact)
        if linear_solver == "exact":
            inner_tol = 1e-10                                             # line 53
        else:
            # Q5 checklist #3: nit_mat (not Python nit) matches MATLAB's literal nit
            inner_tol = max(
                1e-10,
                float(np.min(res)) * float(np.min(x)) ** (m - 1) / nit_mat,
            )                                                             # line 67

        # Call Multi (inner Newton solver)
        if record_inner_history:
            y, chit_py, hal_inn, inner_hist = multi(
                AA_shifted, x, m, inner_tol,
                record_history=True, matlab_compat=matlab_compat,
            )
        else:
            y, chit_py, hal_inn = multi(
                AA_shifted, x, m, inner_tol,
                record_history=False, matlab_compat=matlab_compat,
            )
            inner_hist = None

        # Q6 checklist #4: chit_mat = chit_py + 1 to match MATLAB's 1-based innit
        chit_mat = chit_py + 1
        innit += chit_mat                                                 # line 56/70
        hal_this_outer = int(np.sum(hal_inn))
        hal += hal_this_outer                                             # line 55/69

        # Branch-specific updates (Q1 + §四 checklist #6)
        if linear_solver == "exact":
            # HONI.m line 58-61: ratio → lambda update → res → normalize
            temp = x / y
            min_temp = float(np.min(temp))
            max_temp = float(np.max(temp))
            lambda_U = lambda_U - min_temp ** (m - 1)                    # line 59
            res[nit] = abs(max_temp ** (m - 1) - min_temp ** (m - 1)) / lambda_U  # line 60
            x = y / np.linalg.norm(y)                                    # line 61
        else:  # "inexact"
            # HONI.m line 72-75: normalize → tpv on new x → lambda = max
            x = y / np.linalg.norm(y)                                    # line 72
            temp = tpv(AA, x, m, matlab_compat=matlab_compat) / (x ** (m - 1))  # line 73
            min_temp = float(np.min(temp))
            max_temp = float(np.max(temp))
            lambda_U = max_temp                                          # line 74
            res[nit] = abs(max_temp - min_temp) / lambda_U               # line 75

        lambda_history[nit] = lambda_U

        if record_history:
            x_history[:, nit] = x
            y_history[:, nit] = y
            inner_tol_history[nit] = inner_tol
            chit_history[nit] = chit_mat
            hal_per_outer_history[nit] = hal_this_outer
            innit_history[nit] = innit
            hal_accum_history[nit] = hal
            if record_inner_history:
                inner_histories.append(inner_hist)

    # ---- Upper-bound sanity: pre-fill value 1.0 must bound all written res ----
    assert np.all(res[:nit + 1] <= 1.0 + 1e-10), (
        f"res upper-bound violation: max res[:{nit+1}] = {res[:nit+1].max():.6e}, "
        f"pre-fill was 1.0 (should hold since res = |max-min|/lambda_U <= 1)"
    )

    outer_res_history = res[:nit + 1].copy()
    lambda_history_out = lambda_history[:nit + 1].copy()

    if record_history:
        history = {
            "x_history": x_history[:, :nit + 1].copy(),
            "lambda_history": lambda_history_out.copy(),
            "res_history": outer_res_history.copy(),
            "y_history": y_history[:, :nit + 1].copy(),
            "inner_tol_history": inner_tol_history[:nit + 1].copy(),
            "chit_history": chit_history[:nit + 1].copy(),
            "hal_per_outer_history": hal_per_outer_history[:nit + 1].copy(),
            "innit_history": innit_history[:nit + 1].copy(),
            "hal_accum_history": hal_accum_history[:nit + 1].copy(),
        }
        if record_inner_history:
            history["inner_histories"] = inner_histories
        return (
            float(lambda_U), x, nit, int(innit),
            outer_res_history, lambda_history_out, history,
        )

    return (
        float(lambda_U), x, nit, int(innit),
        outer_res_history, lambda_history_out,
    )


# ---------------------------------------------------------------------------
# NNI (Newton-Noda Iteration) — Layer 3 step 3/3
# ---------------------------------------------------------------------------

# gmres default options (user-facing names; translated to scipy's names at call site)
_NNI_GMRES_DEFAULTS = {"maxit": 1000, "tol": 1e-10, "restart": 20, "M": None}


def nni(
    A, m, tol=1e-12, *,
    linear_solver="spsolve",
    gmres_opts=None,
    halving=False,
    tol_theta=1e-12,
    maxit=200,
    initial_vector=None,
    record_history=False,
    plot_res=False,
    matlab_compat=False,
):
    """求解 m-order n-dim 正 M-tensor 的最大 eigenvalue + 對應 eigenvector。
    單層 Newton 迭代（對比 HONI 的外層 λ + 內層 Multi 雙層結構）。

    對應 MATLAB reference（兩個檔、由 `halving` kwarg 切換）：
    - `source_code/Tensor Eigenvalue Problem/2020_HNI_Revised/NNI.m`
      — 270 行、canonical（halving while 整塊註解）、`chit ≡ 0`、對應 `halving=False`
    - `source_code/Tensor Eigenvalue Problem/2020_HNI_Revised/NNI_ha.m`
      — 270 行、halving 啟用版（2020 paper `Test_Heig2.m` benchmark 實際呼叫）、
      對應 `halving=True`

    兩檔 diff 100% 集中在 `step_length` subfunction、純 halving-only 差異
    （Phase 0 驗證）。演算法結構與陷阱分析見
    `docs/superpowers/notes/nni_hazard_analysis.md`；halving 擴展與風險見同檔 §九。

    **前提假設**：輸入 `A` 是**正 M-tensor**（或等價的非負張量）、初始向量 `x_0 > 0`。
    演算法**不主動投影非負** — 靠 Perron-Frobenius 類性質保證 `x` 在收斂路徑上保持
    非負。若輸入非 M-tensor、迭代可能 diverge 或 `lambda_L < 0`（此時 `res > 1`、
    尾端 `assert` 會 fail）、行為與 MATLAB 原碼一致、非 port bug。

    Parameters
    ----------
    A : array-like or scipy.sparse matrix
        Polymorphic（match MATLAB ndim dispatch、同 HONI）：
        - 2-D `(n, n^(m-1))` — m-tensor 的 mode-1 unfolding，直接使用
        - m-D `(n, n, ..., n)` — 方張量，內部用 `ten2mat(A, k=0)` 做 mode-1 unfolding
        sparse 僅支援 2-D unfolding；tensor 必須 dense。
    m : int
        Tensor order，`m >= 2`。必須 **int 型別**（`x ** (m-1)` 對負 `x` 取非整數
        次方會產生 complex；強制 int 避免）。
    tol : float, default 1e-12
        外層 Newton 停止門檻（MATLAB default `1e-12`）。

    Keyword-only
    ------------
    linear_solver : {"spsolve", "gmres"}, default "spsolve"
        每 iter 的 sparse 線性系統 `(-M_shifted) · y = x^(m-1)` 求解器：
        - ``"spsolve"``：scipy `spsolve`（LU、direct）— 對應 MATLAB `mldivide`
          (`\\`)、適合 dense 或中小 sparse、**parity test 對齊 MATLAB 用此分支**
        - ``"gmres"``：scipy `gmres`（iterative）— Python-only、適合大 sparse
          tensor (LU 爆記憶體時)、**不求 bit-identical**、由 sanity test 驗證
          「兩 solver 給近似 eigenpair」

        MATLAB 原版的 `'GTH'`（Alfa-Xue-Ye M-matrix LU）**未 port** — 傳入會
        `raise ValueError`，訊息建議改用 `'spsolve'` 或 `'gmres'`。
    gmres_opts : dict, optional
        僅在 `linear_solver="gmres"` 時生效。預設 ``None`` 時用
        `{"maxit": 1000, "tol": 1e-10, "restart": 20, "M": None}`；使用者可傳入
        部分 key 來 override（與預設 shallow-merge）。Key 語意：
        - ``maxit`` (int)：GMRES 最大 iteration、對應 scipy 的 ``maxiter``
        - ``tol`` (float)：相對 tolerance、對應 scipy 的 ``rtol``
        - ``restart`` (int)：GMRES restart length
        - ``M`` (LinearOperator or None)：preconditioner、``None`` 為無 precond
    halving : bool, default False
        Step-length halving line search 開關（§九 addendum）：
        - ``False`` — 對應 canonical `NNI.m`；full step `x_new = (m-2)*x + y`、無條件接受、
          `hit_per_outer_history` 恆 0（halving path 在 MATLAB 端被 `%` 註解）
        - ``True`` — 對應 `NNI_ha.m`；先算 full step、若 `lambda_U_new > lambda_U + 1e-13`
          （單調性破壞）則進 halving while、每次 `theta /= 2` 重算 `x_new / λ_U / λ_L`、
          直到 `lambda_U_new <= lambda_U + 1e-13` 或 `theta < tol_theta` 為止
    tol_theta : float, default 1e-12
        Halving while 的 `theta` 下限；**`halving=False` 時無作用**。Default `1e-12`
        對應 `NNI_ha.m` line 166 實際值（不是 `NNI.m` line 166 的 `1e-8`、那是 dead code：
        NNI.m 的 halving while 在註解裡、`tol_theta` 從未被 evaluated）。
    maxit : int, default 200
        外層 Newton 硬上限（MATLAB default `100`；本 port 放寬到 200）。
    initial_vector : array-like 1-D, optional
        初始 eigenvector 估計，shape `(n,)`。`None` 時用
        `np.random.default_rng().random(n)`（**parity test 必須顯式傳入**）。
    record_history : bool, default False
        True 時回傳 history dict（第 7 個 return value）、含 9 個 key，
        長度 ``nit + 1``、index [0] 存初始化狀態。
    plot_res : bool, default False
        **不實作**（同 HONI）；kwarg 保留以維持 API 可擴充；傳 True 會
        `warnings.warn` 後忽略。
    matlab_compat : bool, default False
        **對 nni 本身是 no-op**；會傳進 `tpv / sp_Jaco_Ax / ten2mat`。

    Returns
    -------
    若 `record_history=False`：
        `(lambda_U, x, nit, lambda_L, outer_res_history, lambda_U_history)` — 6-tuple
        - ``lambda_U`` : float — 最大 eigenvalue 估計（λ 上界）
        - ``x`` : np.ndarray, 1-D `(n,)` — 對應 unit-norm eigenvector
        - ``nit`` : int — 外層 iter 完成次數（**Python 0-based**、MATLAB `nit_mat = nit + 1`）
        - ``lambda_L`` : float — 最小 eigenvalue 估計（λ 下界、res = 1 - λ_L/λ_U）
        - ``outer_res_history`` : np.ndarray, 1-D, length `nit + 1` — 每 iter 相對殘差
        - ``lambda_U_history`` : np.ndarray, 1-D, length `nit + 1` — 每 iter λ_U

    若 `record_history=True`：回傳 7-tuple `(..., history)`，history 是 dict：
        - ``x_history`` : `(n, nit+1)`
        - ``lambda_U_history`` : `(nit+1,)`
        - ``lambda_L_history`` : `(nit+1,)`
        - ``res_history`` : `(nit+1,)`
        - ``w_history`` : `(n, nit+1)` — 每 iter 的 RHS `x^(m-1)`；[:, 0] = NaN
        - ``y_history`` : `(n, nit+1)` — 每 iter 正規化 Newton 方向 `y = w_raw / ||w_raw||`；[:, 0] = NaN
        - ``chit_history`` : `(nit+1,)` — 累積 halving 計數；
          ``halving=False`` 恆 0（canonical），``halving=True`` 為跨 outer iter 的累積值
        - ``hit_per_outer_history`` : `(nit+1,)` — 每 iter 的 halving 計數；
          ``halving=False`` 恆 0（canonical），``halving=True`` 記第 i 次 outer iter 進
          halving while 的次數（§九 Q2/Q3：dead → live 欄位語意轉換、保留欄位名不改）
        - ``gmres_info_history`` : `(nit+1,) of int64` 若 ``linear_solver="gmres"``、否則 ``None``

    §四 Hazard checklist（手眼同步點、逐條核對 — 詳見 nni_hazard_analysis.md §四）
    --------------------------------------------------------------------------
    1. **假設 input 是正 M-tensor**、`x > 0` 沿 iteration 保持、**不主動投影** — §五.1
    2. **`diag(x^(m-2))` 用 `scipy.sparse.diags`**（不用 `np.diag`、避免大 dense 爆
       記憶體）、`M_shifted = B - (m-1)*λ_U*D` 整路 sparse — §四.5.1 + §五.2
    3. **`step_length` 兩個分支**（由 `halving` kwarg 切換）：
       - `halving=False`（NNI.m 原 line 174-186 整塊註解）：
         `θ=1; x_new = (m-2)*x + y; normalize; recompute temp, λ_U, λ_L; hit = 0`
       - `halving=True`（NNI_ha.m 原 line 174-186 active）：
         先 full step、若 `λ_U_new - λ_U > 1e-13` 則進 halving while（`θ /= 2`、
         重算 x_new / λ_U / λ_L、每 iter `hit += 1`、`θ < tol_theta` 時 `warnings.warn` 並 break）
       兩分支 `chit += hit`、`hit_per_outer_history[nit] = hit` 行為統一 — §五.5 + §九
    4. **`res = ones(maxit)` pre-fill = 1.0 的 assertion** 在收尾前加、對應 Q6；
       `lambda_L < 0` 時 `res > 1`、pre-fill 策略失效、assert fail — §五.3

    §五 NNI 專屬 fragility
    ---------------------
    當 `x_i → 0` 且 `m ≥ 3`、`x_i^(m-2) → 0`、`diag(x^(m-2))` 第 i 個對角元素 → 0、
    `M_shifted` 在第 i 列 **row-wise ill-conditioned**（對比 HONI 的 `λ_U*II - AA`
    是均勻 shift、收斂尾段 global near-singular）。若 parity test 看到 `w/y_history`
    在收斂尾段 rel error 爆、套 HONI 的三 Tier 框架（rtol fallback）。詳見
    `memory/feedback_honi_multi_fragility_propagation.md`。

    §九 Halving break caveat（僅 `halving=True` 時可能出現）
    -------------------------------------------------------
    當 halving while 因 `theta < tol_theta` break 時、返回的 `lambda_U_new` 可能仍
    `> lambda_U + 1e-13`（單調性被破壞）。`NNI_ha.m` main loop line 60 的 call site
    `[x, lambda_U, ...] = step_length(...)` **無 fallback 邏輯**、直接接受這個非單調值
    繼續下一輪 outer iteration、可能進入震盪或使 `res > 1` 觸發尾端 assert fail。
    此為 MATLAB NNI_ha.m 原碼行為、port 逐字照抄、**非 bug**。Python 端會在每次
    break 時發 `RuntimeWarning`（帶 outer iter 編號、方便 parity test 定位）。

    Raises
    ------
    ValueError
        - `m` 不是 int 或 `m < 2`
        - `tol <= 0`
        - `linear_solver` 不是 `'spsolve'` / `'gmres'`（特別處理 `'GTH'` 訊息）
        - `gmres` 回傳 `info < 0`（illegal input）
        - polymorphic A 的 shape 與 m 不相容
    AssertionError
        收尾 sanity：`res[:nit+1] <= 1.0 + 1e-10`。失敗表示 `lambda_L < 0`、AA 不
        是 clean M-tensor、pre-fill 策略假設破裂。
    """
    # ---- Input validation ----
    if linear_solver == "GTH":
        raise ValueError(
            "GTH solver not ported; use 'spsolve' or 'gmres' instead"
        )
    if linear_solver not in ("spsolve", "gmres"):
        raise ValueError(
            f"linear_solver must be 'spsolve' or 'gmres', got {linear_solver!r}"
        )
    if not isinstance(m, (int, np.integer)):
        raise ValueError(f"m must be int, got {type(m).__name__}")
    m = int(m)
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if maxit < 2:
        raise ValueError(f"maxit must be >= 2, got {maxit}")

    # ---- gmres options merge (user-facing names → scipy names at call site) ----
    if linear_solver == "gmres":
        opts = dict(_NNI_GMRES_DEFAULTS)
        if gmres_opts is not None:
            opts.update(gmres_opts)
    else:
        opts = None

    # ---- Polymorphic A dispatch (same as HONI) ----
    if issparse(A):
        if A.ndim != 2:
            raise ValueError(f"sparse A must be 2-D unfolding, got ndim={A.ndim}")
        n = A.shape[0]
        if A.shape[1] != n ** (m - 1):
            raise ValueError(
                f"sparse A shape {A.shape} incompatible with m={m}; "
                f"expected ({n}, {n ** (m - 1)})"
            )
        AA = A
    else:
        A_arr = np.asarray(A)
        if A_arr.ndim == 2:
            n = A_arr.shape[0]
            if A_arr.shape[1] != n ** (m - 1):
                raise ValueError(
                    f"A shape {A_arr.shape} incompatible with m={m}; "
                    f"expected ({n}, {n ** (m - 1)})"
                )
            AA = A_arr
        elif A_arr.ndim >= 3:
            n = A_arr.shape[0]
            if A_arr.ndim != m:
                raise ValueError(
                    f"tensor A.ndim={A_arr.ndim} but m={m} (need ndim == m)"
                )
            if not all(s == n for s in A_arr.shape):
                raise ValueError(
                    f"A must be square tensor (all dims == n={n}), got shape {A_arr.shape}"
                )
            AA = ten2mat(A_arr, k=0, matlab_compat=matlab_compat)
        else:
            raise ValueError(
                f"A must be 2-D unfolding or m-D tensor, got ndim={A_arr.ndim}"
            )

    # ---- Initial vector ----
    if initial_vector is None:
        rng = np.random.default_rng()
        initial_vector = rng.random(n)
    else:
        initial_vector = np.asarray(initial_vector, dtype=np.float64).ravel()
        if initial_vector.shape != (n,):
            raise ValueError(
                f"initial_vector shape {initial_vector.shape} != ({n},)"
            )

    if plot_res:
        warnings.warn(
            "plot_res is not implemented in this port; ignoring",
            stacklevel=2,
        )

    # ---- 初始化 (NNI.m line 29-41) ----
    x = initial_vector / np.linalg.norm(initial_vector)                          # line 31-32
    temp = tpv(AA, x, m, matlab_compat=matlab_compat) / (x ** (m - 1))           # line 33
    lambda_U = float(np.max(temp))                                                # line 34
    lambda_L = float(np.min(temp))                                                # line 35

    res = np.ones(maxit, dtype=np.float64)                                        # line 38 (res pre-fill)
    res[0] = (lambda_U - lambda_L) / lambda_U                                    # line 39 (0-based)
    nit = 0                                                                        # MATLAB nit=1 → 0-based
    chit = 0                                                                       # line 41 (canonical NNI.m: halving disabled → always 0)

    # ---- history buffers (always track lambda histories; others if record_history) ----
    lambda_U_history = np.full(maxit, np.nan, dtype=np.float64)
    lambda_U_history[0] = lambda_U
    lambda_L_history = np.full(maxit, np.nan, dtype=np.float64)
    lambda_L_history[0] = lambda_L

    if record_history:
        x_history = np.zeros((n, maxit), dtype=np.float64)
        x_history[:, 0] = x
        w_history = np.full((n, maxit), np.nan, dtype=np.float64)
        y_history = np.full((n, maxit), np.nan, dtype=np.float64)
        chit_history = np.zeros(maxit, dtype=np.float64)
        hit_per_outer_history = np.zeros(maxit, dtype=np.float64)
        if linear_solver == "gmres":
            gmres_info_history = np.zeros(maxit, dtype=np.int64)
        else:
            gmres_info_history = None

    _MAX_NIT = maxit - 1  # Python 0-based; MATLAB `nit < maxit` ≡ Python `nit < maxit - 1`

    # ---- 外層 Newton 迴圈 (NNI.m line 42-75) ----
    while np.min(res) > tol and nit < _MAX_NIT:
        nit += 1                                                                  # line 43

        # Jacobian + shifted matrix (NNI.m line 44, 46)
        B = sp_Jaco_Ax(AA, x, m, matlab_compat=matlab_compat)                     # line 44
        D = sp_diags(x ** (m - 2))                                                # §四 checklist #2: sparse, not np.diag
        M_shifted = B - (m - 1) * lambda_U * D                                    # line 46

        # RHS: x^(m-1) (NNI.m line 49)
        w_rhs = x ** (m - 1)

        # Linear solve: y_raw = (-M_shifted) \ w_rhs  (note the MINUS sign, line 49)
        if linear_solver == "spsolve":
            y_raw = spsolve((-M_shifted).tocsc(), w_rhs)
        else:  # "gmres"
            y_raw, info = gmres(
                -M_shifted, w_rhs,
                rtol=opts["tol"],
                atol=0.0,
                restart=opts["restart"],
                maxiter=opts["maxit"],
                M=opts["M"],
            )
            if info < 0:
                raise ValueError(
                    f"gmres illegal input at outer iter {nit} (info={info})"
                )
            if info > 0:
                warnings.warn(
                    f"gmres did not converge in {opts['maxit']} iters at "
                    f"outer iter {nit} (info={info})",
                    stacklevel=2,
                )
            if record_history:
                gmres_info_history[nit] = info

        # Normalize Newton direction (NNI.m line 56)
        y = y_raw / np.linalg.norm(y_raw)

        # step_length (NNI.m / NNI_ha.m line 148-187) — halving kwarg 切換兩分支
        # §四 checklist #3; §九 addendum (Phase 1) 詳述選項 A kwarg 設計
        # 一般式 (m-2)*x + y；m=3 退化成 x + y、m=2 退化成 y (不 special case、同 Q5 決定)
        if not halving:
            # --- canonical NNI.m: full step、無條件接受、hit≡0 (halving 在 MATLAB 端註解掉) ---
            x_new = (m - 2) * x + y
            x_new = x_new / np.linalg.norm(x_new)
            temp = tpv(AA, x_new, m, matlab_compat=matlab_compat) / (x_new ** (m - 1))
            lambda_U_new = float(np.max(temp))
            lambda_L_new = float(np.min(temp))
            hit = 0
        else:
            # --- NNI_ha.m: full-step overshoot (λ_U_new > λ_U + 1e-13) → halve theta ---
            # 逐行 mirror NNI_ha.m line 167-186、順序不動（hit++ 必須在 λ 更新之後、
            # 在 tol_theta break check 之前；這順序決定 break 時的 hit 值、parity 關鍵）
            hit = 0
            theta = 1.0
            x_new = (m - 2) * x + theta * y                                       # full step
            x_new = x_new / np.linalg.norm(x_new)
            temp = tpv(AA, x_new, m, matlab_compat=matlab_compat) / (x_new ** (m - 1))
            lambda_U_new = float(np.max(temp))
            lambda_L_new = float(np.min(temp))
            while lambda_U_new - lambda_U > 1e-13:
                theta = theta / 2.0
                x_new = (m - 2) * x + theta * y
                x_new = x_new / np.linalg.norm(x_new)
                temp = tpv(AA, x_new, m, matlab_compat=matlab_compat) / (x_new ** (m - 1))
                lambda_U_new = float(np.max(temp))
                lambda_L_new = float(np.min(temp))
                hit += 1
                if theta < tol_theta:
                    warnings.warn(
                        f"NNI_ha halving break at outer iter {nit}: can't find a "
                        f"suitable theta (theta={theta:.3e} < tol_theta={tol_theta:.0e}); "
                        f"returning non-monotone lambda_U (see nni_hazard_analysis.md §九)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
        chit += hit

        # Update state
        x = x_new
        lambda_U = lambda_U_new
        lambda_L = lambda_L_new

        # Residual (NNI.m line 63)
        res[nit] = (lambda_U - lambda_L) / lambda_U

        # History tracking
        lambda_U_history[nit] = lambda_U
        lambda_L_history[nit] = lambda_L
        if record_history:
            x_history[:, nit] = x
            w_history[:, nit] = w_rhs
            y_history[:, nit] = y
            hit_per_outer_history[nit] = hit
            chit_history[nit] = chit

    # ---- Upper-bound sanity: pre-fill value 1.0 must bound all written res ----
    # §四 checklist #4: fail 表示 lambda_L < 0、AA 非 clean M-tensor、pre-fill 策略失效
    assert np.all(res[:nit + 1] <= 1.0 + 1e-10), (
        f"res upper-bound violation: max res[:{nit+1}] = {res[:nit+1].max():.6e}, "
        f"pre-fill was 1.0 (should hold for clean M-tensor with lambda_L >= 0)"
    )

    outer_res_history = res[:nit + 1].copy()
    lambda_U_history_out = lambda_U_history[:nit + 1].copy()
    lambda_L_history_out = lambda_L_history[:nit + 1].copy()

    if record_history:
        history = {
            "x_history": x_history[:, :nit + 1].copy(),
            "lambda_U_history": lambda_U_history_out.copy(),
            "lambda_L_history": lambda_L_history_out.copy(),
            "res_history": outer_res_history.copy(),
            "w_history": w_history[:, :nit + 1].copy(),
            "y_history": y_history[:, :nit + 1].copy(),
            "chit_history": chit_history[:nit + 1].copy(),
            "hit_per_outer_history": hit_per_outer_history[:nit + 1].copy(),
            "gmres_info_history": (
                gmres_info_history[:nit + 1].copy()
                if gmres_info_history is not None else None
            ),
        }
        return (
            float(lambda_U), x, nit, float(lambda_L),
            outer_res_history, lambda_U_history_out, history,
        )

    return (
        float(lambda_U), x, nit, float(lambda_L),
        outer_res_history, lambda_U_history_out,
    )
