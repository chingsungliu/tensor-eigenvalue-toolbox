"""HNI / NNI 共用的張量工具函式。

本模組的公用函式遵循以下 port 慣例：

- **向量 shape**：輸入/輸出一律 1-D numpy array `(n,)`，不使用 `(n,1)` column
  vector。呼叫端如有 2-D 意圖請用明確的 `reshape(-1, 1)` 或 `np.outer` 表達。
- **Indexing**：Python 介面採 0-based（Python 自然慣例）。MATLAB 的 1-based
  語義只在 `matlab_compat=True` 路徑內部還原。
- **Column-major 對齊**：任何 `reshape` 呼叫只要在 parity 路徑上，一律顯式
  `order='F'`，匹配 MATLAB 的欄優先儲存。
"""
import numpy as np
from scipy.sparse import csr_matrix, issparse


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
        的 `p`；本函式內部做 `tenpow(x, m - 1)`，把 tensor order 換成
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
