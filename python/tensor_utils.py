"""HNI / NNI 共用的張量工具函式。

本模組的公用函式遵循以下 port 慣例：

- **向量 shape**：輸入/輸出一律 1-D numpy array `(n,)`，不使用 `(n,1)` column
  vector。呼叫端如有 2-D 意圖請用明確的 `reshape(-1, 1)` 或 `np.outer` 表達。
- **Indexing**：Python 介面採 0-based（Python 自然慣例）。MATLAB 的 1-based
  語義只在 `matlab_compat=True` 路徑內部還原。
- **Column-major 對齊**：任何 `reshape` 呼叫只要在 parity 路徑上,一律顯式
  `order='F'`，匹配 MATLAB 的欄優先儲存。
"""
import numpy as np


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
          MATLAB 原碼的 scalar `1`；用 1-D shape 以符合本模組的 shape 慣例）
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
