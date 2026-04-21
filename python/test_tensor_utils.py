"""Sanity tests for tensor_utils — 不依賴 MATLAB reference，驗證 Python
實作本身合理（shape、邊界值、已知輸入的正確結果、輸入檢查）。"""
import numpy as np
from scipy.sparse import csr_matrix, issparse

from tensor_utils import tenpow, tpv, sp_tendiag


def test_tenpow_basic():
    # p = 1 → identity
    x = np.array([3.0, -1.0, 2.5])
    assert np.array_equal(tenpow(x, 1), x), "p=1 should return x unchanged"

    # shape: tenpow(x, p) has length n**p for p >= 1
    x = np.array([1.0, 2.0])  # n = 2
    for p in range(1, 5):
        result = tenpow(x, p)
        assert result.shape == (2**p,), f"p={p}: expected shape ({2**p},), got {result.shape}"

    # 已知結果 p=2: [1,2] ⊗ [1,2] = [1, 2, 2, 4]
    expected_p2 = np.array([1.0, 2.0, 2.0, 4.0])
    got = tenpow(np.array([1.0, 2.0]), 2)
    assert np.array_equal(got, expected_p2), f"p=2: expected {expected_p2}, got {got}"

    # 已知結果 p=3: [1,2] ⊗ [1,2] ⊗ [1,2]
    expected_p3 = np.array([1.0, 2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 8.0])
    got = tenpow(np.array([1.0, 2.0]), 3)
    assert np.array_equal(got, expected_p3), f"p=3: expected {expected_p3}, got {got}"

    # p = 0 邊界：回傳 shape (1,) 內含 1.0
    got = tenpow(np.array([7.0]), 0)
    assert got.shape == (1,), f"p=0: expected shape (1,), got {got.shape}"
    assert got[0] == 1.0, f"p=0: expected [1.0], got {got}"

    # dtype 繼承：float32 input → float32 output
    x_f32 = np.array([1.0, 2.0], dtype=np.float32)
    got = tenpow(x_f32, 3)
    assert got.dtype == np.float32, f"dtype should be preserved: got {got.dtype}"

    # 2-D input 應 raise
    raised = False
    try:
        tenpow(np.array([[1.0, 2.0]]), 2)
    except ValueError as e:
        raised = True
        assert "1-D" in str(e)
    assert raised, "2-D input should raise ValueError"

    # 負 p 應 raise
    raised = False
    try:
        tenpow(np.array([1.0, 2.0]), -1)
    except ValueError as e:
        raised = True
        assert ">= 0" in str(e)
    assert raised, "negative p should raise ValueError"

    # matlab_compat=True 和 False 結果相同（本函式 flag 是 no-op）
    x = np.array([0.5, -0.3, 1.2])
    for p in [1, 2, 3]:
        a = tenpow(x, p, matlab_compat=False)
        b = tenpow(x, p, matlab_compat=True)
        assert np.array_equal(a, b), f"p={p}: matlab_compat should be no-op for tenpow"

    print("test_tenpow_basic passed")


def test_tpv_basic():
    # 恆等 AA (m=2): tpv(I, x, 2) = I @ x = x
    n = 4
    I = np.eye(n)
    x = np.array([1.5, -0.3, 2.0, 0.7])
    y = tpv(I, x, 2)
    assert y.shape == (n,)
    assert np.allclose(y, x), f"tpv(I, x, 2) should equal x, got {y}"

    # 零 AA: y 全零
    AA_zero = np.zeros((n, n ** (3 - 1)))
    y = tpv(AA_zero, x, 3)
    assert y.shape == (n,)
    assert np.all(y == 0)

    # 已知結果：AA 全 1、x 全 1、m=3 → y = [n^(m-1), n^(m-1), ..., n^(m-1)]
    AA_ones = np.ones((n, n ** (3 - 1)))
    x_ones = np.ones(n)
    y = tpv(AA_ones, x_ones, 3)
    assert np.allclose(y, np.full(n, n ** (3 - 1))), f"expected {n**(3-1)} everywhere, got {y}"

    # m=2 退化：tenpow(x, 1) = x，所以 y = AA @ x
    AA = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([1.0, 1.0])
    y = tpv(AA, x2, 2)
    assert np.allclose(y, AA @ x2)

    # Sparse AA：應輸出 1-D dense ndarray
    AA_sp = csr_matrix(AA)
    y_sp = tpv(AA_sp, x2, 2)
    assert isinstance(y_sp, np.ndarray)
    assert y_sp.ndim == 1
    assert np.allclose(y_sp, AA @ x2)

    # 輸入驗證：2-D x 拒收
    raised = False
    try:
        tpv(np.ones((3, 3)), np.array([[1.0, 2.0, 3.0]]), 2)
    except ValueError as e:
        raised = True
        assert "1-D" in str(e)
    assert raised, "2-D x should raise"

    # 輸入驗證：1-D AA 拒收
    raised = False
    try:
        tpv(np.ones(5), np.ones(5), 2)
    except ValueError as e:
        raised = True
        assert "2-D" in str(e)
    assert raised, "1-D AA should raise"

    # 輸入驗證：m = 0 拒收
    raised = False
    try:
        tpv(np.ones((3, 1)), np.ones(3), 0)
    except ValueError as e:
        raised = True
        assert ">= 1" in str(e)
    assert raised, "m=0 should raise"

    # matlab_compat no-op
    y_a = tpv(AA, x2, 2, matlab_compat=False)
    y_b = tpv(AA, x2, 2, matlab_compat=True)
    assert np.array_equal(y_a, y_b)

    print("test_tpv_basic passed")


def test_sp_tendiag_basic():
    # n=3, m=3: 期望 shape (3, 9)，d 在 (0,0), (1,4), (2,8)
    d = np.array([5.0, 7.0, 11.0])
    D = sp_tendiag(d, 3)
    assert issparse(D)
    assert D.shape == (3, 9)
    assert D.nnz == 3
    Dd = D.toarray()
    assert Dd[0, 0] == 5.0
    assert Dd[1, 4] == 7.0
    assert Dd[2, 8] == 11.0
    # 其他位置全零
    mask = np.ones((3, 9), dtype=bool)
    for i, j in [(0, 0), (1, 4), (2, 8)]:
        mask[i, j] = False
    assert np.all(Dd[mask] == 0)

    # n=2, m=3: 期望 (2, 4)，d 在 (0,0), (1,3)
    d = np.array([1.0, 2.0])
    Dd = sp_tendiag(d, 3).toarray()
    assert Dd.shape == (2, 4)
    assert Dd[0, 0] == 1.0
    assert Dd[1, 3] == 2.0

    # n=5, m=4: 期望 (5, 125)，stride = (125-1)/4 = 31
    d = np.arange(1, 6).astype(float)
    Dd = sp_tendiag(d, 4).toarray()
    assert Dd.shape == (5, 125)
    for i, val in enumerate(d):
        assert Dd[i, i * 31] == val, f"case n=5,m=4 position (i={i}, col={i*31}) wrong"

    # m=1 邊界：shape (n, 1)，D 就是 d 做成 column
    d = np.array([1.0, 2.0, 3.0])
    Dd = sp_tendiag(d, 1).toarray()
    assert Dd.shape == (3, 1)
    assert np.allclose(Dd.flatten(), d)

    # n=1 邊界：shape (1, 1)
    d = np.array([42.0])
    Dd = sp_tendiag(d, 5).toarray()
    assert Dd.shape == (1, 1)
    assert Dd[0, 0] == 42.0

    # 輸入驗證：2-D d
    raised = False
    try:
        sp_tendiag(np.array([[1.0, 2.0]]), 3)
    except ValueError as e:
        raised = True
        assert "1-D" in str(e)
    assert raised, "2-D d should raise"

    # 輸入驗證：m = 0
    raised = False
    try:
        sp_tendiag(np.array([1.0, 2.0]), 0)
    except ValueError as e:
        raised = True
        assert ">= 1" in str(e)
    assert raised, "m=0 should raise"

    # matlab_compat no-op（輸出完全一樣）
    d = np.array([1.0, 2.0, 3.0])
    D1 = sp_tendiag(d, 3, matlab_compat=False).toarray()
    D2 = sp_tendiag(d, 3, matlab_compat=True).toarray()
    assert np.array_equal(D1, D2)

    print("test_sp_tendiag_basic passed")


if __name__ == "__main__":
    test_tenpow_basic()
    test_tpv_basic()
    test_sp_tendiag_basic()
