"""Sanity tests for tensor_utils — 不依賴 MATLAB reference，驗證 Python
實作本身合理（shape、邊界值、已知輸入的正確結果、輸入檢查）。"""
import numpy as np
from scipy.sparse import csr_matrix, issparse

from tensor_utils import tenpow, tpv, sp_tendiag, ten2mat, sp_Jaco_Ax


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


def test_ten2mat_basic():
    #
    # --- (1) Paper derivation 驗證：(2,2,2) 小例子 ---
    #
    T = np.arange(1, 9).reshape(2, 2, 2)  # C-order fill, entries 1..8
    B_correct = ten2mat(T, k=0)
    expected_correct = np.array([[1, 3, 2, 4], [5, 7, 6, 8]])
    assert B_correct.shape == (2, 4)
    assert np.array_equal(B_correct, expected_correct), (
        f"Paper derivation failed: expected {expected_correct}, got {B_correct}"
    )

    #
    # --- (2) Column-major vs row-major 對比（Trap #4 守門員）---
    # 直接比 ten2mat(T) 和人工用 order='C' 的 reshape 結果，證明兩者不同。
    # 若有人把 ten2mat 的 order='F' 改成 'C'，這個斷言會立刻失敗。
    #
    for shape in [(2, 2, 2), (3, 3, 3), (2, 3, 4)]:
        T = np.arange(1, int(np.prod(shape)) + 1).reshape(shape)
        B_f = ten2mat(T, k=0)  # 正確：order='F'
        n0 = shape[0]
        other = int(np.prod(shape[1:]))
        B_c_wrong = T.reshape(n0, other, order="C")  # 錯誤版
        assert not np.array_equal(B_f, B_c_wrong), (
            f"shape {shape}: F-order and C-order MUST differ for non-diagonal "
            f"tensor — if this assertion ever passes, someone broke the order='F'"
        )

    # (2,2,2) 特例可以 hard-code 兩邊對比值
    T = np.arange(1, 9).reshape(2, 2, 2)
    B_c_wrong = T.reshape(2, 4, order="C")
    assert np.array_equal(B_c_wrong, np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))

    #
    # --- (3) 公式式驗證：B[i, j] == T[i, *unravel(j, shape[1:], order='F')] ---
    # 對三個 shape 都做完整檢查。
    #
    for shape in [(2, 2, 2), (3, 3, 3), (2, 3, 4)]:
        T = np.arange(1, int(np.prod(shape)) + 1).reshape(shape)
        B = ten2mat(T, k=0)
        n0 = shape[0]
        other = int(np.prod(shape[1:]))
        assert B.shape == (n0, other), f"shape {shape}: wrong output shape {B.shape}"
        for i in range(n0):
            for j in range(other):
                idx_rest = np.unravel_index(j, shape[1:], order="F")
                # T[(i,) + idx_rest] accessed element-wise
                expected = T[(i,) + idx_rest]
                assert B[i, j] == expected, (
                    f"shape {shape}, (i={i}, j={j}): B={B[i,j]} vs T[{(i,) + idx_rest}]={expected}"
                )

    #
    # --- (4) 任意 mode k 驗證（k=1, k=2 on asymmetric shape）---
    #
    shape = (2, 3, 4)
    T = np.arange(1, int(np.prod(shape)) + 1).reshape(shape)
    for k in range(3):
        B_k = ten2mat(T, k=k)
        # Expected shape
        rest = [s for idx, s in enumerate(shape) if idx != k]
        assert B_k.shape == (shape[k], int(np.prod(rest))), (
            f"k={k}: wrong shape {B_k.shape}, expected ({shape[k]}, {int(np.prod(rest))})"
        )
        # 公式式驗證
        for i in range(shape[k]):
            for j in range(int(np.prod(rest))):
                idx_rest = np.unravel_index(j, tuple(rest), order="F")
                full_idx = list(idx_rest)
                full_idx.insert(k, i)
                assert B_k[i, j] == T[tuple(full_idx)], (
                    f"k={k}, (i={i}, j={j}): mismatch"
                )

    #
    # --- (5) dtype 繼承 ---
    #
    for dtype in [np.float32, np.float64, np.int32]:
        T = np.ones((2, 2, 2), dtype=dtype)
        B = ten2mat(T, k=0)
        assert B.dtype == dtype, f"dtype {dtype} not preserved, got {B.dtype}"

    #
    # --- (6) 輸入驗證 ---
    #
    # k out of range
    raised = False
    try:
        ten2mat(np.ones((3, 3, 3)), k=5)
    except ValueError as e:
        raised = True
        assert "[0, 3)" in str(e) or "k must" in str(e)
    assert raised, "k out of range should raise"

    raised = False
    try:
        ten2mat(np.ones((3, 3, 3)), k=-1)
    except ValueError as e:
        raised = True
    assert raised, "negative k should raise"

    #
    # --- (7) matlab_compat no-op ---
    #
    T = np.arange(1, 28).reshape(3, 3, 3)
    B1 = ten2mat(T, k=0, matlab_compat=False)
    B2 = ten2mat(T, k=0, matlab_compat=True)
    assert np.array_equal(B1, B2)

    print("test_ten2mat_basic passed")


def test_sp_Jaco_Ax_basic():
    #
    # --- (1) m=1 邊界：F(x) 是常數，Jacobian = 0 矩陣 (n, n) ---
    #
    x = np.array([1.0, 2.0, 3.0])
    AA = np.ones((3, 1))  # shape (n, n^0) = (n, 1) for m=1
    J = sp_Jaco_Ax(AA, x, 1)
    assert issparse(J)
    assert J.shape == (3, 3)
    assert J.nnz == 0

    #
    # --- (2) m=2 case：F(x) = AA @ x，故 dF/dx = AA ---
    #
    AA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x = np.array([2.0, 3.0, 5.0])
    J = sp_Jaco_Ax(AA, x, 2)
    Jd = J.toarray() if issparse(J) else np.asarray(J)
    assert Jd.shape == (3, 3)
    assert np.allclose(Jd, AA), f"m=2 Jacobian should equal AA, got {Jd}"

    #
    # --- (3) m=3 手算：J = AA @ (kron(I, x) + kron(x, I)) ---
    #
    AA = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # (2, 4)
    x = np.array([2.0, 3.0])
    J = sp_Jaco_Ax(AA, x, 3)
    Jd = J.toarray() if issparse(J) else np.asarray(J)
    I = np.eye(2)
    x_col = x.reshape(-1, 1)
    expected_kron_sum = np.kron(I, x_col) + np.kron(x_col, I)
    expected_J = AA @ expected_kron_sum
    assert np.allclose(Jd, expected_J), (
        f"m=3 Jacobian mismatch:\n  got {Jd}\n  expected {expected_J}"
    )

    #
    # --- (4) Linearity in AA: J(c * AA) = c * J(AA) ---
    #
    rng = np.random.default_rng(123)
    AA_r = rng.standard_normal((3, 9))
    x_r = rng.standard_normal(3)
    Ja = sp_Jaco_Ax(AA_r, x_r, 3)
    Jb = sp_Jaco_Ax(2.5 * AA_r, x_r, 3)
    Jad = Ja.toarray() if issparse(Ja) else np.asarray(Ja)
    Jbd = Jb.toarray() if issparse(Jb) else np.asarray(Jb)
    assert np.allclose(Jbd, 2.5 * Jad), "Linearity in AA failed"

    #
    # --- (5) Sparse vs dense AA input 結果一致 ---
    #
    AA_d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    AA_s = csr_matrix(AA_d)
    x = np.array([1.0, 2.0, 3.0])
    J_d = sp_Jaco_Ax(AA_d, x, 2)
    J_s = sp_Jaco_Ax(AA_s, x, 2)
    J_dd = J_d.toarray() if issparse(J_d) else np.asarray(J_d)
    J_sd = J_s.toarray() if issparse(J_s) else np.asarray(J_s)
    assert np.allclose(J_dd, J_sd), "dense vs sparse AA input gave different J"

    #
    # --- (6) 輸入驗證 ---
    #
    raised = False
    try:
        sp_Jaco_Ax(np.ones(5), np.ones(5), 2)  # 1-D AA
    except ValueError as e:
        raised = True
        assert "2-D" in str(e)
    assert raised, "1-D AA should raise"

    raised = False
    try:
        sp_Jaco_Ax(np.ones((3, 9)), np.array([[1.0, 2.0, 3.0]]), 3)  # 2-D x
    except ValueError as e:
        raised = True
        assert "1-D" in str(e)
    assert raised, "2-D x should raise"

    raised = False
    try:
        sp_Jaco_Ax(np.ones((3, 3)), np.ones(3), 0)  # m = 0
    except ValueError as e:
        raised = True
        assert ">= 1" in str(e)
    assert raised, "m=0 should raise"

    #
    # --- (7) matlab_compat no-op ---
    #
    rng = np.random.default_rng(456)
    AA = rng.standard_normal((3, 9))
    x = np.array([1.5, 2.5, 3.5])
    Ja = sp_Jaco_Ax(AA, x, 3, matlab_compat=False)
    Jb = sp_Jaco_Ax(AA, x, 3, matlab_compat=True)
    Jad = Ja.toarray() if issparse(Ja) else np.asarray(Ja)
    Jbd = Jb.toarray() if issparse(Jb) else np.asarray(Jb)
    assert np.allclose(Jad, Jbd)

    print("test_sp_Jaco_Ax_basic passed")


if __name__ == "__main__":
    test_tenpow_basic()
    test_tpv_basic()
    test_sp_tendiag_basic()
    test_ten2mat_basic()
    test_sp_Jaco_Ax_basic()
