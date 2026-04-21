"""Sanity tests for tensor_utils — 不依賴 MATLAB reference，驗證 Python
實作本身合理（shape、邊界值、已知輸入的正確結果、輸入檢查）。"""
import numpy as np

from tensor_utils import tenpow


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


if __name__ == "__main__":
    test_tenpow_basic()
