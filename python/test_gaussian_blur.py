import numpy as np

from gaussian_blur import gaussian_blur


def test_delta_impulse():
    A = np.zeros((5, 5))
    A[2, 2] = 1.0

    B = gaussian_blur(A, sigma=1.0)

    assert B.shape == A.shape, f"shape mismatch: {B.shape} vs {A.shape}"
    assert np.isclose(B.sum(), 1.0, atol=1e-6), f"mass not preserved: sum={B.sum()}"
    assert B[2, 2] == B.max(), "center should be the maximum"
    assert np.allclose(B, B.T), "result should be symmetric"
    assert np.allclose(B, B[::-1, :]), "result should be symmetric top-bottom"

    print("test_delta_impulse passed")
    print("blurred output:")
    print(np.round(B, 4))


if __name__ == "__main__":
    test_delta_impulse()
