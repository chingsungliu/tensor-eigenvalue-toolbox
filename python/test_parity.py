from pathlib import Path

import numpy as np
from scipy.io import loadmat

from gaussian_blur import gaussian_blur

REF_PATH = Path(__file__).resolve().parent.parent / "matlab_exercise" / "reference.mat"


def test_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"reference.mat not found at {REF_PATH}. "
            "Run matlab_exercise/generate_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))
    A = data["A"]
    B_matlab = data["B"]
    sigma = float(data["sigma"].item())

    B_py = gaussian_blur(A, sigma, matlab_compat=True)

    diff = np.abs(B_py - B_matlab)
    max_err = float(diff.max())

    if max_err > 1e-10:
        mean_err = float(diff.mean())
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"parity test FAILED: "
            f"max error = {max_err:.3e}, "
            f"mean error = {mean_err:.3e}, "
            f"worst position = {worst} "
            f"(B_matlab={B_matlab[worst]:.6f}, B_py={B_py[worst]:.6f})"
        )

    print("parity check in MATLAB compatibility mode (mode='constant', truncate=3.0)")
    print(f"parity test passed, max error = {max_err:.3e}")


if __name__ == "__main__":
    test_parity()
