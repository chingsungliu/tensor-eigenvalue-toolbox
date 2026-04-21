"""Parity test: 驗證 Python 端 ten2mat 跟 MATLAB reference 逐位元一致。

前置：MATLAB 端已跑過 matlab_ref/hni/generate_layer2_reference.m，產出
ten2mat_reference.mat。

本 test 是整個 HNI port 的 **column-major 主檢查點**。任何微小的 reshape
order 錯誤都會在此報出巨大誤差（見 tensor_utils.ten2mat 的 Paper derivation）。
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from tensor_utils import ten2mat

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "ten2mat_reference.mat"
)

TOLERANCE = 1e-10  # 通過門檻；目標 0.000e+00（純 permutation + copy，無浮點運算）


def test_ten2mat_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"ten2mat_reference.mat not found at {REF_PATH}.\n"
            "Run matlab_ref/hni/generate_layer2_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    print(f"Loaded reference from {REF_PATH.name}")

    all_passed = True
    for i in [1, 2, 3]:
        T = data[f"T{i}"]
        k_matlab = int(data[f"k{i}"].item())  # MATLAB 1-based
        k_py = k_matlab - 1                    # Python 0-based
        n = int(data[f"n{i}"].item())
        m = int(data[f"m{i}"].item())
        B_matlab = np.asarray(data[f"B{i}"])

        B_py = ten2mat(T, k=k_py, matlab_compat=True)

        if B_py.shape != B_matlab.shape:
            print(
                f"  case{i} (n={n}, m={m}, k_matlab={k_matlab}): FAIL shape mismatch\n"
                f"           Python {B_py.shape}, MATLAB {B_matlab.shape}\n"
                f"           T shape (loaded): {T.shape}"
            )
            all_passed = False
            continue

        diff = np.abs(B_py - B_matlab)
        max_err = float(diff.max())

        if max_err > TOLERANCE:
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            print(
                f"  case{i} (n={n}, m={m}, k_matlab={k_matlab}): "
                f"FAIL max_err={max_err:.3e} > tol={TOLERANCE:.0e}\n"
                f"           worst at {idx}: "
                f"MATLAB={B_matlab[idx]:.15e}, Python={B_py[idx]:.15e}\n"
                f"           T shape {T.shape}, B shape {B_py.shape}\n"
                f"           >>> 這幾乎可以確定是 column-major / reshape order 問題；\n"
                f"           >>> 檢查 tensor_utils.ten2mat 的 reshape 是否用 order='F'"
            )
            all_passed = False
        else:
            print(
                f"  case{i} (n={n}, m={m}, k_matlab={k_matlab}): "
                f"PASS max_err={max_err:.3e} "
                f"(T shape {T.shape}, B shape {B_py.shape})"
            )

    if not all_passed:
        raise AssertionError("ten2mat parity test FAILED -- see log above")

    print("ten2mat parity test passed -- column-major checkpoint cleared")


if __name__ == "__main__":
    test_ten2mat_parity()
