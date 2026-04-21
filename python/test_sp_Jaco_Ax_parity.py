"""Parity test: 驗證 Python 端 sp_Jaco_Ax 跟 MATLAB reference 逐位元一致。

前置：MATLAB 端已跑過 matlab_ref/hni/generate_layer2_reference.m，產出
sp_Jaco_Ax_reference.mat。
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from tensor_utils import sp_Jaco_Ax

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "sp_Jaco_Ax_reference.mat"
)

TOLERANCE = 1e-10  # 通過門檻；期望結果 ~1e-14 ~ 1e-15（浮點加總累積）


def test_sp_Jaco_Ax_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"sp_Jaco_Ax_reference.mat not found at {REF_PATH}.\n"
            "Run matlab_ref/hni/generate_layer2_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    print(f"Loaded reference from {REF_PATH.name}")

    all_passed = True
    for i in [1, 2, 3]:
        AA = data[f"AA_j{i}"]
        x = data[f"xj{i}"].ravel()
        m = int(data[f"mj{i}"].item())
        n = int(data[f"nj{i}"].item())
        J_matlab_raw = data[f"J{i}"]

        if issparse(J_matlab_raw):
            J_matlab = J_matlab_raw.toarray()
            matlab_stored = f"sparse nnz={J_matlab_raw.nnz}"
        else:
            J_matlab = np.asarray(J_matlab_raw)
            matlab_stored = "dense"

        J_py = sp_Jaco_Ax(AA, x, m, matlab_compat=True)
        J_py_dense = J_py.toarray() if issparse(J_py) else np.asarray(J_py)

        if J_py_dense.shape != J_matlab.shape:
            print(
                f"  case{i} (n={n}, m={m}): FAIL shape mismatch\n"
                f"           Python {J_py_dense.shape}, MATLAB {J_matlab.shape}"
            )
            all_passed = False
            continue

        diff = np.abs(J_py_dense - J_matlab)
        max_err = float(diff.max())

        if max_err > TOLERANCE:
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            print(
                f"  case{i} (n={n}, m={m}): FAIL max_err={max_err:.3e} > tol={TOLERANCE:.0e}\n"
                f"           worst at {idx}: "
                f"MATLAB={J_matlab[idx]:.15e}, Python={J_py_dense[idx]:.15e}\n"
                f"           shape {J_py_dense.shape}, MATLAB {matlab_stored}, "
                f"Python {'sparse' if issparse(J_py) else 'dense'}"
            )
            all_passed = False
        else:
            print(
                f"  case{i} (n={n}, m={m}): PASS max_err={max_err:.3e} "
                f"(shape {J_py_dense.shape}, MATLAB {matlab_stored})"
            )

    if not all_passed:
        raise AssertionError("sp_Jaco_Ax parity test FAILED -- see log above")

    print("sp_Jaco_Ax parity test passed -- layer 2 complete")


if __name__ == "__main__":
    test_sp_Jaco_Ax_parity()
