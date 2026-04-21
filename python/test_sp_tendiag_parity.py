"""Parity test: 驗證 Python 端 sp_tendiag 跟 MATLAB reference 逐位元一致。

前置：MATLAB 端已跑過 matlab_ref/hni/generate_layer1_reference.m，產出
sp_tendiag_reference.mat。
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

from tensor_utils import sp_tendiag

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "sp_tendiag_reference.mat"
)

TOLERANCE = 1e-10  # 通過門檻；期望結果應該 = 0（純 scatter，無浮點運算）


def test_sp_tendiag_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"sp_tendiag_reference.mat not found at {REF_PATH}.\n"
            "Run matlab_ref/hni/generate_layer1_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    print(f"Loaded reference from {REF_PATH.name}")

    all_passed = True
    for i in [1, 2, 3]:
        d = data[f"d_sp{i}"].ravel()
        m = int(data[f"m_sp{i}"].item())
        D_matlab_raw = data[f"D_sp{i}"]

        # MATLAB sparse loads as scipy.sparse; dense if originally dense
        if issparse(D_matlab_raw):
            D_matlab = D_matlab_raw.toarray()
            matlab_stored = f"sparse nnz={D_matlab_raw.nnz}"
        else:
            D_matlab = np.asarray(D_matlab_raw)
            matlab_stored = "dense"

        D_py = sp_tendiag(d, m, matlab_compat=True)
        D_py_dense = D_py.toarray()

        if D_py_dense.shape != D_matlab.shape:
            print(
                f"  case{i} (n={len(d)}, m={m}): FAIL shape mismatch -- "
                f"Python {D_py_dense.shape}, MATLAB {D_matlab.shape}"
            )
            all_passed = False
            continue

        diff = np.abs(D_py_dense - D_matlab)
        max_err = float(diff.max())

        if max_err > TOLERANCE:
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            print(
                f"  case{i} (n={len(d)}, m={m}): FAIL max_err={max_err:.3e} > tol={TOLERANCE:.0e}\n"
                f"           worst at {idx}: "
                f"MATLAB={D_matlab[idx]:.15e}, Python={D_py_dense[idx]:.15e}\n"
                f"           shape {D_py_dense.shape}, Python nnz={D_py.nnz}, MATLAB {matlab_stored}"
            )
            all_passed = False
        else:
            print(
                f"  case{i} (n={len(d)}, m={m}): PASS max_err={max_err:.3e} "
                f"(shape {D_py_dense.shape}, Python nnz {D_py.nnz}, MATLAB {matlab_stored})"
            )

    if not all_passed:
        raise AssertionError("sp_tendiag parity test FAILED -- see log above")

    print("sp_tendiag parity test passed")


if __name__ == "__main__":
    test_sp_tendiag_parity()
