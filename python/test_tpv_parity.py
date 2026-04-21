"""Parity test: 驗證 Python 端 tpv 跟 MATLAB reference 逐位元一致。

前置：MATLAB 端已跑過 matlab_ref/hni/generate_layer1_reference.m，產出
tpv_reference.mat。
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from tensor_utils import tpv

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "tpv_reference.mat"
)

TOLERANCE = 1e-10  # 通過門檻；期望結果落在 1e-14 ~ 1e-16 量級


def test_tpv_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"tpv_reference.mat not found at {REF_PATH}.\n"
            "Run matlab_ref/hni/generate_layer1_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    print(f"Loaded reference from {REF_PATH.name}")

    all_passed = True
    for i in [1, 2, 3]:
        AA = data[f"AA_tpv{i}"]
        x = data[f"x_tpv{i}"].ravel()
        m = int(data[f"m_tpv{i}"].item())
        n = int(data[f"n_tpv{i}"].item())
        y_matlab_raw = data[f"y_tpv{i}"]
        y_matlab = y_matlab_raw.ravel()

        y_py = tpv(AA, x, m, matlab_compat=True)

        if y_py.shape != y_matlab.shape:
            print(
                f"  case{i} (m={m}, n={n}): FAIL shape mismatch -- "
                f"Python {y_py.shape}, MATLAB {y_matlab.shape} (raw {y_matlab_raw.shape})"
            )
            all_passed = False
            continue

        diff = np.abs(y_py - y_matlab)
        max_err = float(diff.max())

        if max_err > TOLERANCE:
            idx = int(np.argmax(diff))
            print(
                f"  case{i} (m={m}, n={n}): FAIL max_err={max_err:.3e} > tol={TOLERANCE:.0e}\n"
                f"           worst at index {idx}: "
                f"MATLAB={y_matlab[idx]:.15e}, Python={y_py[idx]:.15e}\n"
                f"           AA shape {AA.shape}, x shape {x.shape}, y_matlab raw {y_matlab_raw.shape}"
            )
            all_passed = False
        else:
            print(
                f"  case{i} (m={m}, n={n}): PASS max_err={max_err:.3e} "
                f"(AA shape {AA.shape}, y length {len(y_py)})"
            )

    if not all_passed:
        raise AssertionError("tpv parity test FAILED -- see log above")

    print("tpv parity test passed")


if __name__ == "__main__":
    test_tpv_parity()
