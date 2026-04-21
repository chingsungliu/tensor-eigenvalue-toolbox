"""Parity test: 驗證 Python 端 tenpow 跟 MATLAB reference 逐位元一致。

前置：MATLAB 端已跑過 matlab_ref/hni/generate_tenpow_reference.m，產出
tenpow_reference.mat。
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from tensor_utils import tenpow

REF_PATH = (
    Path(__file__).resolve().parent.parent
    / "matlab_ref"
    / "hni"
    / "tenpow_reference.mat"
)

TOLERANCE = 1e-10  # 通過門檻；目標 ~1e-16


def test_tenpow_parity():
    if not REF_PATH.exists():
        raise FileNotFoundError(
            f"tenpow_reference.mat not found at {REF_PATH}.\n"
            "Run matlab_ref/hni/generate_tenpow_reference.m in MATLAB first."
        )

    data = loadmat(str(REF_PATH))

    # MATLAB column vectors load as (n, 1); squeeze to 1-D per project convention.
    x_matlab_shape = data["x"].shape
    x = data["x"].ravel()

    print(f"Loaded reference from {REF_PATH.name}")
    print(f"  x: MATLAB shape {x_matlab_shape} -> Python shape {x.shape}, x[0] = {x[0]:.15f}")

    cases = [
        (2, data["tp2"]),
        (3, data["tp3"]),
        (4, data["tp4"]),
    ]

    all_passed = True
    for p, tp_matlab_raw in cases:
        tp_matlab = tp_matlab_raw.ravel()
        tp_py = tenpow(x, p, matlab_compat=True)

        if tp_py.shape != tp_matlab.shape:
            print(
                f"  p={p}: FAIL shape mismatch -- "
                f"MATLAB raw {tp_matlab_raw.shape} ravel {tp_matlab.shape}, "
                f"Python {tp_py.shape}"
            )
            all_passed = False
            continue

        diff = np.abs(tp_py - tp_matlab)
        max_err = float(diff.max())

        if max_err > TOLERANCE:
            idx = int(np.argmax(diff))
            print(
                f"  p={p}: FAIL max error {max_err:.3e} > tolerance {TOLERANCE:.0e}\n"
                f"         worst at index {idx}: "
                f"MATLAB = {tp_matlab[idx]:.15e}, Python = {tp_py[idx]:.15e}\n"
                f"         MATLAB raw shape {tp_matlab_raw.shape}, Python shape {tp_py.shape}"
            )
            all_passed = False
        else:
            print(
                f"  p={p}: PASS max error {max_err:.3e} "
                f"(length {len(tp_py)}, tolerance {TOLERANCE:.0e})"
            )

    if not all_passed:
        raise AssertionError("tenpow parity test FAILED -- see log above")

    print("tenpow parity test passed")


if __name__ == "__main__":
    test_tenpow_parity()
