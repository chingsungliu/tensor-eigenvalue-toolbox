import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_blur(A: np.ndarray, sigma: float, matlab_compat: bool = False) -> np.ndarray:
    """Apply Gaussian blur to A.

    matlab_compat=True configures scipy to match matlab_exercise/gaussian_blur.m
    bit-for-bit: mode='constant' matches conv2's zero-padding, truncate=3.0
    matches the MATLAB kernel half-width ceil(3*sigma). Used by test_parity.py
    to verify numerical agreement. Leave False for normal use — scipy's defaults
    (reflect boundary, truncate=4.0) are the physically saner choice.
    """
    if matlab_compat:
        return gaussian_filter(A, sigma=sigma, mode="constant", cval=0, truncate=3.0)
    return gaussian_filter(A, sigma=sigma)
