"""Newton's method for x^2 - 2 = 0 — POC for Layer 3 per-iteration parity.

Ported verbatim from matlab_ref/poc_iteration/sqrt2_newton.m. The point of
this POC is NOT to solve sqrt(2) (MATLAB and Python both have `sqrt`) — it
is to establish the **逐 iteration 比對** framework that Multi / HONI / NNI
will reuse. Returns the full iteration history so the parity test can do
step-by-step comparison against MATLAB's saved history.
"""
import numpy as np


def sqrt2_newton(x0=1.0, n_iter=10, matlab_compat=False):
    """Newton iteration for x^2 - 2 = 0: x_{n+1} = x_n - (x_n^2 - 2) / (2 * x_n).

    Parameters
    ----------
    x0 : float, default 1.0
        Initial guess.
    n_iter : int, default 10
        Number of Newton iterations to run.
    matlab_compat : bool, default False
        **No-op.** Kept for API consistency with the rest of the port. This
        function is pure scalar float arithmetic, so none of the four traps
        (boundary / truncation / indexing / column-major) applies.

    Returns
    -------
    x_history : np.ndarray of shape (n_iter + 1,)
        `x_history[k]` is the k-th iterate; `x_history[0] = x0`.
    res_history : np.ndarray of shape (n_iter + 1,)
        `res_history[k] = x_history[k]**2 - 2` (the Newton residual).
    """
    x_history = np.zeros(n_iter + 1)
    res_history = np.zeros(n_iter + 1)

    x_history[0] = x0
    res_history[0] = x0 ** 2 - 2

    x = x0
    for k in range(n_iter):
        x = x - (x ** 2 - 2) / (2 * x)
        x_history[k + 1] = x
        res_history[k + 1] = x ** 2 - 2

    return x_history, res_history


if __name__ == "__main__":
    x_hist, res_hist = sqrt2_newton()

    print(f"x_0  = {x_hist[0]}")
    print(f"x_10 = {x_hist[-1]:.20f}")
    print(f"true sqrt(2) = {np.sqrt(2):.20f}")
    print(f"res_10 = {res_hist[-1]:.3e}")

    # Sanity checks
    assert x_hist.shape == (11,), f"x_history shape {x_hist.shape} != (11,)"
    assert res_hist.shape == (11,), f"res_history shape {res_hist.shape} != (11,)"
    assert abs(x_hist[-1] - np.sqrt(2)) < 1e-10, "x_10 not close to sqrt(2)"
    assert abs(res_hist[-1]) < 1e-10, "res_10 not close to 0"
    # Monotone convergence for this problem / x0:
    assert all(abs(r) >= abs(res_hist[i + 1]) for i, r in enumerate(res_hist[:-1])), \
        "residual should be monotonically non-increasing"
    print("sanity passed")
