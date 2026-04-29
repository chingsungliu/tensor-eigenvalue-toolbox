"""Paper-example tensor builders for Phase B reproduction.

Each builder returns the dense tensor (or its mode-1 unfolding) plus the
paper's specified initial vector and tensor order. Designed to be
fed straight into ``tensor_utils.nni``:

    >>> A, x0, m = build_liu2017_example1()
    >>> lam_U, x, nit, lam_L, *_ = nni(
    ...     A, m, tol=1e-13,
    ...     initial_vector=x0, maxit=100,
    ...     linear_solver="spsolve", halving=False,
    ... )

The builders are kept here (next to ``defaults.py``) so the demo can
expose them as alternative test cases when Phase B's UI integration
lands. The Phase B parity tests live in ``python/test_paper_*.py``
and exercise these builders directly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_liu2017_example1() -> Tuple[np.ndarray, np.ndarray, int]:
    """Liu / Guo / Lin (Numer. Math. 2017) §7 Example 1 — wind-power Markov chain.

    The paper specifies a "tensor of order 4 and dimension 4" with

        A_{ijk} = μ_1 · Q_{ij} + μ_2 · Q_{ik} + μ_3 · Q_i,
        (μ_1, μ_2, μ_3) = (0.629, 0.206, 0.165),

    and Q is the column-stochastic 4×4 matrix shown below. The formula
    as printed has only three subscripts on A while the tensor order
    is four — Phase B B1 (Day 13) treated this as a typeset truncation
    of the natural symmetric extension

        A_{ijkl} = μ_1 · Q_{ij} + μ_2 · Q_{ik} + μ_3 · Q_{il}

    (each follower index contributing one Q-channel weighted by the
    matching μ). With this reading, ∑_i A_{ijkl} = μ_1 + μ_2 + μ_3 = 1
    for every (j, k, l), so the leading-index marginal is
    column-stochastic — the natural property of a transition
    probability tensor.

    Disambiguation evidence (Phase B B1):

    - Paper §7 prints "tensor of order 4 and dimension 4" verbatim,
      so ``m = 4`` is explicit and ``m = 3`` candidates were ruled out.
    - Six candidates were tested (m ∈ {3, 4} × Q_i ∈ {diag, rowsum,
      stationary} plus an m=4 third-follower variant). Two m=4
      candidates produced the paper Figure 1 NNI nit=5 target, and
      both were column-stochastic in the leading index:
      a "third-follower" reading (this builder) and an "l-dummy"
      reading where A is independent of l. The third-follower
      reading is preferred because the l-dummy reading makes A
      rank-deficient along the fourth axis, which is unusual for a
      transition probability tensor.

    Returns
    -------
    A : np.ndarray, shape (4, 4, 4, 4)
        Dense order-4 tensor. Pass directly to ``nni()``; its
        polymorphic dispatch handles the mode-1 unfolding internally.
    x0 : np.ndarray, shape (4,)
        Paper §7 default initial vector ``(1/√n) · 1``.
    m : int
        Tensor order. Always 4 for this example.
    """
    Q = np.array([
        [0.837, 0.058, 0.000, 0.000],
        [0.163, 0.854, 0.113, 0.000],
        [0.000, 0.088, 0.847, 0.116],
        [0.000, 0.000, 0.040, 0.884],
    ])
    mu_1, mu_2, mu_3 = 0.629, 0.206, 0.165
    n = 4

    # A[i, j, k, l] = μ_1·Q[i,j] + μ_2·Q[i,k] + μ_3·Q[i,l]
    # Each Q[:, follower] term contributes along one of the three
    # follower axes (j, k, l); the (1+1+1)/(0+0+0) μ-weighted mixture
    # makes the leading-index marginal sum to 1.
    A = (
        mu_1 * Q[:, :, None, None]
        + mu_2 * Q[:, None, :, None]
        + mu_3 * Q[:, None, None, :]
    )

    x0 = np.ones(n) / np.sqrt(n)
    m = 4
    return A, x0, m
