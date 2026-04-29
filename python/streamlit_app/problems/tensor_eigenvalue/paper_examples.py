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
from scipy.sparse import csr_matrix

from streamlit_app.problems.tensor_eigenvalue.hypergraph_utils import (
    build_signless_laplacian,
)


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


def build_liu2017_example2(m: int, n: int) -> Tuple[csr_matrix, np.ndarray]:
    """Liu / Guo / Lin (Numer. Math. 2017) §7 Example 2 — signless Laplacian
    of an m-uniform hypergraph.

    The paper's Table 1 reports NNI iteration counts for the signless
    Laplacian ``A = D + C`` on the m-uniform connected hypergraph with

        E = {(i, j, j+1, …, j+m-2) : i ∈ {1,2,3}, j ∈ {i+1, …, n-m+2}}

    (paper §7 Example 2; 1-indexed in print). ``D`` is the diagonal
    degree tensor and ``C`` is the Cooper-Dutle (2012) adjacency tensor
    with `C[σ(e)] = 1/(m-1)!` over all permutations of each edge.

    The tensor is returned as a sparse mode-1 unfolding of shape
    `(n, n^(m-1))` (dense storage would be `n^m`, infeasible for the
    m=5 / n=100 case in paper Table 1 — `10^10` entries).

    Parameters
    ----------
    m : int
        Tensor order. Paper Table 1 tests m ∈ {3, 4, 5}.
    n : int
        Dimension. Paper Table 1 tests n ∈ {20, 50, 100}.

    Returns
    -------
    AA : scipy.sparse.csr_matrix, shape (n, n**(m-1))
        Mode-1 unfolding of the signless Laplacian tensor.
    x0 : np.ndarray, shape (n,)
        Paper §7 default initial vector ``(1/√n) · 1``.
    """
    AA = build_signless_laplacian(m, n)
    x0 = np.ones(n) / np.sqrt(n)
    return AA, x0


def build_liu2017_example3() -> Tuple[csr_matrix, np.ndarray]:
    """Liu / Guo / Lin (Numer. Math. 2017) §7 Example 3 — halving demonstration.

    Same edge set as Example 2 at `m=4, n=20`, but with the diagonal
    degree tensor amplified by 100×: `A = 100·D + C`. The amplification
    pushes the spectral structure into a regime where the halving
    procedure (NNI-hav, `halving=True`) is most actively triggered.

    Paper Figure 2 reports two trajectories on this case:

    - ``halving=True`` (NNI-hav): convergence in **74 outer iterations**,
      with up to six halvings per iter to satisfy the acceptance
      condition (37). Trajectory is monotone in λ_U.
    - ``halving=False`` (canonical, θ_k = 1): convergence in **no more
      than 20 outer iterations**, but the λ_U trajectory is
      non-monotone.

    The example is the demo motivating Sub-step 4.1's UI variant
    selector — both halving and canonical are first-class options
    here, with paper-explicit trade-off labels.

    Returns
    -------
    AA : scipy.sparse.csr_matrix, shape (20, 20**3)
        Mode-1 unfolding of `100·D + C` for the Example 2 edge set
        with `m=4, n=20`.
    x0 : np.ndarray, shape (20,)
        Paper §7 default initial vector `(1/√n) · 1`.
    """
    AA = build_signless_laplacian(m=4, n=20, diagonal_coeff=100.0)
    x0 = np.ones(20) / np.sqrt(20)
    return AA, x0
