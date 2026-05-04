"""Hypergraph adjacency / signless-Laplacian tensor builders for paper examples.

Used by Phase B paper-reproduction tests (Liu/Guo/Lin 2017 §7 Example 2,
signless Laplacian of an m-uniform hypergraph; possibly other examples
later).

Convention (Cooper-Dutle 2012, Hu-Qi 2015):

- m-uniform hypergraph H on vertex set V = {0, ..., n-1} (0-indexed
  internally; the paper's E is 1-indexed in print).
- Adjacency tensor C of order m, dimension n: for each edge e ∈ E with
  |e| = m, every permutation σ of e gives `C[σ] = 1/(m-1)!`. All other
  entries are zero. This makes C symmetric in all m indices and gives
  `(C · 1^{m-1})_v = deg(v)` (the (m-1)! permutations of e\{v} sum to
  exactly deg(v)).
- Diagonal degree tensor D: `D[v, v, ..., v] = deg(v)`.
- Signless Laplacian: `A = D + C`. So `(A · 1^{m-1})_v = 2·deg(v)`.

The construction is **sparse-only**: each edge contributes `m!` nonzero
entries, totalling `|E|·m! + n` over the full tensor. For m=5, n=100,
the dense tensor would be 10¹⁰ entries; the sparse representation is
~10⁵, well within memory.
"""
from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_paper_edge_set(m: int, n: int) -> List[Tuple[int, ...]]:
    """Liu 2017 §7 Example 2 edge set.

    E = {(i, j, j+1, ..., j+m-2)} for i = 1, 2, 3 and j = i+1, ..., n-m+2
    (1-indexed in the paper). Returns 0-indexed tuples ready for
    tensor construction.
    """
    edges: List[Tuple[int, ...]] = []
    for i in range(1, 4):  # paper i = 1, 2, 3
        # j ranges from i+1 to n-m+2 inclusive (paper §7).
        for j in range(i + 1, n - m + 3):
            edge_paper = (i,) + tuple(range(j, j + m - 1))  # m vertices, 1-indexed
            edges.append(tuple(v - 1 for v in edge_paper))   # 0-indexed
    return edges


def build_signless_laplacian(
    m: int, n: int, *, diagonal_coeff: float = 1.0
) -> csr_matrix:
    """Build the m-th order n-dim signless Laplacian tensor
    `A = diagonal_coeff · D + C`, return its mode-1 unfolding as a CSR
    matrix of shape `(n, n^(m-1))`.

    Column-major unfolding (consistent with `tenpow` / `tpv` /
    `sp_Jaco_Ax` in the toolbox): column index for tensor entry
    `T[i, j_0, j_1, ..., j_{m-2}]` is

        c = j_0 + j_1·n + j_2·n² + ... + j_{m-2}·n^(m-2)

    Uses `csr_matrix((data, (rows, cols)), ...)` constructor, which sums
    duplicate (row, col) pairs — natural for the adjacency contributions
    that may overlap across permutations of different edges.

    Parameters
    ----------
    m, n : int
        Tensor order and dimension. The paper edge set requires `n >= m`.
    diagonal_coeff : float, default 1.0
        Multiplier on the diagonal degree tensor. Default `1.0` reproduces
        the standard signless Laplacian `A = D + C` used by paper §7
        Example 2 / Table 1. Paper §7 Example 3 uses `diagonal_coeff=100`
        to amplify D so the spectral structure is dominated by the
        diagonal — that is the regime where the halving procedure fires
        actively (paper Figure 2).
    """
    if m < 3:
        # Out of scope for tensor eigenvalue (Sub-step 3.5 enforces m≥3 at
        # the algorithm entry; this builder reflects the same scope).
        raise ValueError(f"m must be >= 3 for signless-Laplacian tensor, got m={m}")
    if n < m:
        raise ValueError(
            f"n must be >= m so the paper edge set is non-empty (got m={m}, n={n})"
        )

    edges = build_paper_edge_set(m, n)
    if not edges:
        raise ValueError(
            f"empty edge set for m={m}, n={n} — paper §7 needs n >= m+1"
        )

    # Vertex degrees.
    degrees = np.zeros(n, dtype=np.int64)
    for edge in edges:
        for v in edge:
            degrees[v] += 1

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    # Diagonal D: D[v, v, ..., v] = deg(v) (scaled by diagonal_coeff).
    # Column index for (v, v, ..., v) under column-major unfolding:
    # c = v · (1 + n + n² + ... + n^(m-2)).
    col_factor = sum(n ** k for k in range(m - 1))
    for v in range(n):
        rows.append(v)
        cols.append(v * col_factor)
        vals.append(diagonal_coeff * float(degrees[v]))

    # Adjacency C: each edge e contributes `1/(m-1)!` to every permutation
    # of e (m! permutations per edge).
    inv_fact = 1.0 / factorial(m - 1)
    for edge in edges:
        for perm in permutations(edge):
            i = perm[0]
            # Column index from the remaining m-1 entries (column-major).
            c = 0
            for k in range(m - 1):
                c += perm[1 + k] * (n ** k)
            rows.append(i)
            cols.append(c)
            vals.append(inv_fact)

    AA = csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n ** (m - 1)),
    )
    return AA


def build_paper_ex5_edge_set(m: int, n: int) -> List[Tuple[int, ...]]:
    """Liu 2017 §7 Example 5 cyclic edge set.

    E = {{i - m + 2, i - m + 3, ..., i, i + 1} : i = m - 1, ..., n}
    where the index ``n + 1`` is identified with ``1`` (cyclic wrap).
    Yields ``n - m + 2`` edges of size ``m``, each a window of ``m``
    consecutive vertices on the cycle ``Z / nZ``. Returns 0-indexed
    tuples ready for tensor construction.
    """
    if n < m:
        raise ValueError(f"n must be >= m for Example 5 cyclic edges (got m={m}, n={n})")
    edges: List[Tuple[int, ...]] = []
    for i in range(m - 1, n + 1):
        edge_paper = list(range(i - m + 2, i + 2))      # 1-indexed window of m
        edge_paper = [1 if v == n + 1 else v for v in edge_paper]
        edges.append(tuple(v - 1 for v in edge_paper))   # 0-indexed
    return edges


def build_z_tensor(
    m: int, n: int, *, seed: int = 42
) -> Tuple[csr_matrix, float]:
    """Liu 2017 §7 Example 5 Z-tensor — return mode-1 unfolding of the
    shifted nonneg tensor ``A = s·I − Z`` plus the shift ``s``.

    Z = D − C + V where:

    - D, C are degree diagonal and Cooper-Dutle adjacency (paper Ex 5
      definitions, identical to Example 2 modulo the edge set).
    - C uses the cyclic edge set ``build_paper_ex5_edge_set(m, n)``.
    - V is a diagonal "trap potential" tensor, ``V[v,v,...,v] = 0.1·u_v``
      with ``u_v ~ rng.uniform(0, 1)`` (paper §7: ``v = 0.1·rand(n,1)``).

    The returned matrix is ``A = s·I − Z = (s−D−V) + C`` where
    ``s = max_v Z[v,v,...,v] = max_v (deg(v) + 0.1·u_v)``. ``A`` is
    nonneg by construction (the diagonal is ``s − Z[v,...,v] ≥ 0``,
    achieving 0 at the argmax vertex; the off-diagonal contributions
    come from ``+C`` and are positive). The smallest eigenpair of ``Z``
    is recovered as ``μ* = s − ρ(A)``, ``x* = x_A``.

    Cyclic edges have ``m`` distinct vertices (no repeats since the
    window is ``m`` consecutive integers on ``Z/nZ`` with ``n ≥ m``),
    so ``C`` has no diagonal entries — the only diagonal contributions
    to ``A`` are from ``s·I − D − V``.

    Note on RNG: ``np.random.default_rng`` is not bit-compatible with
    MATLAB ``rand()``, so the random ``v`` differs between toolbox and
    MATLAB. Iter counts may differ by 1 from MATLAB Octave reference;
    the Phase B B5 test absorbs this with ``ALLOWED_NIT_DIFF = 2``.

    Parameters
    ----------
    m, n : int
        Tensor order and dimension. Paper Table 2 tests
        ``m ∈ {3, 4, 5}`` × ``n ∈ {20, 50, 100}``.
    seed : int, default 42
        Seed for the trap potential ``v`` (matches Day 15 Octave
        reference, modulo the MATLAB / Python RNG difference noted
        above).

    Returns
    -------
    AA : scipy.sparse.csr_matrix, shape (n, n**(m-1))
        Mode-1 unfolding of ``A = s·I − Z`` (column-major, consistent
        with ``build_signless_laplacian``).
    s : float
        The shift ``max_v Z[v,...,v]``. Caller computes the smallest
        eigenvalue of ``Z`` as ``s − ρ(A)``.
    """
    if m < 3:
        raise ValueError(f"m must be >= 3 for Z-tensor builder, got m={m}")
    if n < m:
        raise ValueError(
            f"n must be >= m so the cyclic edge set is non-degenerate "
            f"(got m={m}, n={n})"
        )

    edges = build_paper_ex5_edge_set(m, n)

    degrees = np.zeros(n, dtype=np.int64)
    for edge in edges:
        for v in edge:
            degrees[v] += 1

    rng = np.random.default_rng(seed)
    v_diag = 0.1 * rng.uniform(0.0, 1.0, size=n)

    z_diag = degrees.astype(np.float64) + v_diag
    s = float(np.max(z_diag))

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    # Diagonal of A: s − Z[v,...,v]. Column index for (v, v, ..., v):
    # c = v · (1 + n + n² + ... + n^(m-2)).
    col_factor = sum(n ** k for k in range(m - 1))
    for v in range(n):
        rows.append(v)
        cols.append(v * col_factor)
        vals.append(s - z_diag[v])

    # Off-diagonal of A from +C (note sign: A = s·I − D + C − V, so the
    # adjacency contribution carries a positive sign here, in contrast
    # to D + C in build_signless_laplacian). Each cyclic edge contributes
    # ``1/(m-1)!`` to every permutation; m! permutations per edge.
    inv_fact = 1.0 / factorial(m - 1)
    for edge in edges:
        for perm in permutations(edge):
            i = perm[0]
            c = 0
            for k in range(m - 1):
                c += perm[1 + k] * (n ** k)
            rows.append(i)
            cols.append(c)
            vals.append(inv_fact)

    AA = csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n ** (m - 1)),
    )
    return AA, s
