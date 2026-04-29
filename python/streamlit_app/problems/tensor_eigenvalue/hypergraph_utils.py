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


def build_signless_laplacian(m: int, n: int) -> csr_matrix:
    """Build the m-th order n-dim signless Laplacian tensor `A = D + C`,
    return its mode-1 unfolding as a CSR matrix of shape `(n, n^(m-1))`.

    Column-major unfolding (consistent with `tenpow` / `tpv` /
    `sp_Jaco_Ax` in the toolbox): column index for tensor entry
    `T[i, j_0, j_1, ..., j_{m-2}]` is

        c = j_0 + j_1·n + j_2·n² + ... + j_{m-2}·n^(m-2)

    Uses `csr_matrix((data, (rows, cols)), ...)` constructor, which sums
    duplicate (row, col) pairs — natural for the adjacency contributions
    that may overlap across permutations of different edges.
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

    # Diagonal D: D[v, v, ..., v] = deg(v).
    # Column index for (v, v, ..., v) under column-major unfolding:
    # c = v · (1 + n + n² + ... + n^(m-2)).
    col_factor = sum(n ** k for k in range(m - 1))
    for v in range(n):
        rows.append(v)
        cols.append(v * col_factor)
        vals.append(float(degrees[v]))

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
