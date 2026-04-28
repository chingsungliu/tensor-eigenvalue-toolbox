"""Unit tests for sp_Jaco_Ax (Sub-step 3.3 — A4 vectorized rewrite).

Coverage matrix:
- m=1 (p=0)        : zero matrix corner case
- m=2 (p=1)        : Jacobian == AA (no Kronecker chain)
- m=3 (p=2)        : the production case (Q7 / Multi / HONI / NNI all use m=3)
- m=4 (p=3)        : exercises the general-m axis loop
- Dense AA input   : ensures the dense → CSR conversion path is correct
- Empty rows in AA : np.repeat handles zero-length spans without crashing
- n=1              : trivial corner case (single-element vector / scalar AA)

Strategy:
- Old vs new equivalence (primary): the previous kron-based implementation
  is preserved as ``sp_Jaco_Ax_legacy`` inside this test module so each
  case can be cross-checked against it. atol=1e-12 / rtol=1e-10 guards
  against summation-order machine-epsilon drift.
- Numerical Jacobian spot-check (secondary): one m=3 case is also
  verified against finite-difference of ``tpv``, catching any bug
  inherited from the legacy implementation.

Run:
    .venv/bin/python test_sp_Jaco_Ax_unit.py
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, eye as sp_eye, issparse, kron as sp_kron

from tensor_utils import sp_Jaco_Ax, tenpow, tpv


# ---------------------------------------------------------------------------
# Reference implementation: the kron-based sp_Jaco_Ax that lived in
# tensor_utils.py before Sub-step 3.3. Inlined here so the test module is
# self-contained — it does not pin tensor_utils' previous behaviour.
# ---------------------------------------------------------------------------


def sp_Jaco_Ax_legacy(AA, x, m):
    """The kron-based reference implementation (pre-Sub-step-3.3)."""
    if not issparse(AA):
        AA = np.asarray(AA)
    x = np.asarray(x)
    n = len(x)
    p = m - 1
    if p == 0:
        return csr_matrix((n, n), dtype=x.dtype)
    AA_sp = AA.tocsr() if issparse(AA) else csr_matrix(AA)
    I = sp_eye(n, format="csr")
    J = csr_matrix((n, n), dtype=x.dtype)
    for i in range(1, p + 1):
        left = tenpow(x, i - 1).reshape(-1, 1)
        right = tenpow(x, p - i).reshape(-1, 1)
        inner = sp_kron(I, right, format="csr")
        outer = sp_kron(left, inner, format="csr")
        term = AA_sp @ outer
        J = J + term
    return J


def _assert_close(J_new, J_legacy, label, atol=1e-12, rtol=1e-10):
    """Compare two sparse Jacobians for numerical equality."""
    A_new = np.asarray(J_new.todense()) if issparse(J_new) else np.asarray(J_new)
    A_legacy = np.asarray(J_legacy.todense()) if issparse(J_legacy) else np.asarray(J_legacy)
    assert A_new.shape == A_legacy.shape, (
        f"{label}: shape mismatch {A_new.shape} vs {A_legacy.shape}"
    )
    diff = np.abs(A_new - A_legacy)
    max_abs = float(diff.max()) if diff.size else 0.0
    norm_legacy = float(np.linalg.norm(A_legacy))
    rel = max_abs / norm_legacy if norm_legacy > 0 else max_abs
    assert max_abs < atol or rel < rtol, (
        f"{label}: max abs diff {max_abs:.3e}, rel {rel:.3e} "
        f"(tol abs {atol:.0e} / rel {rtol:.0e})"
    )


# ---------------------------------------------------------------------------
# Helpers to build representative AA inputs
# ---------------------------------------------------------------------------


def _random_sparse_AA(n, m, density=0.05, seed=42):
    """Build a (n, n^(m-1)) random sparse CSR matrix."""
    rng = np.random.default_rng(seed)
    n_cols = n ** (m - 1)
    nnz = max(1, int(round(density * n * n_cols)))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_cols, size=nnz)
    vals = rng.random(nnz)
    return csr_matrix((vals, (rows, cols)), shape=(n, n_cols))


def _random_x(n, seed=42):
    rng = np.random.default_rng(seed)
    return np.abs(rng.random(n)) + 0.1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_m1_zero_matrix():
    n = 7
    AA = csr_matrix((n, 1))  # AA shape (n, n^0) = (n, 1)
    x = _random_x(n)
    J = sp_Jaco_Ax(AA, x, m=1)
    assert J.shape == (n, n)
    assert J.nnz == 0, f"m=1 must give zero Jacobian, got nnz={J.nnz}"
    print("  test_m1_zero_matrix              PASS")


def test_m2_identity_passthrough():
    """When m=2 (p=1), the Jacobian is AA itself."""
    n = 8
    AA = _random_sparse_AA(n, m=2, density=0.3)
    x = _random_x(n)
    J_new = sp_Jaco_Ax(AA, x, m=2)
    J_legacy = sp_Jaco_Ax_legacy(AA, x, m=2)
    _assert_close(J_new, J_legacy, label="m=2")
    # Also assert J equals AA (within machine epsilon — m=2 is the special
    # case where the Jacobian of F(x) = AA · x is AA).
    _assert_close(J_new, AA, label="m=2 (J == AA)")
    print("  test_m2_identity_passthrough     PASS")


def test_m3_random_sparse():
    """Production case: m=3, sparse AA, expected J[i, l] formula."""
    for seed in (42, 7, 137):
        n = 12
        AA = _random_sparse_AA(n, m=3, density=0.05, seed=seed)
        x = _random_x(n, seed=seed + 1)
        J_new = sp_Jaco_Ax(AA, x, m=3)
        J_legacy = sp_Jaco_Ax_legacy(AA, x, m=3)
        _assert_close(J_new, J_legacy, label=f"m=3 seed={seed}")
    print("  test_m3_random_sparse            PASS  (3 seeds)")


def test_m4_general_loop():
    """Exercises the general-m axis loop (p=3, three contributions per nnz)."""
    n = 6  # n^3 = 216 columns, manageable
    AA = _random_sparse_AA(n, m=4, density=0.02, seed=11)
    x = _random_x(n, seed=11)
    J_new = sp_Jaco_Ax(AA, x, m=4)
    J_legacy = sp_Jaco_Ax_legacy(AA, x, m=4)
    _assert_close(J_new, J_legacy, label="m=4")
    print("  test_m4_general_loop             PASS")


def test_dense_AA_input():
    """Dense numpy AA must convert to CSR and produce the same Jacobian."""
    n = 5
    AA_sparse = _random_sparse_AA(n, m=3, density=0.4, seed=99)
    AA_dense = AA_sparse.toarray()
    x = _random_x(n, seed=99)
    J_dense = sp_Jaco_Ax(AA_dense, x, m=3)
    J_sparse = sp_Jaco_Ax(AA_sparse, x, m=3)
    _assert_close(J_dense, J_sparse, label="dense AA")
    print("  test_dense_AA_input              PASS")


def test_empty_rows():
    """AA with an entirely-empty row should not crash and should give the
    same Jacobian as a dense reference."""
    n = 6
    rows = np.array([0, 0, 2, 2, 5])  # rows 1, 3, 4 are empty
    cols = np.array([3, 7, 1, 11, 5])
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    AA = csr_matrix((vals, (rows, cols)), shape=(n, n ** 2))
    x = _random_x(n, seed=21)
    J_new = sp_Jaco_Ax(AA, x, m=3)
    J_legacy = sp_Jaco_Ax_legacy(AA, x, m=3)
    _assert_close(J_new, J_legacy, label="empty rows")
    print("  test_empty_rows                  PASS")


def test_n1_trivial():
    """n=1: AA is (1, 1), x is scalar-shaped."""
    n = 1
    AA = csr_matrix(np.array([[2.5]]))  # (1, 1)
    x = np.array([3.0])
    J_new = sp_Jaco_Ax(AA, x, m=2)  # m=2 is the simplest non-trivial case
    J_legacy = sp_Jaco_Ax_legacy(AA, x, m=2)
    _assert_close(J_new, J_legacy, label="n=1 m=2")
    # Also m=3 — Jacobian = AA · 2x for n=1 is 2 * AA * x
    AA3 = csr_matrix(np.array([[1.5]]))  # (1, 1) (n^2 = 1 column)
    J_new = sp_Jaco_Ax(AA3, x, m=3)
    J_legacy = sp_Jaco_Ax_legacy(AA3, x, m=3)
    _assert_close(J_new, J_legacy, label="n=1 m=3")
    print("  test_n1_trivial                  PASS")


def test_finite_difference_spot_check():
    """Numerical Jacobian via central differences agrees with sp_Jaco_Ax to
    O(h^2) for a single random m=3 case. Validates the *mathematical* result
    of sp_Jaco_Ax independently of the legacy reference.
    """
    n = 8
    AA = _random_sparse_AA(n, m=3, density=0.1, seed=314)
    x = _random_x(n, seed=314)
    J = sp_Jaco_Ax(AA, x, m=3)
    J_dense = np.asarray(J.todense())

    h = 1e-5
    fd = np.zeros((n, n))
    for l in range(n):
        e_l = np.zeros(n)
        e_l[l] = 1.0
        f_plus = tpv(AA, x + h * e_l, m=3)
        f_minus = tpv(AA, x - h * e_l, m=3)
        fd[:, l] = (f_plus - f_minus) / (2 * h)

    diff = float(np.max(np.abs(J_dense - fd)))
    norm = float(np.linalg.norm(J_dense))
    rel = diff / norm if norm > 0 else diff
    assert rel < 1e-6, (
        f"finite difference disagrees with sp_Jaco_Ax: max diff {diff:.3e}, "
        f"rel {rel:.3e}"
    )
    print(f"  test_finite_difference_spot_check PASS  (rel diff {rel:.2e})")


def main():
    print("test_sp_Jaco_Ax_unit — Sub-step 3.3 unit coverage")
    print("=" * 60)
    test_m1_zero_matrix()
    test_m2_identity_passthrough()
    test_m3_random_sparse()
    test_m4_general_loop()
    test_dense_AA_input()
    test_empty_rows()
    test_n1_trivial()
    test_finite_difference_spot_check()
    print("=" * 60)
    print("All sp_Jaco_Ax unit tests passed.")


if __name__ == "__main__":
    main()
