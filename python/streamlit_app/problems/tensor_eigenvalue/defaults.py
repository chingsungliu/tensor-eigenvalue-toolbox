"""Default test-case builder for tensor eigenvalue algorithms (shared by Multi / HONI / NNI)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from tensor_utils import sp_tendiag


def build_q7_tensor(n: int, m: int, rng_seed: int = 42):
    """Build a Q7-style sparse M-tensor + positive initial vector.

    Mirrors the parity-test setup shared by Multi Q5 / HONI Q4 / NNI Q7
    reference generators: diagonal ``d ~ U[1, 11]`` plus sparse perturbation
    (density 0.02, magnitude 0.01), ``x0 = |rand(n)| + 0.1``. Keeping the
    Streamlit defaults aligned with the parity cases means the UI exercises
    the same regime documented in
    ``memory/feedback_nni_rayleigh_quotient_noise_floor.md``.

    Returns
    -------
    AA : scipy.sparse.csr_matrix, shape (n, n**(m-1))
        Mode-1 unfolding of the m-order n-dim positive M-tensor.
    x0 : np.ndarray, shape (n,)
        Positive initial vector (``|rand(n)| + 0.1``, all entries > 0).
    """
    rng = np.random.default_rng(rng_seed)
    d = rng.uniform(1.0, 11.0, size=n)
    AA_diag = sp_tendiag(d, m)

    n_cols = n ** (m - 1)
    nnz = max(1, int(round(0.02 * n * n_cols)))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_cols, size=nnz)
    vals = 0.01 * rng.random(nnz)
    pert = csr_matrix((vals, (rows, cols)), shape=(n, n_cols))

    AA = (AA_diag + pert).tocsr()
    x0 = np.abs(rng.random(n)) + 0.1
    return AA, x0
