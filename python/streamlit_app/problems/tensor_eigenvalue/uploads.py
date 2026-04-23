"""Tensor file upload parser for Layer 3 renderers (.mat / .npz).

Public API
----------
``load_tensor_file(uploaded_file) -> dict``
    Top-level entry point consumed by Streamlit ``st.file_uploader`` output.
    Returns ``{"AA", "x0", "n", "m", "source_info"}``.

``load_tensor_bytes(raw, filename) -> dict``
    Testable variant — same behaviour but takes raw bytes + filename directly
    instead of a Streamlit ``UploadedFile`` (which cannot easily be mocked).

Key conventions
---------------
- Reserved tensor keys (tried in order): ``AA`` (2-D sparse or dense mode-1
  unfolding) → ``A_tensor`` (m-D dense full tensor). Any found match wins.
- Auto-detect fallback: any ndarray with ``ndim >= 2`` (or a sparse matrix)
  that is **not** the reserved x0 key — picks the first such entry.
- Reserved x0 key: ``x0`` (1-D, length n). Optional. Auto-detect fallback:
  any 1-D ndarray of length n that is not the tensor array itself.

Shape semantics
---------------
Uploaded arrays are normalized to a ``scipy.sparse.csr_matrix`` mode-1
unfolding of shape ``(n, n**(m-1))``:
- ``ndim == 2``: treated directly as mode-1 unfolding; ``m`` inferred by
  solving ``n**(m-1) == second_dim``.
- ``ndim >= 3``: treated as a full tensor with all modes of equal size;
  unfolded via :func:`tensor_utils.ten2mat` with ``k=0`` (column-major).

Validation
----------
Only shape and non-zero norm for x0; algorithm-specific positivity checks
(NNI / HONI require ``x > 0``) are left to runtime — ``source_info`` flags
non-positive x0 entries so the UI can surface a warning.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse, load_npz

from tensor_utils import ten2mat

RESERVED_TENSOR_KEYS = ("AA", "A_tensor")
RESERVED_X0_KEY = "x0"


def load_tensor_file(uploaded_file) -> dict:
    """Parse a Streamlit ``UploadedFile`` into a normalized tensor dict.

    Raises ``ValueError`` on any parse / validation failure with a message
    suitable for ``st.error``.
    """
    return load_tensor_bytes(uploaded_file.getvalue(), uploaded_file.name)


def load_tensor_bytes(raw: bytes, filename: str) -> dict:
    """Testable entry: parse raw file bytes given the filename for ext dispatch."""
    lower = filename.lower()
    if lower.endswith(".mat"):
        data = parse_mat(raw)
        fmt = "mat"
    elif lower.endswith(".npz"):
        data = parse_npz(raw)
        fmt = "npz"
    else:
        raise ValueError(
            f"Unsupported extension: {filename!r} (expected .mat or .npz)"
        )

    tensor_key, tensor_arr = find_tensor_key(data)
    AA, n, m = normalize_to_mode1(tensor_arr)
    x0 = find_x0_key(data, n)

    unfold_desc = (
        "mode-1 unfolding" if (issparse(tensor_arr) or np.ndim(tensor_arr) == 2)
        else f"{np.ndim(tensor_arr)}-D tensor, auto-unfolded via ten2mat(k=0)"
    )
    x0_desc = "provided" if x0 is not None else "not provided (will use |rand(n)|+0.1)"
    if x0 is not None and np.any(x0 <= 0):
        x0_desc += " ⚠️ contains non-positive entries (HONI / NNI expect x > 0)"
    source_info = (
        f"{fmt}: key {tensor_key!r} ({unfold_desc}), "
        f"n={n}, m={m}, x0 {x0_desc}"
    )
    return {"AA": AA, "x0": x0, "n": n, "m": m, "source_info": source_info}


def parse_mat(raw: bytes) -> dict:
    """Load .mat into a plain dict, dropping MATLAB metadata keys."""
    buf = io.BytesIO(raw)
    try:
        mat = loadmat(buf, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise ValueError(f"Failed to load .mat file: {type(e).__name__}: {e}")
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def parse_npz(raw: bytes) -> dict:
    """Load .npz. Supports both np.savez output and scipy.sparse.save_npz output."""
    buf = io.BytesIO(raw)
    try:
        npz = np.load(buf, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Failed to load .npz file: {type(e).__name__}: {e}")

    # scipy.sparse.save_npz writes a 'format' key identifying the sparse type.
    if "format" in npz.files:
        buf2 = io.BytesIO(raw)
        try:
            sp = load_npz(buf2)
        except Exception as e:
            raise ValueError(
                f"File looks like scipy.sparse.save_npz but failed to load: "
                f"{type(e).__name__}: {e}"
            )
        return {"AA": sp}

    return {k: np.asarray(npz[k]) for k in npz.files}


def find_tensor_key(data: dict) -> tuple[str, Any]:
    """Return (key, array) for the tensor variable.

    Tries reserved names first, then auto-detects the first ndarray with
    ndim >= 2 (or any sparse matrix) that is not the reserved x0 key.
    """
    for rk in RESERVED_TENSOR_KEYS:
        if rk in data and _looks_like_tensor(data[rk]):
            return rk, data[rk]

    for k, v in data.items():
        if k == RESERVED_X0_KEY:
            continue
        if _looks_like_tensor(v):
            return k, v

    raise ValueError(
        "No tensor-like variable found in file. "
        f"Looked for reserved keys {list(RESERVED_TENSOR_KEYS)!r}, then "
        f"auto-detected sparse / ndim>=2 arrays. "
        f"Keys present: {sorted(data.keys())!r}"
    )


def find_x0_key(data: dict, n: int) -> np.ndarray | None:
    """Return the x0 vector (1-D, length n) or None.

    Reserved ``x0`` key wins; otherwise auto-detects a 1-D ndarray of length n
    that isn't in ``RESERVED_TENSOR_KEYS``. Raises ``ValueError`` only if the
    reserved key exists but has the wrong shape.
    """
    if RESERVED_X0_KEY in data:
        v = np.asarray(data[RESERVED_X0_KEY]).flatten()
        if v.shape != (n,):
            raise ValueError(
                f"Reserved key 'x0' has wrong shape: got {v.shape}, expected ({n},) "
                f"(must match tensor's first dimension)."
            )
        if np.linalg.norm(v) == 0:
            raise ValueError("Reserved key 'x0' is a zero vector; cannot be used.")
        return v

    for k, v in data.items():
        if k in RESERVED_TENSOR_KEYS:
            continue
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == n:
            if np.linalg.norm(v) > 0:
                return v
    return None


def normalize_to_mode1(arr) -> tuple[csr_matrix, int, int]:
    """Convert a 2-D or m-D input into a mode-1 unfolding sparse matrix.

    Returns ``(AA_csr, n, m)``.
    """
    if issparse(arr):
        arr = arr.tocsr()
        n, ncols = arr.shape
        m = _infer_m_from_ncols(n, ncols)
        return arr, n, m

    arr = np.asarray(arr)
    if arr.ndim == 2:
        n, ncols = arr.shape
        m = _infer_m_from_ncols(n, ncols)
        return csr_matrix(arr), n, m
    if arr.ndim >= 3:
        shape = arr.shape
        if len(set(shape)) != 1:
            raise ValueError(
                f"Full tensor must have equal modes (n, n, ..., n); got shape {shape}."
            )
        n = shape[0]
        m = arr.ndim
        AA = ten2mat(arr, k=0)
        return csr_matrix(AA), n, m
    raise ValueError(
        f"Expected 2-D mode-1 unfolding or m-D tensor (ndim>=2); got ndim={arr.ndim}."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _looks_like_tensor(v) -> bool:
    if issparse(v):
        return True
    if isinstance(v, np.ndarray) and v.ndim >= 2:
        return True
    return False


def _infer_m_from_ncols(n: int, ncols: int) -> int:
    """Given ``n`` and ``ncols == n**(m-1)`` for some integer ``m >= 2``, return m.

    Raises ``ValueError`` if no such m exists (up to m=10).
    """
    if n < 2:
        raise ValueError(f"First dimension must be >= 2, got n={n}.")
    for m in range(2, 11):
        if n ** (m - 1) == ncols:
            return m
    raise ValueError(
        f"Cannot infer tensor order: second dim {ncols} is not n**(m-1) for n={n} "
        f"and any integer m in [2, 10]. Expected values: "
        f"{[n**(mm-1) for mm in range(2, 8)]} for m in [2..7]."
    )
