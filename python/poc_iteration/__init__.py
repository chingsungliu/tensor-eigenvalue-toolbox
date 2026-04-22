"""Per-iteration parity POC + reusable three-piece API.

    from poc_iteration import find_divergence, report, print_neighborhood

See parity_utils.py for docstrings.
"""
from .parity_utils import (
    TOLERANCE,
    find_divergence,
    print_neighborhood,
    report,
)

__all__ = ["TOLERANCE", "find_divergence", "print_neighborhood", "report"]
