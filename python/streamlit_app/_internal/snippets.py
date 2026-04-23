"""MATLAB source snippet loader — shared between utilities and Layer 3 renderers."""
from pathlib import Path

MATLAB_SNIPPETS_DIR = Path(__file__).resolve().parent.parent / "matlab_snippets"


def read_snippet(name: str) -> str:
    """Read a MATLAB source snippet from matlab_snippets/<name>.m."""
    return (MATLAB_SNIPPETS_DIR / f"{name}.m").read_text(encoding="utf-8")
