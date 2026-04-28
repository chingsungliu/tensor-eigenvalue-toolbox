"""About page for the Algorithm Toolbox demo.

Renders researcher bio, paper citations, and source-code link. Reached
from the sidebar `About` button in `demo_v0.py`; the back button at the
top of the page clears the `show_about` flag in `st.session_state` and
returns the user to the algorithm view.
"""
from __future__ import annotations

import streamlit as st


# Paper citations. Each entry: id, title, authors, year, venue, doi, note,
# optional bibtex. DOI links resolve via doi.org. To add a new paper, append
# a dict with these keys; set `bibtex` to a multi-line string to render a
# syntax-highlighted code block beneath the entry.
PAPERS = [
    {
        "id": 1,
        "title": "Newton-Noda iteration for finding the Perron pair of a weakly irreducible nonnegative tensor",
        "authors": "Liu, C.-S., Guo, C.-H., Lin, W.-W.",
        "year": 2017,
        "venue": "Numerische Mathematik, 137(1), 63-90",
        "doi": "10.1007/s00211-017-0869-7",
        "note": "Reference paper for NNI / NNI_ha in this toolbox",
        "bibtex": None,
    },
    {
        "id": 2,
        "title": "Exact and inexact iterative methods for finding the largest eigenpair of a weakly irreducible nonnegative tensor",
        "authors": "Liu, C.-S.",
        "year": 2022,
        "venue": "Journal of Scientific Computing, 91(3), 78",
        "doi": "10.1007/s10915-022-01852-5",
        "note": "Reference paper for HONI (exact / inexact branches)",
        "bibtex": None,
    },
    {
        "id": 3,
        "title": "Newton-Noda iteration for computing the ground states of nonlinear Schrödinger equations",
        "authors": "Du, C.-E., Liu, C.-S.",
        "year": 2022,
        "venue": "SIAM Journal on Scientific Computing, 44(4), A2370-A2385",
        "doi": "10.1137/21M1435793",
        "note": "NNI extension to nonlinear Schrödinger equations (potential next problem class)",
        "bibtex": None,
    },
    {
        "id": 4,
        "title": "A positivity preserving inexact Noda iteration for computing the smallest eigenpair of a large irreducible M-matrix",
        "authors": "Jia, Z., Lin, W.-W., Liu, C.-S.",
        "year": 2015,
        "venue": "Numerische Mathematik, 130(4), 645-679",
        "doi": "10.1007/s00211-014-0677-2",
        "note": "Theoretical foundation: M-matrix Noda iteration (NNI's predecessor)",
        "bibtex": None,
    },
]

GITHUB_URL = "https://github.com/chingsungliu/tensor-eigenvalue-toolbox"
PERSONAL_SITE = "https://sites.google.com/go.nuk.edu.tw/csliu"


def render_about() -> None:
    """Render the About page — author bio, research, citations, source."""
    col_back, _ = st.columns([2, 8])
    with col_back:
        if st.button("← Back to algorithms", key="about_back"):
            st.session_state["show_about"] = False

    st.header("About — Algorithm Toolbox")
    st.caption(
        "An interactive companion to the author's research on tensor "
        "eigenvalue computation. Each algorithm is ported from MATLAB to "
        "Python and validated against the original reference via per-iteration "
        "parity testing; this site lets you run them directly in the browser."
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Author")
        st.markdown(
            "**Liu, Ching-Sung 劉青松**  \n"
            "Professor  \n"
            "Department of Applied Mathematics  \n"
            "National University of Kaohsiung  \n"
            "Taiwan"
        )

        st.subheader("Contact")
        st.markdown(
            "[chingsungliu@nuk.edu.tw](mailto:chingsungliu@nuk.edu.tw)  \n"
            "[chingsungliu@gmail.com](mailto:chingsungliu@gmail.com)  \n"
            "+886-7-5919169  \n"
            "R440, Science Building  \n"
            "700 Kaohsiung University Road  \n"
            "Nantzu District, Kaohsiung, Taiwan"
        )

        st.markdown(f"[🌐 Personal website]({PERSONAL_SITE})")

    with col_r:
        st.subheader("Research interests")
        st.markdown(
            "- Tensor / matrix analysis and computations\n"
            "- Optimization: theory and algorithms\n"
            "- Numerical analysis and scientific computing"
        )

        st.subheader("Education")
        st.markdown(
            "- **Ph.D.** Mathematics, National Tsing Hua University (2007 – 2012)\n"
            "- **M.S.** Mathematics, National Tsing Hua University (2005 – 2007)\n"
            "- **B.S.** Mathematics, Tunghai University (2001 – 2005)"
        )

        st.subheader("Academic positions")
        st.markdown(
            "- **Professor**, Applied Mathematics, NUK (2023/08 – present)\n"
            "- **Associate Professor**, Applied Mathematics, NUK (2019/08 – 2023/07)\n"
            "- **Assistant Professor**, Applied Mathematics, NUK (2016/08 – 2019/07)\n"
            "- **Postdoctoral Researcher**, Applied Mathematics, NCTU (2014/08 – 2016/07)\n"
            "- **Postdoctoral Researcher**, Mathematics, NTHU (2012/08 – 2014/07)"
        )

    st.divider()

    st.subheader("Citations")
    st.caption(
        "If you use this toolbox or build on the algorithms, please cite "
        "the relevant paper(s). BibTeX entries to be added."
    )
    for p in PAPERS:
        st.markdown(
            f"**[{p['id']}]** {p['title']}  \n"
            f"{p['authors']} ({p['year']}). _{p['venue']}_.  \n"
            f"DOI: [{p['doi']}](https://doi.org/{p['doi']})"
        )
        st.caption(f"→ {p['note']}")
        if p["bibtex"]:
            st.code(p["bibtex"], language="bibtex")

    st.divider()

    st.subheader("Source code")
    st.markdown(
        f"This toolbox is open-source and available on GitHub: "
        f"[{GITHUB_URL}]({GITHUB_URL})"
    )
    st.caption(
        "Issues, pull requests, and questions welcome. The repository "
        "contains the Python ports, MATLAB reference implementations, "
        "parity tests, and per-algorithm hazard analyses."
    )
