"""About page for the Algorithm Toolbox demo.

Renders researcher bio, paper citations, and source-code link. Reached
from the sidebar `About` button in `demo_v0.py`; the back button at the
top of the page clears the `show_about` flag in `st.session_state` and
returns the user to the algorithm view.
"""
from __future__ import annotations

import streamlit as st


# Paper citations. Each entry: title, venue, optional bibtex.
# To fill a citation later: replace the placeholder fields and (optionally)
# set `bibtex` to a multi-line string — it will render as a syntax-
# highlighted code block.
PAPERS = [
    {
        "title": "Paper 1 — title to be added",
        "venue": "Journal name, Year",
        "bibtex": None,
    },
    {
        "title": "Paper 2 — title to be added",
        "venue": "Journal name, Year",
        "bibtex": None,
    },
    {
        "title": "Paper 3 — title to be added",
        "venue": "Journal name, Year",
        "bibtex": None,
    },
    {
        "title": "Paper 4 — title to be added",
        "venue": "Journal name, Year",
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
        "An interactive demo accompanying the author's research on tensor "
        "eigenvalue computation. The algorithms are ported from MATLAB to "
        "Python with per-iteration parity testing; this site lets you run "
        "them directly in the browser."
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
    for i, p in enumerate(PAPERS, start=1):
        st.markdown(f"**[{i}]** {p['title']}  \n_{p['venue']}_")
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
