# app.py
import streamlit as st

from config import APP_TITLE, APP_SUBTITLE
from ui_predict import render_predict_page, init_predict_state
from ui_educational import render_educational_lab_page, init_educational_state
from ui_about import render_about_page


def setup_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown("---")


def render_sidebar() -> str:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Predict", "Educational Lab", "About & Credits"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**GrazeGauge**\n
Capstone project - computer vision for pasture biomass.
"""
    )

    return page


def init_session_state() -> None:
    """Initialize all page-specific state containers."""
    init_predict_state()
    init_educational_state()
    # About page currently doesn't need its own state


def main() -> None:
    setup_page()
    init_session_state()

    page = render_sidebar()

    if page == "Predict":
        render_header()
        render_predict_page()
    elif page == "Educational Lab":
        render_header()
        render_educational_lab_page()
    elif page == "About & Credits":
        render_header()
        render_about_page()
    else:
        render_header()
        st.error("Unknown page selected.")


if __name__ == "__main__":
    main()
