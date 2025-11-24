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
        initial_sidebar_state="collapsed",
    )


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_top_nav() -> str:
    """
    Top navigation bar, horizontally, above the page content.
    """
    page = st.radio(
        "Navigation",
        options=["Predict", "Educational Lab", "About & Credits"],
        index=0,
        horizontal=True,
    )
    st.markdown("---")
    return page


def init_session_state() -> None:
    """Initialize all page-specific state containers."""
    init_predict_state()
    init_educational_state()
    # About page currently doesn't need its own state


def main() -> None:
    setup_page()
    init_session_state()

    # Header + top navigation
    render_header()
    page = render_top_nav()

    # Page content
    if page == "Predict":
        render_predict_page()
    elif page == "Educational Lab":
        render_educational_lab_page()
    elif page == "About & Credits":
        render_about_page()
    else:
        st.error("Unknown page selected.")


if __name__ == "__main__":
    main()
