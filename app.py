# app.py
import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, MODEL_METADATA, MODEL_ORDER
from ui_predict import render_predict_page, init_predict_state
from ui_educational import render_educational_lab_page, init_educational_state
from ui_about import render_about_page
from api_client import fetch_models
from streamlit_option_menu import option_menu

st.markdown("""
<style>
    /* slightly tighter layout */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


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
# --- Navigation bar ---
    selected = option_menu(
        menu_title=None,  # no extra title
        options=["Predict", "Educational Lab", "About & Credits"],
        icons=["camera", "bezier", "info-circle"],  # Bootstrap icon names
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {
                "padding": "0.25rem 0",
                "background-color": "rgba(0,0,0,0)",
                "justify-content": "center",
            },
            "nav": {
                "justify-content": "center",
            },
            "nav-item": {
                "margin": "0 0.4rem",
            },
            "nav-link": {
                "font-size": "17px",
                "font-weight": "500",
                "border-radius": "999px",
                "padding": "0.35rem 1.3rem",
                "color": "#374151",
                "background-color": "#F3F4F6",
                "border": "1px solid #E5E7EB",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.04)",
                "transition": "all 200ms ease-in-out",
            },
            "nav-link-hover": {
                "background-color": "#E8F5E9",
                "color": "#1B5E20",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #4CAF50, #26A69A)",
                "color": "white",
                "box-shadow": "0 4px 10px rgba(0,0,0,0.18)",
                "border": "none",
            },
        },
    )
    return selected


def init_models() -> None:
    """
    Fetch /models once and populate MODEL_METADATA / MODEL_ORDER.
    """
    if "models_loaded" in st.session_state and st.session_state["models_loaded"]:
        return

    try:
        models_dict = fetch_models()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load models from API: {exc}")
        return

    MODEL_METADATA.clear()
    MODEL_METADATA.update(models_dict)

    MODEL_ORDER.clear()
    MODEL_ORDER.extend(models_dict.keys())

    st.session_state["models_loaded"] = True


def init_session_state() -> None:
    init_predict_state()
    init_educational_state()
    # About has no state


def main() -> None:
    setup_page()
    init_models()
    init_session_state()

    render_header()
    page = render_top_nav()

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
