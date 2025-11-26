# app.py
import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, MODEL_METADATA, MODEL_ORDER
from ui_predict import render_predict_page, init_predict_state
from ui_educational import render_educational_lab_page, init_educational_state
from ui_about import render_about_page
from api_client import fetch_models


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
    page = st.radio(
        "Navigation",
        options=["Predict", "Educational Lab", "About & Credits"],
        index=0,
        horizontal=True,
    )
    st.markdown("---")
    return page


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
