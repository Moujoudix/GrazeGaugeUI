from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import streamlit as st
import altair as alt

from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    MODEL_METADATA,
    MODEL_ORDER,
    FOCUS_OPTIONS,      # now we treat as list of biomass keys we care about
    FOCUS_TO_LABEL,
)
from api_client import call_compare_api



def init_educational_state() -> None:
    if "edu_state" not in st.session_state:
        st.session_state["edu_state"] = {
            "model_1_id": None,
            "model_2_id": None,
            "example_image": None,  # UploadedFile
            "comparison_result": None,  # raw JSON from /compare
            "focus": FOCUS_OPTIONS[0],  # default first key
            "api_error": None,
        }



def init_educational_state() -> None:
    if "edu_state" not in st.session_state:
        st.session_state["edu_state"] = {
            "model_1_id": None,
            "model_2_id": None,
            "model_1_aux": False,
            "model_2_aux": False,
            "example_index": None,
            "gt_biomass": None,
            "pred_model1": None,
            "pred_model2": None,
            "error_model1": None,
            "error_model2": None,
            "focus": "Green",
        }


def render_educational_lab_page() -> None:
    init_educational_state()
    state = st.session_state["edu_state"]

    _render_educational_controls(state)
    st.markdown("---")
    _render_comparison_result(state)



# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
def _render_educational_controls(state: Dict[str, Any]) -> None:
    st.subheader("Model comparison on a single image")

    col_models, col_image = st.columns([1, 1])

    with col_models:
        st.markdown("**1. Choose two models**")

        model_1_id = st.selectbox(
            "Model 1",
            options=[None] + MODEL_ORDER,
            format_func=lambda m: MODEL_METADATA[m]["display_name"]
            if m is not None
            else "Select...",
            key="edu_model_1",
        )

        model_2_id = st.selectbox(
            "Model 2",
            options=[None] + MODEL_ORDER,
            format_func=lambda m: MODEL_METADATA[m]["display_name"]
            if m is not None
            else "Select...",
            key="edu_model_2",
        )

        state["model_1_id"] = model_1_id
        state["model_2_id"] = model_2_id

    with col_image:
        st.markdown("**2. Upload an image to compare on**")
        uploaded = st.file_uploader(
            "Upload a single pasture image",
            type=["jpg", "jpeg", "png"],
            key="edu_image_uploader",
        )
        state["example_image"] = uploaded
        if uploaded:
            st.image(uploaded, use_column_width=True)

    st.markdown("**3. Run comparison**")
    if st.button("Compare on this image", type="primary", key="edu_compare_button"):
        _handle_compare(state)



def _handle_compare(state: Dict[str, Any]) -> None:
    model_1_id = state["model_1_id"]
    model_2_id = state["model_2_id"]
    img = state["example_image"]

    if model_1_id is None or model_2_id is None:
        st.warning("Please select both Model 1 and Model 2.")
        return
    if img is None:
        st.warning("Please upload an image to compare on.")
        return

    state["api_error"] = None

    with st.spinner("Running comparison and Grad-CAM..."):
        try:
            result = call_compare_api(
                image=img,
                model_1=model_1_id,
                model_2=model_2_id,
                method="grad_cam",
            )
        except Exception as exc:  # noqa: BLE001
            state["api_error"] = str(exc)
            try:
                st.toast(f"Comparison failed: {exc}", icon="❌")
            except Exception:
                st.error(f"Comparison failed: {exc}")
            return

    state["comparison_result"] = result
    try:
        st.toast("Comparison ready!", icon="📊")
    except Exception:
        st.success("Comparison ready!")



# -----------------------------------------------------------------------------
# Comparison result
# -----------------------------------------------------------------------------
def _render_comparison_result(state: Dict[str, Any]) -> None:
    if state.get("api_error"):
        st.error(f"Last comparison error: {state['api_error']}")

    result = state.get("comparison_result")
    if not result:
        st.info(
            "No comparison yet. Choose two models, upload an image, "
            "and click **Compare on this image**."
        )
        return

    models_data = result.get("models", [])
    if len(models_data) != 2:
        st.warning("Comparison result did not return exactly two models.")
        return

    m1 = models_data[0]
    m2 = models_data[1]

    model_1_name = MODEL_METADATA.get(m1["model_name"], {}).get(
        "display_name", m1["model_name"]
    )
    model_2_name = MODEL_METADATA.get(m2["model_name"], {}).get(
        "display_name", m2["model_name"]
    )

    st.subheader("Biomass predictions comparison")

    _render_comparison_bar_chart(
        pred_model1=m1["biomass"],
        pred_model2=m2["biomass"],
        model_1_name=model_1_name,
        model_2_name=model_2_name,
    )

    _render_error_summary_simple(m1["biomass"], m2["biomass"], model_1_name, model_2_name)

    st.markdown("---")
    _render_explainability_section(state, m1, m2, model_1_name, model_2_name)



def _render_comparison_bar_chart(
    pred_model1: Dict[str, float],
    pred_model2: Dict[str, float],
    model_1_name: str,
    model_2_name: str,
) -> None:
    rows = []
    for key in BIOMASS_KEYS:
        display_name = BIOMASS_DISPLAY[key]
        rows.append(
            {
                "Biomass": display_name,
                "Source": model_1_name,
                "Value": float(pred_model1.get(key, 0.0)),
            }
        )
        rows.append(
            {
                "Biomass": display_name,
                "Source": model_2_name,
                "Value": float(pred_model2.get(key, 0.0)),
            }
        )

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Biomass:N", title="Biomass type"),
            y=alt.Y("Value:Q", title="Value (g)"),
            color=alt.Color("Source:N"),
            column=alt.Column("Biomass:N", title=None),
        )
        .resolve_scale(y="independent")
        .properties(height=200)
    )

    st.altair_chart(chart, use_container_width=True)



def _render_error_summary_simple(
    pred_model1: Dict[str, float],
    pred_model2: Dict[str, float],
    model_1_name: str,
    model_2_name: str,
) -> None:
    st.markdown("### Summary (no ground truth available)")
    vals1 = np.array([pred_model1.get(k, 0.0) for k in BIOMASS_KEYS], dtype=float)
    vals2 = np.array([pred_model2.get(k, 0.0) for k in BIOMASS_KEYS], dtype=float)

    mean1 = float(vals1.mean())
    mean2 = float(vals2.mean())

    st.markdown(
        f"- Mean predicted biomass for **{model_1_name}**: `{mean1:.3f}` g\n"
        f"- Mean predicted biomass for **{model_2_name}**: `{mean2:.3f}` g"
    )



# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
def _render_explainability_section(
    state: Dict[str, Any],
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    model_1_name: str,
    model_2_name: str,
) -> None:
    st.subheader("Explainability (Grad-CAM)")

    focus = st.radio(
        "Choose biomass focus",
        options=FOCUS_OPTIONS,
        format_func=lambda k: FOCUS_TO_LABEL.get(k, k),
        key="edu_focus_radio",
        horizontal=True,
    )
    state["focus"] = focus

    expl1 = m1.get("explanation", {})
    expl2 = m2.get("explanation", {})

    heatmaps1 = expl1.get("heatmaps", {})
    heatmaps2 = expl2.get("heatmaps", {})

    hm1 = heatmaps1.get(focus)
    hm2 = heatmaps2.get(focus)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Model 1: {model_1_name}**")
        if hm1 is not None:
            _show_heatmap_values(hm1)
        else:
            st.warning(f"No heatmap for {focus}.")

    with col2:
        st.markdown(f"**Model 2: {model_2_name}**")
        if hm2 is not None:
            _show_heatmap_values(hm2)
        else:
            st.warning(f"No heatmap for {focus}.")

    st.caption(
        "These Grad-CAM heatmaps highlight where each model focused when predicting "
        "the selected biomass component. Darker/hotter regions indicate higher importance."
    )


def _show_heatmap_values(hm: Dict[str, Any], cmap_name: str = "jet") -> None:
    """
    Show Grad-CAM / saliency heatmap as a colored image.
    `values` is expected to be a 2D list of floats in [0,1].
    """
    values = hm.get("values")
    if values is None:
        st.warning("Heatmap has no 'values' field.")
        return

    arr = np.array(values, dtype=float)   # [H, W]

    if arr.size == 0:
        st.warning("Empty heatmap.")
        return

    # Robust normalisation to [0, 1]
    arr = np.nan_to_num(arr)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    # Apply colormap -> RGB in [0,1]
    cmap = cm.get_cmap(cmap_name)         # "jet", "viridis", "plasma", ...
    colored = cmap(arr)[..., :3]          # [H, W, 3], drop alpha channel

    # Convert to uint8 for st.image
    colored_u8 = (colored * 255).astype(np.uint8)

    st.image(colored_u8, use_column_width=True)
