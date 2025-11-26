from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import streamlit as st
import altair as alt
from PIL import Image

from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    MODEL_METADATA,
    MODEL_ORDER,
    FOCUS_OPTIONS,      # list of biomass keys to focus on, e.g. ["Dry_Green_g", ...]
    FOCUS_TO_LABEL,
)
from api_client import call_compare_api


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def init_educational_state() -> None:
    """
    State for the Educational Lab page.

    - model_1_id / model_2_id: selected model names
    - example_image: UploadedFile from Streamlit
    - comparison_result: raw JSON from /compare
    - focus: current biomass key focus (one of FOCUS_OPTIONS)
    - api_error: last API error (if any)
    """
    if "edu_state" not in st.session_state:
        st.session_state["edu_state"] = {
            "model_1_id": None,
            "model_2_id": None,
            "example_image": None,       # UploadedFile
            "comparison_result": None,   # raw JSON from /compare
            "focus": FOCUS_OPTIONS[0],   # default first key
            "api_error": None,
        }


# -----------------------------------------------------------------------------
# Page entrypoint
# -----------------------------------------------------------------------------
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

    # Model selection
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

    # Image upload
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
# Comparison result (predictions + summary)
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
        gt_biomass=None,  # or real GT dict when you have it
    )


    _render_summary_simple(m1["biomass"], m2["biomass"], model_1_name, model_2_name)

    st.markdown("---")
    _render_explainability_section(state, m1, m2, model_1_name, model_2_name)


def _render_comparison_bar_chart(
    pred_model1: Dict[str, float],
    pred_model2: Dict[str, float],
    model_1_name: str,
    model_2_name: str,
    gt_biomass: Optional[Dict[str, float]] = None,
) -> None:
    """
    Grouped bar chart: one group per biomass type, 2 or 3 bars per group:
    - Ground truth (if available)
    - Model 1
    - Model 2
    """
    rows = []

    # Optional ground truth as a third bar
    if gt_biomass is not None:
        for key in BIOMASS_KEYS:
            rows.append(
                {
                    "Biomass": BIOMASS_DISPLAY[key],
                    "Source": "Ground truth",
                    "Value": float(gt_biomass.get(key, 0.0)),
                }
            )

    # Model 1
    for key in BIOMASS_KEYS:
        rows.append(
            {
                "Biomass": BIOMASS_DISPLAY[key],
                "Source": model_1_name,
                "Value": float(pred_model1.get(key, 0.0)),
            }
        )

    # Model 2
    for key in BIOMASS_KEYS:
        rows.append(
            {
                "Biomass": BIOMASS_DISPLAY[key],
                "Source": model_2_name,
                "Value": float(pred_model2.get(key, 0.0)),
            }
        )

    df = pd.DataFrame(rows)

    # Grouped bars: x = Biomass, xOffset = Source
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Biomass:N", title="Biomass type"),
            xOffset="Source:N",
            y=alt.Y("Value:Q", title="Value (g)"),
            color=alt.Color("Source:N", title=""),
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)



def _render_summary_simple(
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
# Explainability (Grad-CAM overlays)
# -----------------------------------------------------------------------------
def _render_explainability_section(
    state: Dict[str, Any],
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    model_1_name: str,
    model_2_name: str,
) -> None:
    st.subheader("Explainability (Grad-CAM)")

    # Choose biomass focus
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

    # Per-model normalisation: global vmin/vmax across the three focus targets
    vmin1, vmax1 = _compute_global_range(
        [heatmaps1.get(k) for k in FOCUS_OPTIONS if heatmaps1.get(k) is not None]
    )
    vmin2, vmax2 = _compute_global_range(
        [heatmaps2.get(k) for k in FOCUS_OPTIONS if heatmaps2.get(k) is not None]
    )

    hm1 = heatmaps1.get(focus)
    hm2 = heatmaps2.get(focus)

    uploaded = state.get("example_image")
    base_image: Optional[Image.Image] = None
    if uploaded is not None:
        # Convert UploadedFile to PIL Image
        base_image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Model 1: {model_1_name}**")
        if hm1 is not None:
            if base_image is not None:
                _show_overlay(base_image, hm1, vmin=vmin1, vmax=vmax1)
            else:
                _show_heatmap_values(hm1, vmin=vmin1, vmax=vmax1)
        else:
            st.warning(f"No heatmap for {FOCUS_TO_LABEL.get(focus, focus)}.")

    with col2:
        st.markdown(f"**Model 2: {model_2_name}**")
        if hm2 is not None:
            if base_image is not None:
                _show_overlay(base_image, hm2, vmin=vmin2, vmax=vmax2)
            else:
                _show_heatmap_values(hm2, vmin=vmin2, vmax=vmax2)
        else:
            st.warning(f"No heatmap for {FOCUS_TO_LABEL.get(focus, focus)}.")

    st.caption(
        "These Grad-CAM heatmaps highlight where each model focused when predicting "
        "the selected biomass component. Hotter regions indicate higher importance. "
        "Colors are normalised per model across the three focus targets."
    )


# -----------------------------------------------------------------------------
# Heatmap utilities
# -----------------------------------------------------------------------------
def _compute_global_range(hms: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Compute global min/max across a list of heatmaps (each with 'values').
    Used for per-model normalisation across multiple targets.
    """
    all_vals = []
    for hm in hms:
        if hm is None:
            continue
        vals = hm.get("values")
        if vals is None:
            continue
        arr = np.array(vals, dtype=float)
        arr = np.nan_to_num(arr)
        all_vals.append(arr.ravel())

    if not all_vals:
        return 0.0, 1.0

    stacked = np.concatenate(all_vals)
    return float(stacked.min()), float(stacked.max())


def _heatmap_to_rgb(
    values: Any,
    vmin: float,
    vmax: float,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Convert a 2D heatmap + shared vmin/vmax into RGB uint8.
    """
    arr = np.array(values, dtype=float)  # [H, W]
    arr = np.nan_to_num(arr)

    if vmax <= vmin:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm)[..., :3]  # [H, W, 3] floats in [0,1]
    rgb_u8 = (rgb * 255).astype(np.uint8)
    return rgb_u8


def _show_heatmap_values(
    hm: Dict[str, Any],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "jet",
) -> None:
    """
    Fallback: show heatmap alone (no overlay).
    """
    values = hm.get("values")
    if values is None:
        st.warning("Heatmap has no 'values' field.")
        return

    arr = np.array(values, dtype=float)
    arr = np.nan_to_num(arr)

    if vmin is None or vmax is None:
        vmin = float(arr.min())
        vmax = float(arr.max())

    rgb_u8 = _heatmap_to_rgb(arr, vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    st.image(rgb_u8, use_column_width=True)


def _show_overlay(
    base_image: Image.Image,
    hm: Dict[str, Any],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.35,
    cmap_name: str = "jet",
) -> None:
    """
    Overlay a heatmap onto the original image using the given vmin/vmax.
    """
    values = hm.get("values")
    if values is None:
        st.warning("Heatmap has no 'values' field.")
        return

    arr = np.array(values, dtype=float)
    arr = np.nan_to_num(arr)

    if vmin is None or vmax is None:
        vmin = float(arr.min())
        vmax = float(arr.max())

    # Heatmap RGB
    rgb_hm = _heatmap_to_rgb(arr, vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    H, W, _ = rgb_hm.shape

    # Resize original image to match heatmap resolution
    base_resized = base_image.resize((W, H))
    base_np = np.array(base_resized, dtype=np.float32) / 255.0
    if base_np.ndim == 2:  # grayscale
        base_np = np.stack([base_np] * 3, axis=-1)

    hm_np = rgb_hm.astype(np.float32) / 255.0

    overlay = (1.0 - alpha) * base_np + alpha * hm_np
    overlay_u8 = (overlay * 255).astype(np.uint8)

    st.image(overlay_u8, use_column_width=True)
