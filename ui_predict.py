# ui_predict.py
from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objs as go

from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    BIOMASS_UNIT,
    MODEL_METADATA,
    MODEL_ORDER,
)
from api_client import call_predict_api


# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------
def init_predict_state() -> None:
    """Initialize nested predict_state in st.session_state if missing."""
    if "predict_state" not in st.session_state:
        st.session_state["predict_state"] = {
            "uploaded_files": None,        # list[UploadedFile] or None
            "predictions_by_filename": {},  # filename -> {"biomass": {...}}
            "raw_response": None,          # last API JSON
            "batch_summary": None,         # dict or None
            "show_advanced": False,
            "last_model_id": None,
            #"last_aux_heads": False,
            "api_error": None,
        }


# -----------------------------------------------------------------------------
# Core page entrypoint
# -----------------------------------------------------------------------------
def render_predict_page() -> None:
    """
    Predict page layout:

    - Top: controls area in two columns
      - Left: upload + run + advanced toggle
      - Right: model config + AUX + description
    - Below: results section (single or multi-image)
    """
    init_predict_state()
    state = st.session_state["predict_state"]

    _render_predict_controls(state)
    st.markdown("---")
    _render_predict_results(state)


# -----------------------------------------------------------------------------
# Controls area (top)
# -----------------------------------------------------------------------------
def _render_predict_controls(state: Dict[str, Any]) -> None:
    """
    Top controls layout:

    [Col 1]  1. Upload images & Run
    [Col 2]  2. Model configuration & model info
    """
    col_upload, col_model = st.columns([1, 1])

    # We define model config first so we can use model_id/aux_heads when user clicks "Run"
    with col_model:
        st.subheader("2. Model configuration")

        model_id = st.selectbox(
            "Choose model",
            options=MODEL_ORDER,
            format_func=lambda m: MODEL_METADATA[m]["display_name"],
            key="predict_model_select",
        )

        _render_model_description_card(model_id)

    with col_upload:
        st.subheader("1. Upload images & run")

        uploaded_files = st.file_uploader(
            "Upload one or more pasture images (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="predict_uploader",
        )
        state["uploaded_files"] = uploaded_files

        # Space between uploader and controls
        st.markdown("&nbsp;", unsafe_allow_html=True)

        state["show_advanced"] = False

        if st.button("Run prediction", type="primary", key="predict_run_button"):
            _handle_run_prediction(state, model_id)


def _render_model_description_card(model_id: str) -> None:
    meta = MODEL_METADATA[model_id]
    with st.expander("Model info", expanded=True):
        st.markdown(f"**Model:** {meta.get('display_name', model_id)}")

        backbone = meta.get("backbone_name")
        if backbone:
            st.markdown(f"**Backbone:** `{backbone}`")

        img_h = meta.get("img_height")
        img_w = meta.get("img_width")
        if img_h and img_w:
            st.markdown(f"**Input size:** {img_h} × {img_w}")

        desc = meta.get("description")
        if desc:
            st.markdown(f"**Description:** {desc}")



def _handle_run_prediction(
    state: Dict[str, Any],
    model_id: str,
) -> None:
    uploaded_files = state.get("uploaded_files") or []
    if not uploaded_files:
        st.warning("Please upload at least one image before running predictions.")
        return

    state["api_error"] = None

    with st.spinner("Running model on uploaded images..."):
        try:
            response = call_predict_api(
                images=uploaded_files,
                model_name=model_id,
            )
        except Exception as exc:  # noqa: BLE001
            state["api_error"] = str(exc)
            try:
                st.toast(f"Prediction failed: {exc}", icon="❌")
            except Exception:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
            return

    predictions = _parse_prediction_response(response)
    state["predictions_by_filename"] = predictions
    state["raw_response"] = response
    state["last_model_id"] = model_id

    # Compute batch summary if multiple images
    if len(predictions) > 1:
        batch_summary = _compute_batch_summary(predictions)
        state["batch_summary"] = batch_summary
    else:
        state["batch_summary"] = None

    try:
        st.toast("Predictions ready!", icon="✅")
    except Exception:  # noqa: BLE001
        st.success("Predictions ready!")


def _parse_prediction_response(response: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert API JSON into:
    {
      "filename.jpg": {
          "biomass": {...},
          "uncertainty": {...}   # NEW
      },
      ...
    }
    """
    preds: Dict[str, Dict[str, Any]] = {}
    raw_preds = response.get("predictions", [])

    for item in raw_preds:
        filename = item.get("filename", "unknown")
        biomass = item.get("biomass", {}) or {}
        uncertainty = item.get("uncertainty", {}) or {}

        clean_biomass = {k: float(biomass.get(k, 0.0)) for k in BIOMASS_KEYS}
        clean_uncertainty = {k: float(uncertainty.get(k, 0.0)) for k in BIOMASS_KEYS}

        preds[filename] = {
            "biomass": clean_biomass,
            "uncertainty": clean_uncertainty,   # NEW
        }

    return preds


# -----------------------------------------------------------------------------
# Results area (below controls)
# -----------------------------------------------------------------------------
def _render_predict_results(state: Dict[str, Any]) -> None:
    api_error = state.get("api_error")
    if api_error:
        st.error(f"Last prediction error: {api_error}")

    uploaded_files = state.get("uploaded_files") or []
    preds = state.get("predictions_by_filename") or {}
    show_advanced = state.get("show_advanced", False)

    if not uploaded_files:
        st.info(
            "No predictions yet. Upload one or more pasture images above and "
            "click **Run prediction**."
        )
        return

    if not preds:
        # Images uploaded but no predictions yet
        st.info(
            f"{len(uploaded_files)} image(s) uploaded. "
            "Click **Run prediction** to get biomass estimates."
        )
        return

    filenames = list(preds.keys())
    if len(filenames) == 1:
        filename = filenames[0]
        biomass = preds[filename]["biomass"]
        uncertainty = preds[filename].get("uncertainty")

        _render_single_prediction_card(
            filename=filename,
            biomass=biomass,
            show_advanced=show_advanced,
            raw_response=state.get("raw_response"),
            uploaded_files=uploaded_files,
            uncertainty=uncertainty,
        )

    else:
        batch_summary = state.get("batch_summary")
        if batch_summary is not None:
            _render_batch_summary_card(batch_summary)
        _render_prediction_grid_multi(
            predictions=preds,
            uploaded_files=uploaded_files,
            show_advanced=show_advanced,
            raw_response=state.get("raw_response"),
        )


def _find_uploaded_file(
    uploaded_files: List[Any],
    filename: str,
) -> Optional[Any]:
    for f in uploaded_files:
        if f.name == filename:
            return f
    return None


def _render_single_prediction_card(
    filename: str,
    biomass: Dict[str, float],
    show_advanced: bool,
    raw_response: Dict[str, Any] | None,
    uploaded_files: List[Any],
    uncertainty: Optional[Dict[str, float]] = None,  # NEW
    show_image: bool = True,
    show_table:bool = True,
) -> None:
    """
    Single-image layout:

    - (optional) Image
    - Biomass table
    - Two-column "subplot":
        left  = bar chart
        right = radar chart
    """
    st.subheader(f"Prediction for **{filename}**")

    file_obj = _find_uploaded_file(uploaded_files, filename)

    with st.container():
        # Image at the top (if requested)
        if show_image:
            if file_obj is not None:
                st.image(file_obj, width='stretch', caption=filename)
            else:
                st.warning("Could not find the original image in uploaded files.")

        # Table just under the image
        if show_table:
            st.markdown("**Biomass estimates**")
            df = _build_biomass_table(biomass, uncertainty)
            st.table(df)

        # 2-chart "subplot" row
        st.markdown("**Visualizations**")
        col_bar, col_radar = st.columns(2)

        with col_bar:
            st.markdown("Biomass composition")
            _render_biomass_bar_chart(biomass)

        with col_radar:
            st.markdown("Shape overview")
            _render_biomass_radar_chart(biomass)

    if show_advanced and raw_response is not None:
        with st.expander("Show raw JSON from API"):
            st.json(raw_response)


def _render_biomass_bar_chart(biomass: Dict[str, float]) -> None:
    data = pd.DataFrame(
        {
            "Biomass": [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS],
            "Value": [biomass.get(k, 0.0) for k in BIOMASS_KEYS],
            "Key": BIOMASS_KEYS,
        }
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Biomass:N", sort=None, title="Biomass type"),
            y=alt.Y("Value:Q", title=f"Value ({BIOMASS_UNIT})"),
            color=alt.Color("Biomass:N", legend=None),
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width='stretch')


def _render_biomass_radar_chart(biomass: Dict[str, float]) -> None:
    # Radar chart using Plotly
    categories = [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS]
    values = [biomass.get(k, 0.0) for k in BIOMASS_KEYS]
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Prediction",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig, width='stretch')

def _build_biomass_table(
    biomass: Dict[str, float],
    uncertainty: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a table with biomass values and optional uncertainty (±).
    If uncertainty is None, the column will be filled with '–'.
    """
    rows = []
    for key in BIOMASS_KEYS:
        value = float(biomass.get(key, 0.0))
        if uncertainty is not None:
            unc = float(uncertainty.get(key, 0.0))
            unc_str = f"±{unc:.2f}"
        else:
            unc_str = "––"

        rows.append(
            {
                "Biomass": BIOMASS_DISPLAY[key],
                "Value": value,
                "Uncertainty (±)": unc_str,
            }
        )

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Batch summary & grid
# -----------------------------------------------------------------------------
def _compute_batch_summary(
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute average biomasses + grazing score/suggestion.
    """
    values = np.array(
        [
            [predictions[f]["biomass"][k] for k in BIOMASS_KEYS]
            for f in predictions.keys()
        ]
    )
    mean_values = values.mean(axis=0)
    avg_biomass = {k: float(v) for k, v in zip(BIOMASS_KEYS, mean_values)}

    score, label, suggestion = _compute_grazing_score_and_suggestion(avg_biomass)

    return {
        "avg_biomass": avg_biomass,
        "grazing_score": score,
        "grazing_label": label,
        "suggestion": suggestion,
    }


def _compute_grazing_score_and_suggestion(
    avg_biomass: Dict[str, float],
) -> tuple[float, str, str]:
    total = avg_biomass.get("Dry_Total_g", 0.0)
    green = avg_biomass.get("Dry_Green_g", 0.0)
    dead = avg_biomass.get("Dry_Dead_g", 0.0)
    clover = avg_biomass.get("Dry_Clover_g", 0.0)

    if total > 0:
        green_ratio = green / total
        dead_ratio = dead / total
        clover_ratio = clover / total
    else:
        green_ratio = dead_ratio = clover_ratio = 0.0

    # Simple illustrative scoring heuristic
    score = 100 * (0.6 * green_ratio + 0.3 * clover_ratio - 0.4 * dead_ratio)
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 70:
        label = "High"
        suggestion = (
            "Pasture appears in good condition, with a high proportion of "
            "green biomass. Suitable for grazing."
        )
    elif score >= 40:
        label = "Medium"
        suggestion = (
            "Pasture is moderately green. Consider moderate grazing intensity "
            "or some rest periods."
        )
    else:
        label = "Low"
        suggestion = (
            "Green biomass is relatively low compared to dead material. "
            "Pasture may require rest or management before further grazing."
        )

    return score, label, suggestion


def _render_batch_summary_card(batch_summary: Dict[str, Any]) -> None:
    st.subheader("Batch summary")

    avg = batch_summary["avg_biomass"]
    score = batch_summary["grazing_score"]
    label = batch_summary["grazing_label"]
    suggestion = batch_summary["suggestion"]

    col_top, col_bottom = st.columns([1, 1])

    with col_top:
        st.markdown("**Average biomass across images**")
        df = pd.DataFrame(
            {
                "Biomass": [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS],
                "Average value": [avg.get(k, 0.0) for k in BIOMASS_KEYS],
            }
        )
        st.table(df)

    with col_bottom:
        st.markdown("**Average composition**")
        _render_biomass_bar_chart(avg)

    st.markdown("---")
    st.markdown(f"**Grazing quality score:** {score:.1f} / 100  —  **{label}**")

    st.info(
        suggestion
        + "  \n\n"
        "_Note: this is an illustrative, rule-based score, not a "
        "professional agronomic recommendation._"
    )


def _render_prediction_grid_multi(
    predictions: Dict[str, Dict[str, Any]],
    uploaded_files: List[Any],
    show_advanced: bool,
    raw_response: Dict[str, Any] | None,
) -> None:
    st.subheader("Per-image predictions")

    filenames = list(predictions.keys())
    n = len(filenames)
    n_cols = 2 if n <= 4 else 3

    cols = st.columns(n_cols)

    for i, filename in enumerate(filenames):
        col = cols[i % n_cols]
        with col:
            st.markdown(f"**{filename}**")
            file_obj = _find_uploaded_file(uploaded_files, filename)
            if file_obj is not None:
                st.image(file_obj, width='stretch')

            biomass = predictions[filename]["biomass"]
            uncertainty = predictions[filename].get("uncertainty")

            df = _build_biomass_table(biomass, uncertainty)
            st.table(df)


            with st.expander("View full details"):
                _render_single_prediction_card(
                    filename=filename,
                    biomass=biomass,
                    show_advanced=show_advanced,
                    raw_response=raw_response,
                    uploaded_files=uploaded_files,
                    uncertainty=uncertainty,
                    show_image=False,
                    show_table=False,
                )
