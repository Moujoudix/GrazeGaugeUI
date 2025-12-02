# ui_predict.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objs as go
import textwrap
from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    BIOMASS_UNIT,
    MODEL_METADATA,
    MODEL_ORDER,
    CORE_BIOMASS_KEYS,
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
# Controls area
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

    # Build domain & range for the color scale from your keys and dict
    color_scale = alt.Scale(
        domain=BIOMASS_KEYS,
        range=[BIOMASS_COLORS[k] for k in BIOMASS_KEYS],
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Biomass:N", sort=None, title="Biomass type"),
            y=alt.Y("Value:Q", title=f"Value ({BIOMASS_UNIT})"),
            color=alt.Color("Key:N", scale=color_scale, legend=None),
            tooltip=["Biomass", "Value"],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width='stretch')


def _render_biomass_radar_chart(biomass: Dict[str, float]) -> None:
    # Radar chart using Plotly
    categories = [BIOMASS_DISPLAY[k] for k in CORE_BIOMASS_KEYS]
    values = [biomass.get(k, 0.0) for k in CORE_BIOMASS_KEYS]
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
    If uncertainty is None, the column will be filled with '-'.
    """
    rows = []
    for key in BIOMASS_KEYS:
        value = float(biomass.get(key, 0.0))
        if uncertainty is not None:
            unc = float(uncertainty.get(key, 0.0))
            unc_str = f"±{unc:.2f}"
        else:
            unc_str = "--"

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


QUADRAT_AREA_M2 = 0.7 * 0.3  # 0.21 m²
def biomass_g_to_kg_ha(biomass_g: float) -> float:
    """Convert grams per 70x30 cm quadrat to kg DM/ha."""
    return (biomass_g / QUADRAT_AREA_M2) * 10.0

def _triangular_score(x: float,
                      low: float,
                      opt_low: float,
                      opt_high: float,
                      high: float) -> float:
    """
    Generic triangular scoring function in [0, 1].

    low      ... minimum, below/at this score = 0
    opt_low  ... start of 'ideal' plateau
    opt_high ... end of 'ideal' plateau
    high     ... maximum, above/at this score = 0
    """
    if x <= low or x >= high:
        return 0.0
    if opt_low <= x <= opt_high:
        return 1.0
    if x < opt_low:
        return (x - low) / (opt_low - low)
    # x > opt_high
    return (high - x) / (high - opt_high)


def _linear_score(x: float, low: float, high: float) -> float:
    """Linear score in [0, 1] between low and high."""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)

def _compute_grazing_score_and_suggestion(
    avg_biomass: Dict[str, float]) -> tuple[float, str, str]:
    """
    Returns (score_0_to_100, label, recommendation).

    score:
        0 -30  = poor for grazing now
        30-60 = marginal / OK
        60-80 = good
        80-100= very good / ideal

    label:
        'BARE', 'REST', 'GRAZE', or 'REJUVENATE'
    """
    g = avg_biomass.get("Dry_Green_g", 0.0)
    d = avg_biomass.get("Dry_Dead_g", 0.0)
    c = avg_biomass.get("Dry_Clover_g", 0.0)

    total = g + c + d
    if total < 1e-3:
        return 0.0, "BARE", """
                            NO FEED: effectively bare ground. Biomass in this quadrat is close to zero,
                            so there is nothing meaningful for animals to harvest. Keep livestock off this
                            area and allow plants to establish or recover. If this pattern persists,
                            consider assessing soil fertility, compaction and weed pressure, and think
                            about reseeding or pasture renovation.
                            """

    gdm = g + c
    kg_ha_gdm = biomass_g_to_kg_ha(gdm)

    quality_ratio = gdm / total              # share of green+clover
    clover_frac   = (c / gdm) if gdm > 0 else 0.0

    # ----- LABEL + TEXT (same basic logic as before) -----

    # Too little green feed
    if kg_ha_gdm < 1500:
        label = "REST"
        rec = """
                Too little green feed to graze safely. Grazeable biomass is below the
                typical post-grazing residual (~1500 kg DM/ha), so grazing now would force
                animals to eat very close to the base of the plants. That weakens root
                reserves, slows regrowth and increases soil exposure. Leave this paddock to
                regrow until the sward has rebuilt a leafy canopy before re-introducing stock.
                """

    # Too much cover
    elif kg_ha_gdm > 3000:
        if quality_ratio < 0.6:
            label = "REJUVENATE"
            rec = """
                    pasture is over-mature with a high proportion of dead or stemmy
                    material. At this stage digestibility and energy are reduced and animals will
                    selectively avoid the old material, leading to patchy grazing and thatch
                    build-up. Use a clean-up strategy (tight grazing with more stock, or topping /
                    mowing) to remove the old material, then allow a full leafy regrowth before
                    the next grazing.
                    """

        else:
            label = "GRAZE"
            rec = """
                    REDUCE COVER: feed level is higher than ideal (>3000 kg green DM/ha),
                    but the sward is still mostly green leaf. If you delay much longer, more of
                    this leaf will turn into dead material and overall quality will drop. Consider
                    grazing with a higher stocking density or cutting for silage / hay so that the
                    paddock returns to a more optimal 2500 → 1500 kg DM/ha rotation.
                    """

    # In normal grazing window (1500-3000 kg GDM/ha)
    else:
        if quality_ratio < 0.6:
            label = "GRAZE"
            rec = """
                    GRAZE SOON: total biomass is in a normal grazing window, but dead material is
                    starting to build up (a significant share of the sward). This reduces
                    digestibility and voluntary intake even though there is still reasonable
                    green cover. Plan to graze this paddock in the next pass before more of the
                    green leaf is lost and the dead fraction increases further.
                    """
        else:
            if 0.15 <= clover_frac <= 0.4:
                label = "GRAZE"
                rec = """
                        IDEAL: feed level and quality are both in the sweet spot. Most of
                        the biomass is green leaf with a healthy proportion of clover, which typically
                        means high digestibility, good metabolisable energy and strong crude protein
                        levels. Grazing now should support good live-weight gain or milk production
                        while still leaving a dense residual to protect soil and drive rapid regrowth.
                        """
            elif clover_frac < 0.15:
                label = "GRAZE"
                rec = """
                        grass quantity and quality are good, but clover content is on the
                        low side. Animals can still perform well, but most of the nitrogen in this
                        paddock will be coming from fertiliser or soil reserves rather than biological
                        fixation by legumes. Over the medium term you may want to encourage more
                        clover by moderating nitrogen fertiliser, reducing shading from rank grass,
                        or oversowing improved clover cultivars.
                        """
            else:  # clover_frac > 0.4
                label = "GRAZE"
                rec = """
                        GRAZE BUT MANAGE BLOAT RISK: feed level and quality are high and the
                        sward is very clover-rich. Clover is highly digestible and protein-dense,
                        which is excellent for production but increases the risk of frothy bloat,
                        especially in cattle. Introduce animals gradually, avoid turning them in when
                        they are very hungry, provide access to roughage (hay or straw), and consider
                        bloat oil, blocks or other preventive measures where appropriate.
                        """

    # ----- NUMERIC SCORE (0-100) -----
    # 1) Quantity score: best around 1800-2600 kg DM/ha
    quantity_score = _triangular_score(
        kg_ha_gdm,
        low=1000.0,
        opt_low=1800.0,
        opt_high=2600.0,
        high=3800.0,
    )

    # 2) Quality score: 0 when <=40% green, 1 around 90%+ green
    quality_score = _linear_score(
        quality_ratio,
        low=0.40,
        high=0.90,
    )

    # 3) Clover score: best ~20-35% of green DM
    clover_score = _triangular_score(
        clover_frac,
        low=0.05,
        opt_low=0.20,
        opt_high=0.35,
        high=0.55,
    )

    # Combine: quantity has most weight, then overall green quality, then clover
    overall = (
        0.5 * quantity_score +
        0.3 * quality_score +
        0.2 * clover_score
    )

    score = round(100.0 * max(0.0, min(1.0, overall)), 1)  # clamp & keep 1 decimal

    return score, label, rec


def render_grazing_card(score: float, label: str, suggestion: str) -> None:
    color = grazing_color(label, score)
    bg = "#f9fafb"
    text = "#111827"
    subtext = "#4b5563"

    card_html = f"""
<div style="border-radius:16px;border:1px solid {color};
            padding:1.3rem 1.5rem;margin-top:0.75rem;
            background:{bg};box-shadow:0 6px 18px rgba(15,23,42,.08);">

  <!-- Header: title + label side by side -->
  <div style="display:flex;align-items:center;margin-bottom:0.55rem;gap:0.75rem;">
    <div style="font-size:2.2rem;text-transform:uppercase;
                letter-spacing:.16em;color:{subtext};
                font-weight:700;white-space:nowrap;">
      Grazing quality score
    </div>

    <span style="display:inline-flex;align-items:right;
                 padding:0.25rem 1.0rem;border-radius:999px;
                 border:1px solid {color};background:rgba(34,197,94,.06);
                 font-size:2.0rem;font-weight:700;color:{color};">
      {label}
    </span>
  </div>

 <!-- Score row: score + bar aligned horizontally -->
<div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;">

  <!-- Score number -->
  <div style="display:flex;align-items:baseline;gap:0.25rem;">
    <div style="font-size:2.0rem;font-weight:800;color:{color};">
      {score:.1f}/100
    </div>
  </div>

  <!-- Progress bar + label in one horizontal row -->
  <div style="flex:1;min-width:220px;display:flex;align-items:center;gap:0.75rem;">

    <div style="font-size:1.0rem;color:{subtext};white-space:nowrap;">
      Readiness to graze
    </div>

    <div style="position:relative;height:12px;width:30%;
                border-radius:999px;background:#e5e7eb;overflow:hidden;border:1px solid {color};">
      <div style="position:absolute;top:0;left:0;height:100%;
                  width:{max(0, min(score, 100))}%;background:{color};">
      </div>
    </div>

  </div>

</div>


  <!-- Recommendation text -->
  <div style="margin-top:0.9rem;font-size:1.5rem;
              color:{text};line-height:1.5;font-weight:500;">
    {suggestion}
  </div>

</div>
"""

    st.html(card_html)

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
    render_grazing_card(score, label, suggestion)



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

def grazing_color(label: str, score: float) -> str:
    """
    Map (label, score) -> hex color.

    Labels:
        BARE        -> greys
        REST        -> reds
        REJUVENATE  -> ambers/oranges
        GRAZE       -> yellow-green to deep green
    """
    s = max(0.0, min(score, 100.0))

    label = (label or "").upper()

    # 1) BARE: neutral / no feed
    if label == "BARE":
        # darker grey for very low scores (really bare)
        if s < 10:
            return "#4b5563"   # gray-600
        elif s < 30:
            return "#6b7280"   # gray-500
        else:
            return "#9ca3af"   # gray-400

    # 2) REST: not enough grass -> red scale
    if label == "REST":
        if s < 15:
            return "#7f1d1d"   # red-800 (very urgent)
        elif s < 25:
            return "#b91c1c"   # red-700
        elif s > 50:
            return "#f97316"
        else:
            return "#ef4444"   # red-500 (still clear "stop")

    # 3) REJUVENATE: over-mature / clean-up -> amber/orange
    if label == "REJUVENATE":
        if s < 40:
            return "#92400e"   # amber-800 (poor quality, lots of dead)
        elif s < 70:
            return "#f97316"   # orange-500
        else:
            return "#fbbf24"   # amber-400 (still warning-ish but lighter)

    # 4) GRAZE: can graze, gradient from lime → deep green
    if label == "GRAZE":
        if s < 50:
            return "#a3e635"   # lime-400 (borderline / ok)
        elif s < 80:
            return "#22c55e"   # green-500 (good)
        else:
            return "#15803d"   # green-700 (excellent / ideal)

    # Fallback if label is unknown
    # use score-only gradient: red -> amber -> green
    if s < 30:
        return "#ef4444"
    elif s < 60:
        return "#f97316"
    elif s < 80:
        return "#4ade80"
    else:
        return "#15803d"
