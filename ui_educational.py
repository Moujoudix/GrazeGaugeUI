# ui_educational.py
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    MODEL_METADATA,
    MODEL_ORDER,
    FOCUS_OPTIONS,
    FOCUS_TO_BIOMASS_KEY,
    load_validation_examples,
)


# Validation examples are read-only and can be cached at module import
VALIDATION_EXAMPLES = load_validation_examples()


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
    st.subheader("Model comparison")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        model_1_id = st.selectbox(
            "Model 1",
            options=[None] + MODEL_ORDER,
            format_func=lambda m: MODEL_METADATA[m]["display_name"]
            if m is not None
            else "Select...",
            key="edu_model_1",
        )

    with col2:
        model_2_id = st.selectbox(
            "Model 2",
            options=[None] + MODEL_ORDER,
            format_func=lambda m: MODEL_METADATA[m]["display_name"]
            if m is not None
            else "Select...",
            key="edu_model_2",
        )

    with col3:
        st.markdown("**AUX heads**")
        model_1_aux = st.checkbox(
            "Model 1 AUX",
            value=True
            if (model_1_id and MODEL_METADATA[model_1_id].get("supports_aux", False))
            else False,
            disabled=not (model_1_id and MODEL_METADATA[model_1_id].get("supports_aux", False)),
            key="edu_model_1_aux",
        )
        model_2_aux = st.checkbox(
            "Model 2 AUX",
            value=True
            if (model_2_id and MODEL_METADATA[model_2_id].get("supports_aux", False))
            else False,
            disabled=not (model_2_id and MODEL_METADATA[model_2_id].get("supports_aux", False)),
            key="edu_model_2_aux",
        )

    state["model_1_id"] = model_1_id
    state["model_2_id"] = model_2_id
    state["model_1_aux"] = model_1_aux
    state["model_2_aux"] = model_2_aux

    st.markdown(
        "_Select two models and click **Compare on random image** to see how "
        "they perform on the same validation example._"
    )

    if st.button("Compare on random image", type="primary", key="edu_compare_button"):
        _handle_compare_on_random(state)


def _handle_compare_on_random(state: Dict[str, Any]) -> None:
    model_1_id = state["model_1_id"]
    model_2_id = state["model_2_id"]

    if model_1_id is None or model_2_id is None:
        st.warning("Please select both Model 1 and Model 2.")
        return

    if not VALIDATION_EXAMPLES:
        st.warning(
            "No validation examples are configured yet. "
            "Please add them in config.load_validation_examples()."
        )
        return

    # Pick random validation example
    idx = np.random.randint(0, len(VALIDATION_EXAMPLES))
    example = VALIDATION_EXAMPLES[idx]

    gt_biomass = example["ground_truth"]
    preds_for_example = example.get("predictions", {})

    pred_model1 = preds_for_example.get(model_1_id)
    pred_model2 = preds_for_example.get(model_2_id)

    if pred_model1 is None or pred_model2 is None:
        st.warning(
            "This validation example does not have predictions for one or both "
            "selected models. Try again or adjust your validation data."
        )
        return

    error_model1 = {
        k: abs(float(pred_model1.get(k, 0.0)) - float(gt_biomass.get(k, 0.0)))
        for k in BIOMASS_KEYS
    }
    error_model2 = {
        k: abs(float(pred_model2.get(k, 0.0)) - float(gt_biomass.get(k, 0.0)))
        for k in BIOMASS_KEYS
    }

    # Update state
    state.update(
        {
            "example_index": idx,
            "gt_biomass": gt_biomass,
            "pred_model1": pred_model1,
            "pred_model2": pred_model2,
            "error_model1": error_model1,
            "error_model2": error_model2,
            "focus": state.get("focus", "Green"),
        }
    )

    try:
        st.toast("Comparison ready!", icon="📊")
    except Exception:  # noqa: BLE001
        st.success("Comparison ready!")


# -----------------------------------------------------------------------------
# Comparison result
# -----------------------------------------------------------------------------
def _render_comparison_result(state: Dict[str, Any]) -> None:
    idx = state.get("example_index")
    if idx is None or state.get("gt_biomass") is None:
        st.info(
            "No comparison yet. Select two models above and click "
            "**Compare on random image** to see an example."
        )
        return

    if idx < 0 or idx >= len(VALIDATION_EXAMPLES):
        st.warning("Selected validation example index is out of range.")
        return

    example = VALIDATION_EXAMPLES[idx]
    gt_biomass = state["gt_biomass"]
    pred_model1 = state["pred_model1"]
    pred_model2 = state["pred_model2"]

    model_1_id = state["model_1_id"]
    model_2_id = state["model_2_id"]

    st.subheader(f"Validation example: {example.get('id', f'#{idx}')}")
    st.caption(
        "Example from the validation set. Ground-truth biomass values come "
        "from real field measurements."
    )

    # Image display
    img_path = example.get("image_path")
    if img_path:
        st.image(img_path, use_container_width=True)
    else:
        st.warning("No image_path provided for this validation example.")

    st.markdown("### Biomass predictions vs ground truth")

    _render_comparison_bar_chart(
        gt_biomass=gt_biomass,
        pred_model1=pred_model1,
        pred_model2=pred_model2,
        model_1_name=MODEL_METADATA[model_1_id]["display_name"],
        model_2_name=MODEL_METADATA[model_2_id]["display_name"],
    )

    _render_error_summary(state)

    st.markdown("---")
    _render_explainability_section(state, example)


def _render_comparison_bar_chart(
    gt_biomass: Dict[str, float],
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
                "Source": "Ground truth",
                "Value": float(gt_biomass.get(key, 0.0)),
            }
        )
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


def _render_error_summary(state: Dict[str, Any]) -> None:
    error_model1 = state["error_model1"]
    error_model2 = state["error_model2"]
    model_1_id = state["model_1_id"]
    model_2_id = state["model_2_id"]

    model_1_name = MODEL_METADATA[model_1_id]["display_name"]
    model_2_name = MODEL_METADATA[model_2_id]["display_name"]

    mae1 = float(np.mean(list(error_model1.values())))
    mae2 = float(np.mean(list(error_model2.values())))

    st.markdown("### Error summary")
    st.markdown(
        f"- Average absolute error for **{model_1_name}**: `{mae1:.3f}` g\n"
        f"- Average absolute error for **{model_2_name}**: `{mae2:.3f}` g"
    )

    # Optionally highlight which is better on this example
    if mae1 < mae2:
        better = model_1_name
    elif mae2 < mae1:
        better = model_2_name
    else:
        better = None

    if better:
        st.info(
            f"On this example, **{better}** is closer to ground truth on average."
        )


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
def _render_explainability_section(state: Dict[str, Any], example: Dict[str, Any]) -> None:
    st.subheader("Explainability: where do models look?")

    col_focus, _ = st.columns([1, 1])
    with col_focus:
        focus = st.radio(
            "Explainability focus",
            options=FOCUS_OPTIONS,
            key="edu_focus_radio",
            horizontal=True,
        )
    state["focus"] = focus

    model_1_id = state["model_1_id"]
    model_2_id = state["model_2_id"]
    model_1_name = MODEL_METADATA[model_1_id]["display_name"]
    model_2_name = MODEL_METADATA[model_2_id]["display_name"]

    expl = example.get("explainability", {})
    maps_model1 = expl.get(model_1_id, {})
    maps_model2 = expl.get(model_2_id, {})

    map1_path = maps_model1.get(focus)
    map2_path = maps_model2.get(focus)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Model 1: {model_1_name}**")
        if map1_path:
            st.image(map1_path, use_container_width=True)
        else:
            st.warning(
                f"No explainability map found for {model_1_name} ({focus})."
            )

    with col2:
        st.markdown(f"**Model 2: {model_2_name}**")
        if map2_path:
            st.image(map2_path, use_container_width=True)
        else:
            st.warning(
                f"No explainability map found for {model_2_name} ({focus})."
            )

    st.caption(
        "The colored regions highlight areas of the image that contributed most "
        "to the selected biomass prediction. These are Grad-CAM style "
        "visualizations and are meant for educational purposes."
    )
