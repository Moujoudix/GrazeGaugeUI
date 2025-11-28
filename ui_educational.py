from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import streamlit as st
import altair as alt
from PIL import Image
from api_client import call_compare_api
from config import (
    BIOMASS_KEYS,
    BIOMASS_DISPLAY,
    BIOMASS_COLORS,
    MODEL_METADATA,
    MODEL_ORDER,
    FOCUS_OPTIONS,
    FOCUS_TO_LABEL,
    CORE_BIOMASS_KEYS,
)


# Relative thresholds (tune as you want)
CLOSE_REL = 0.05   # both models considered "close" if <= 10% relative error
FAR_REL   = 0.2   # considered "far" if >= 50% relative error
DIFF_REL  = 0.10   # difference in relative error considered "large" if >= 10 points


@st.cache_data
def load_ground_truth_table() -> pd.DataFrame:
    """
    Load the wide CSV with ground truth biomass values.

    Expected columns:
      - 'image_id' (matching uploaded filename without extension)
      - one column per BIOMASS_KEYS
    """
    df = pd.read_csv("data/wide.csv")
    return df


def get_ground_truth_row_for_filename(filename: str) -> Optional[pd.Series]:
    """
    Given an uploaded filename like 'ID1001187975.jpg',
    return the matching row from the GT table as a pandas Series.
    """
    df = load_ground_truth_table()

    if "image_id" not in df.columns:
        return None

    mask = df["image_id"].astype(str) == str(filename)
    matches = df.loc[mask]

    if matches.empty:
        return None

    return matches.iloc[0]

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
            st.image(uploaded, width='stretch')

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

    # NEW: ground-truth lookup based on uploaded filename
    uploaded = state.get("example_image")
    gt_row = None
    if uploaded is not None:
        gt_row = get_ground_truth_row_for_filename(uploaded.name)

    st.subheader("Biomass predictions comparison")

    _render_comparison_bar_chart(
        pred_model1=m1["biomass"],
        pred_model2=m2["biomass"],
        model_1_name=model_1_name,
        model_2_name=model_2_name,
        gt_row=gt_row,  # <-- pass GT row (Series) or None
    )

    _render_summary_with_optional_gt(
    pred_model1=m1["biomass"],
    pred_model2=m2["biomass"],
    model_1_name=model_1_name,
    model_2_name=model_2_name,
    gt_row=gt_row,
    )

    st.markdown("---")
    _render_explainability_section(state, m1, m2, model_1_name, model_2_name)



def _render_comparison_bar_chart(
    pred_model1: Dict[str, float],
    pred_model2: Dict[str, float],
    model_1_name: str,
    model_2_name: str,
    gt_row: Optional[pd.Series] = None,
) -> None:
    """
    Grouped bar chart: one group per biomass type, 2 or 3 bars per group:
      - Ground truth (if gt_row is not None)
      - Model 1
      - Model 2

    Colors:
      - Ground truth: green
      - Model 1: dark blue
      - Model 2: light blue

    Shows value labels on top of each bar.
    """
    df_parts: List[pd.DataFrame] = []

    sources_order: List[str] = []
    colors: List[str] = []

    # Ground truth part (from wide.csv row)
    if gt_row is not None:
        df_gt = pd.DataFrame(
            {
                "Biomass": [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS],
                "Source": ["Ground truth"] * len(BIOMASS_KEYS),
                "Value": [float(gt_row[k]) for k in BIOMASS_KEYS],
            }
        )
        df_parts.append(df_gt)
        sources_order.append("Ground truth")
        colors.append("#4CAF50")  # green

    # Model 1 part
    df_m1 = pd.DataFrame(
        {
            "Biomass": [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS],
            "Source": [model_1_name] * len(BIOMASS_KEYS),
            "Value": [float(pred_model1.get(k, 0.0)) for k in BIOMASS_KEYS],
        }
    )
    df_parts.append(df_m1)
    sources_order.append(model_1_name)
    colors.append("#1565C0")  # dark blue

    # Model 2 part
    df_m2 = pd.DataFrame(
        {
            "Biomass": [BIOMASS_DISPLAY[k] for k in BIOMASS_KEYS],
            "Source": [model_2_name] * len(BIOMASS_KEYS),
            "Value": [float(pred_model2.get(k, 0.0)) for k in BIOMASS_KEYS],
        }
    )
    df_parts.append(df_m2)
    sources_order.append(model_2_name)
    colors.append("#90CAF9")  # light blue

    df = pd.concat(df_parts, ignore_index=True)

    base_encodings = dict(
        x=alt.X("Biomass:N", title="Biomass type"),
        xOffset="Source:N",
        y=alt.Y("Value:Q", title="Value (g)"),
    )

    # Bars
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            **base_encodings,
            color=alt.Color(
                "Source:N",
                title="",
                scale=alt.Scale(
                    domain=sources_order,
                    range=colors,
                ),
            ),
        )
    )

    # Value labels on top of bars
    labels = (
        alt.Chart(df)
        .mark_text(
            dy=-5,          # move text slightly above bar
            size=11,
        )
        .encode(
            **base_encodings,
            text=alt.Text("Value:Q", format=".2f"),
            color=alt.value("black"),
        )
    )

    chart = (bars + labels).properties(height=350)

    st.altair_chart(chart, width='stretch')


def _render_summary_with_optional_gt(
    pred_model1: Dict[str, float],
    pred_model2: Dict[str, float],
    model_1_name: str,
    model_2_name: str,
    gt_row: Optional[pd.Series] = None,
) -> None:
    if gt_row is not None:
        st.markdown("### Summary vs ground truth for core components")

        col_gt = "Ground truth (g)"
        col1 = f"{model_1_name} error"
        col2 = f"{model_2_name} error"

        rows = []
        for key in CORE_BIOMASS_KEYS:
            label = BIOMASS_DISPLAY[key]
            gt_val = float(gt_row[key])
            p1 = float(pred_model1.get(key, 0.0))
            p2 = float(pred_model2.get(key, 0.0))

            err1 = gt_val - p1  # signed error
            err2 = gt_val - p2  # signed error

            rows.append(
                {
                    "Biomass": label,
                    col_gt: gt_val,
                    col1: err1,
                    col2: err2,
                }
            )

        df_core = pd.DataFrame(rows)

        # ----- body cell styling (relative error) stays the same -----
        def _style_row(row: pd.Series) -> List[str]:
            styles = [""] * len(row)

            gt_val = float(row[col_gt])
            denom = max(abs(gt_val), 1.0)

            e1 = float(row[col1])
            e2 = float(row[col2])
            rel1 = abs(e1) / denom
            rel2 = abs(e2) / denom

            both_close = rel1 <= CLOSE_REL and rel2 <= CLOSE_REL
            diff_big = abs(rel1 - rel2) >= DIFF_REL
            both_far = rel1 >= FAR_REL and rel2 >= FAR_REL
            max_rel = max(rel1, rel2)

            idx_gt = row.index.get_loc(col_gt)
            idx1 = row.index.get_loc(col1)
            idx2 = row.index.get_loc(col2)

            GT_BG = "background-color: #F5F5F5;"
            GOOD_BG = "background-color: #C8E6C9;"
            BAD_BG = "background-color: #FFCCBC;"

            styles[idx_gt] = GT_BG

            if both_close:
                styles[idx1] = GOOD_BG
                styles[idx2] = GOOD_BG
            else:
                if diff_big:
                    if rel1 < rel2:
                        styles[idx1] = GOOD_BG
                        if rel2 >= CLOSE_REL:
                            styles[idx2] = BAD_BG
                    else:
                        styles[idx2] = GOOD_BG
                        if rel1 >= CLOSE_REL:
                            styles[idx1] = BAD_BG
                else:
                    if both_far or max_rel >= FAR_REL:
                        styles[idx1] = BAD_BG
                        styles[idx2] = BAD_BG

            return styles

        styled = df_core.style.apply(_style_row, axis=1).format(
            {
                col_gt: "{:.2f}",
                col1: "{:+.2f}",
                col2: "{:+.2f}",
            }
        )

        # ----- header colors (works with st.table, not st.dataframe) -----
        # columns: [Biomass, Ground truth, model1, model2]
        # index column is first <th>, so headers are nth-child(2..4)
        styled = styled.set_table_styles(
            [
                # GT header
                {
                    "selector": "th.col_heading.level0:nth-child(3)",
                    "props": [("background-color", "#4CAF50"), ("color", "white")],
                },
                # Model 1 header
                {
                    "selector": "th.col_heading.level0:nth-child(4)",
                    "props": [("background-color", "#1565C0"), ("color", "white")],
                },
                # Model 2 header
                {
                    "selector": "th.col_heading.level0:nth-child(5)",
                    "props": [("background-color", "#90CAF9"), ("color", "black")],
                },
            ],
            overwrite=False,
        )

        # IMPORTANT: use st.table, not st.dataframe
        st.table(styled)

        st.caption(
            "Signed error = Ground truth − prediction (grams). "
            "Cell background uses **relative error** per component (close vs far), "
            "while header colors match the bar chart legend."
        )

    else:
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
    st.image(rgb_u8, width='stretch')


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

    st.image(overlay_u8, width='stretch')
