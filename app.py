import os
import requests
from dotenv import load_dotenv

import streamlit as st
from PIL import Image
import pandas as pd

# ============================================================
# 0) CONFIG / ENV
# ============================================================
load_dotenv()
API_URL = os.getenv("GRAZEGAUGE_API_URL", "").rstrip("/")

# ============================================================
# 1) HELPER: Pretty rendering of predictions (nested JSON aware)
# ============================================================
def render_predictions(pred: dict):
    """
    The API returns something like:
    {
      "predictions": [
        {
          "filename": "...",
          "biomass": { "Dry_Green_g": ..., ... }
        }
      ]
    }

    So we:
    1) find the biomass dict
    2) display KPI cards + bar chart
    3) keep raw JSON hidden for debugging
    """

    # ----------------------------
    # 1) Locate the biomass numbers
    # ----------------------------
    biomass = None

    # Case A: nested format (your current API)
    if isinstance(pred, dict) and "predictions" in pred:
        preds = pred.get("predictions", [])
        if isinstance(preds, list) and len(preds) > 0:
            first = preds[0]
            if isinstance(first, dict) and "biomass" in first:
                biomass = first["biomass"]

    # Case B: maybe flat dict (fallback)
    if biomass is None and isinstance(pred, dict):
        # try using pred directly if it's numeric
        biomass = pred

    # If still nothing usable
    if not isinstance(biomass, dict):
        st.warning("Could not find biomass values in API response.")
        st.json(pred)
        return

    # ----------------------------
    # 2) Keep only numeric fields
    # ----------------------------
    numeric = {
        k: v for k, v in biomass.items()
        if isinstance(v, (int, float))
    }

    if not numeric:
        st.warning("No numeric predictions found to display.")
        st.json(pred)
        return

    # ----------------------------
    # 3) Fixed order + friendly labels (optional but nicer)
    # ----------------------------
    order = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]

    # Keep the order if keys exist, otherwise append leftovers
    ordered_items = [(k, numeric[k]) for k in order if k in numeric]
    leftovers = [(k, v) for k, v in numeric.items() if k not in order]
    items = ordered_items + leftovers

    s = pd.Series(dict(items))

    # ----------------------------
    # 4) KPI cards
    # ----------------------------
    st.subheader("🌾 Biomass estimates")
    cols = st.columns(len(s))

    for col, (label, value) in zip(cols, s.items()):
        col.metric(label=label, value=f"{value:,.3f}")

    # ----------------------------
    # 5) Bar chart
    # ----------------------------
    st.subheader("Comparison view")
    st.bar_chart(s)

    # ----------------------------
    # 6) Raw JSON expander
    # ----------------------------
    with st.expander("See raw JSON"):
        st.json(pred)

# ============================================================
# 2) STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="GrazeGauge",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 GrazeGauge UI")
st.write("Upload a pasture photo and get biomass estimates from the GrazeGauge API.")

# ============================================================
# 3) API HEALTH CHECK (cold-start friendly retry)
# ============================================================
with st.expander("API status", expanded=True):
    if not API_URL:
        st.error("Missing GRAZEGAUGE_API_URL in .env")
    else:
        max_tries = 3
        for attempt in range(1, max_tries + 1):
            try:
                r = requests.get(API_URL, timeout=45)
                if r.ok:
                    st.success(f"API is up ✅: {r.json()}")
                else:
                    st.warning(f"API responded with status {r.status_code}")
                break
            except Exception as e:
                if attempt == max_tries:
                    st.error(f"API not reachable after {max_tries} tries: {e}")
                else:
                    st.info(f"API waking up… retry {attempt}/{max_tries}")

# ============================================================
# 4) PREDICTION UI: Upload image → POST to /predict
# ============================================================
st.divider()
st.header("📸 Predict biomass")

uploaded = st.file_uploader(
    "Upload a pasture image (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict biomass 🚀"):

        with st.spinner("Calling GrazeGauge model..."):
            try:
                image_bytes = uploaded.getvalue()

                # API expects "files"
                files_payload = {
                    "files": (uploaded.name, image_bytes, uploaded.type)
                }

                resp = requests.post(
                    f"{API_URL}/predict",
                    files=files_payload,
                    timeout=120
                )

                if resp.ok:
                    pred = resp.json()
                    st.success("Prediction received ✅")

                    # Pretty dashboard
                    render_predictions(pred)

                else:
                    st.error(f"API error {resp.status_code}")
                    st.code(resp.text)

            except Exception as e:
                st.error(f"Request failed: {e}")
