# config.py
import os
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# API base URLs
# -----------------------------------------------------------------------------
# For local dev:
LOCAL_API_BASE = "http://localhost:8000"
# For deployed Cloud Run:
CLOUD_API_BASE = "https://grazegauge-api-415044706954.europe-west1.run.app"

# Choose which one to use; you can control via env var if you want
API_BASE_URL = os.getenv("GRAZEGAUGE_API_BASE", LOCAL_API_BASE)

MODELS_URL = f"{API_BASE_URL}/models"
PREDICT_URL = f"{API_BASE_URL}/predict"
COMPARE_URL = f"{API_BASE_URL}/compare"

API_TIMEOUT_SECONDS = 60  # a bit higher for Grad-CAM in /compare

# -----------------------------------------------------------------------------
# App-level config
# -----------------------------------------------------------------------------
APP_TITLE = "🍀 GrazeGauge 🍀: Estimating Pasture Biomass from Images"
APP_SUBTITLE = (
    "Upload pasture images to estimate green, clover, dead, and total biomass "
    "and explore how different models behave."
)

# -----------------------------------------------------------------------------
# Biomass configuration
# -----------------------------------------------------------------------------
BIOMASS_KEYS: List[str] = [
    "Dry_Green_g",
    "Dry_Clover_g",
    "Dry_Dead_g",
    "GDM_g",
    "Dry_Total_g",
]

BIOMASS_DISPLAY: Dict[str, str] = {
    "Dry_Green_g": "Dry Green (g)",
    "Dry_Clover_g": "Dry Clover (g)",
    "Dry_Dead_g": "Dry Dead (g)",
    "GDM_g": "GDM (g)",
    "Dry_Total_g": "Dry Total (g)",
}

BIOMASS_UNIT = "g"

BIOMASS_COLORS: Dict[str, str] = {
    "Dry_Green_g": "#319D17",
    "Dry_Clover_g": "#3EA055",
    "Dry_Dead_g": "#FFBD13",
    "GDM_g": "#0D2313",
    "Dry_Total_g": "#212121",
}

# -----------------------------------------------------------------------------
# Model metadata placeholders
# -----------------------------------------------------------------------------
# Filled from /models at runtime, not hard-coded
MODEL_METADATA: Dict[str, Dict[str, Any]] = {}
MODEL_ORDER: List[str] = []

# Educational Lab focus options
CORE_BIOMASS_KEYS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]
FOCUS_OPTIONS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]
FOCUS_TO_LABEL = {
    "Dry_Green_g": "Green biomass",
    "Dry_Clover_g": "Clover biomass",
    "Dry_Dead_g": "Dead biomass",
}

VALIDATION_IMAGES_DIR = "data/edu_images"
N_EDU_GRID_IMAGES = 9
