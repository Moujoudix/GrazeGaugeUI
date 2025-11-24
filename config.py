# config.py
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# App-level config
# -----------------------------------------------------------------------------
APP_TITLE = "GrazeGauge: Estimating Pasture Biomass from Images"
APP_SUBTITLE = (
    "Upload pasture images to estimate green, clover, dead, and total biomass "
    "and explore how different models behave."
)

# Backend URL
PREDICT_API_URL = "https://grazegauge-api-415044706954.europe-west1.run.app/predict"
API_TIMEOUT_SECONDS = 30

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

# Consistent colors for all charts
BIOMASS_COLORS: Dict[str, str] = {
    "Dry_Green_g": "#4CAF50",   # green
    "Dry_Clover_g": "#2E7D32",  # dark green
    "Dry_Dead_g": "#8D6E63",    # brown-ish
    "GDM_g": "#607D8B",         # blue-grey
    "Dry_Total_g": "#212121",   # dark
}

# -----------------------------------------------------------------------------
# Model metadata
# -----------------------------------------------------------------------------
# TODO: Fill with your real models, descriptions, histories, etc.
MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "resnet_baseline": {
        "display_name": "ResNet50 baseline",
        "history": (
            "Initial baseline model fine-tuned from ImageNet ResNet50 on the "
            "pasture dataset."
        ),
        "description": (
            "Convolutional backbone with global pooling and a single regression "
            "head predicting all biomass components."
        ),
        "params": "Input 224x224, Adam optimizer, MSE loss, ~25M parameters.",
        "did_you_know": (
            "Did you know? This baseline already outperformed a naive linear "
            "regression on hand-crafted features by a large margin."
        ),
        "supports_aux": True,
    },
    "effnet_aug": {
        "display_name": "EfficientNet (augmented)",
        "history": (
            "Second-generation model using EfficientNet backbone and heavier "
            "data augmentation (color jitter, rotations, random crops)."
        ),
        "description": (
            "Lighter backbone with strong regularization; trained with "
            "augmented images to improve generalization to new paddocks."
        ),
        "params": "Input 256x256, AdamW, cosine LR schedule.",
        "did_you_know": (
            "Did you know? Adding stronger augmentations helped reduce "
            "overfitting especially on clover biomass."
        ),
        "supports_aux": True,
    },
    # Add more models here...
}

MODEL_ORDER: List[str] = list(MODEL_METADATA.keys())

# -----------------------------------------------------------------------------
# Educational Lab config
# -----------------------------------------------------------------------------
FOCUS_OPTIONS = ["Green", "Clover", "Dead"]
FOCUS_TO_BIOMASS_KEY = {
    "Green": "Dry_Green_g",
    "Clover": "Dry_Clover_g",
    "Dead": "Dry_Dead_g",
}


def load_validation_examples() -> list[dict]:
    """
    Load or construct validation examples for the Educational Lab.

    Each example should look like:
    {
        "id": "val_001",
        "image_path": "assets/val_001.jpg",
        "ground_truth": { biom_key: float, ... },
        "predictions": {
            "resnet_baseline": { biom_key: float, ... },
            "effnet_aug": { biom_key: float, ... },
        },
        "explainability": {
            "resnet_baseline": {
                "Green": "assets/expl/resnet/val_001_green.png",
                "Clover": "...",
                "Dead": "...",
            },
            "effnet_aug": {
                "Green": "assets/expl/effnet/val_001_green.png",
                ...
            },
        },
    }

    For now, we return an empty list so the app still runs;
    you can plug in your real data later.
    """
    # TODO: implement real loading from disk or a JSON file
    return []
