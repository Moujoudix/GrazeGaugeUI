# api_client.py
from typing import List, Dict, Any
import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import MODELS_URL, PREDICT_URL, COMPARE_URL, API_TIMEOUT_SECONDS


# -----------------------------------------------------------------------------
# 1. Fetch models metadata
# -----------------------------------------------------------------------------
def fetch_models() -> Dict[str, Dict[str, Any]]:
    """
    Call GET /models and return a dict keyed by model name.

    Returns:
        {
          "effnetv2_s_baseline": { ... },
          "convnext_tiny_aug": { ... },
          ...
        }
    """
    resp = requests.get(MODELS_URL, timeout=API_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    models_list = data.get("models", [])
    models_dict: Dict[str, Dict[str, Any]] = {}

    for m in models_list:
        name = m["name"]
        models_dict[name] = m

    return models_dict


# -----------------------------------------------------------------------------
# 2. Predictions for Predict tab (POST /predict)
# -----------------------------------------------------------------------------
def call_predict_api(
    images: List[UploadedFile],
    model_name: str | None = None,
) -> Dict[str, Any]:
    """
    Call POST /predict with one or more images and an optional model_name.

    images: list of Streamlit UploadedFile
    model_name: optional model ID (from /models.name)

    Returns backend JSON, e.g.:
    {
      "predictions": [
        {
          "filename": "...",
          "model_name": "...",
          "biomass": { ... }
        },
        ...
      ]
    }
    """
    if not images:
        raise ValueError("No images provided to call_predict_api")

    files = []
    for img in images:
        file_bytes = img.getvalue()
        mime_type = img.type or "image/jpeg"
        files.append(("files", (img.name, file_bytes, mime_type)))

    params = {}
    if model_name:
        params["model_name"] = model_name

    resp = requests.post(
        PREDICT_URL,
        params=params,
        files=files,
        timeout=API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


# -----------------------------------------------------------------------------
# 3. Comparison + Grad-CAM for Educational Lab (POST /compare)
# -----------------------------------------------------------------------------
def call_compare_api(
    image: UploadedFile,
    model_1: str,
    model_2: str,
    method: str = "grad_cam",
) -> Dict[str, Any]:
    """
    Call POST /compare with a single image and two model names.

    Returns backend JSON, e.g.:

    {
      "models": [
        {
          "model_name": "effnetv2_s_baseline",
          "biomass": {...},
          "explanation": {...}
        },
        {...}
      ]
    }
    """
    file_bytes = image.getvalue()
    mime_type = image.type or "image/jpeg"

    files = [
        ("file", (image.name, file_bytes, mime_type)),
    ]
    data = {
        "model_1": model_1,
        "model_2": model_2,
        "method": method,
    }

    resp = requests.post(
        COMPARE_URL,
        data=data,
        files=files,
        timeout=API_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()
