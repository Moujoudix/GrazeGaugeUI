# api_client.py
from typing import List, Dict, Any
import json

import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import PREDICT_API_URL, API_TIMEOUT_SECONDS


def call_predict_api(
    images: List[UploadedFile],
    model_id: str,
    aux_heads: bool,
) -> Dict[str, Any]:
    """
    Call the /predict endpoint with one or more images.

    Parameters
    ----------
    images : list of UploadedFile
        Files returned by st.file_uploader.
    model_id : str
        Selected model identifier (must match backend).
    aux_heads : bool
        Whether to use AUX heads for the selected model.

    Returns
    -------
    dict
        Parsed JSON response from the backend.

    Raises
    ------
    requests.RequestException
        If an HTTP error / timeout occurs.
    ValueError
        If response JSON is malformed.
    """
    # Build multipart form data for files
    files = []
    for img in images:
        # UploadedFile has .name and .getvalue()
        file_bytes = img.getvalue()
        mime_type = img.type or "image/jpeg"
        files.append(
            (
                "files",
                (img.name, file_bytes, mime_type),
            )
        )

    # Adapt this payload to match your backend spec
    data = {
        "model_id": model_id,
        "aux_heads": json.dumps(aux_heads),  # or "true"/"false" depending on backend
    }

    response = requests.post(
        PREDICT_API_URL,
        data=data,
        files=files,
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    try:
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse JSON from /predict: {exc}") from exc
