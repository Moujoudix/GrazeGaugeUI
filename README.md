# GreenVision – Pasture Biomass Viewer (Streamlit UI)

Interactive Streamlit front-end for exploring pasture biomass predictions from a FastAPI inference service.

The app allows you to:

* **Predict biomass for a single image** (per-image report, uncertainty and visualisations)
* **Run batch predictions** over multiple images and see aggregate summaries
* **Compare two different models** on the *same* image (predictions + Grad-CAM overlays)
* Browse a small **educational gallery** of example pasture images

> ⚠️ This repository contains **only the Streamlit UI**.
> It expects a compatible **FastAPI backend** exposing `/models`, `/predict` and `/compare` endpoints (see [Backend API](#backend-api) below).

---

## Table of contents

* [Demo](#demo)
* [Project structure](#project-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the app](#running-the-app)
* [Usage](#usage)

  * [Predict page](#predict-page)
  * [Batch summary](#batch-summary)
  * [Educational Lab (model comparison)](#educational-lab-model-comparison)
  * [About & Credits](#about--credits)
* [Backend API](#backend-api)
* [Data folder](#data-folder)
* [Development notes](#development-notes)

  * [Code layout](#code-layout)
  * [Tests](#tests)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Demo

The UI is organised into three tabs:

1. **Predict**

   * Upload one or more pasture images.
   * Choose a model.
   * Get per-image biomass estimates (Dry Green, Dry Clover, Dry Dead, GDM, Dry Total) plus uncertainties.
   * Visualisations: bar charts, radar/shape overview, grazing-readiness score.

2. **Educational Lab**

   * Select **two models** to compare.
   * Use either your own image or one of the bundled **example images**.
   * See predictions vs ground truth for core biomass components.
   * Inspect **Grad-CAM overlays** for different biomass targets (Green, Clover, Dead).

3. **About & Credits**

   * Short description of the project, data and models.
   * Attribution and usage notes.

---

## Project structure

Repository layout:

```text
.
├── api_client.py           # Thin client for talking to the FastAPI backend
├── app.py                  # Streamlit entrypoint, sets up pages & routing
├── config.py               # UI configuration (API URL, paths, constants, etc.)
├── data
│   ├── edu_images          # Example images for the Educational Lab
│   │   ├── ID1052620238.jpg
│   │   ├── ID1119739385.jpg
│   │   ├── ID1119761112.jpg
│   │   ├── ID1121692672.jpg
│   │   ├── ID1163061745.jpg
│   │   ├── ID1208644039.jpg
│   │   ├── ID383231615.jpg
│   │   ├── ID415656958.jpg
│   │   └── ID969218269.jpg
│   └── wide.csv            # Ground-truth biomass table used for comparisons
├── Makefile                # Optional helper commands for development (see file)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── tests
│   └── test_structure.sh   # Basic structure/check script
├── ui_about.py             # Implementation of the “About & Credits” page
├── ui_educational.py       # Implementation of the Educational Lab page
└── ui_predict.py           # Implementation of the Predict / batch view
```

Compiled bytecode (`__pycache__/`) is not relevant for development and can be ignored.

---

## Requirements

* Python **3.12**
* A running **FastAPI backend** that exposes:

  * `GET /models`
  * `POST /predict`
  * `POST /compare`

* All Python requirements are listed in `requirements.txt`.

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone <YOUR-REPO-URL>.git
   cd <YOUR-REPO-NAME>
   ```

2. (Recommended) create and activate a **virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on Linux / macOS
   # .venv\Scripts\activate         # on Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Make sure your **FastAPI backend** is running and reachable (see [Backend API](#backend-api)).

---

## Configuration

All UI-specific configuration lives in `config.py`.
Typical things to configure there:

* **Backend URL / API base URL**
  e.g. `API_BASE_URL = "http://localhost:8000"` or read from an environment variable.

* **Paths** for local data used by the UI:

  * `DATA_DIR` (root `data` folder)
  * `EDU_IMAGES_DIR` (for the example grid)
  * `WIDE_CSV_PATH` (ground-truth table)

* Any additional display / feature flags (e.g. number of images in the educational grid, default focus options).

> 👉 Open `config.py` and adjust the values to match your environment before running the app.

---

## Running the app

From the project root:

```bash
streamlit run app.py
```

You can override the port if needed:

```bash
streamlit run app.py --server.port 8501
```

By default Streamlit will open the UI in your browser at:

* `http://localhost:8000` (or the port you specify)

---

## Usage

### Predict page

1. Go to the **“Predict”** tab (default when the app starts).
2. **Upload images**:

   * Drag & drop one or more pasture images into the upload area *or* click “Browse files”.
   * Supported formats: **JPG**, **JPEG**, **PNG**.
3. **Choose a model** from the dropdown on the right.

   * The list is fetched dynamically from the backend via `GET /models`.
   * Expanding “Model info” shows backbone, input size and a short description.
4. Click **“Run prediction”**.

For each image, the page displays:

* A large preview of the input image.
* A **Biomass estimates** table:

  * Dry Green (g)
  * Dry Clover (g)
  * Dry Dead (g)
  * GDM (g)
  * Dry Total (g)
  * plus **uncertainty intervals** when available.
* Visualisations:

  * A **composition bar chart** (biomass components).
  * A **shape / radar** overview of the component mix.
* A **Grazing Quality Score** (0–100) with:

  * A long progress bar.
  * A textual interpretation (e.g. “IDEAL: feed level and quality are both in the sweet spot…”).

#### Batch predictions

When multiple images are uploaded, the page also shows a **Batch summary**:

* Table of **average biomass** across all images.
* Bar chart of **average composition**.
* Overall **Grazing Quality Score** for the batch.
* A grid of **per-image prediction cards**, each with the same per-image summary as above.

---

### Educational Lab (model comparison)

The **“Educational Lab”** tab is designed for exploring how different models behave on the same image, and where they “look” when making decisions.

Workflow:

1. **Select two models**

   At the top of the page, choose “Model 1” and “Model 2” from the dropdowns.
   Labels are taken from backend metadata (e.g. `EfficientNetV2-S (baseline)`, `EfficientNetV2-S (aux-loss variant)`).

2. **Choose an image**

   You have two options:

   * **Example images**
     Scroll through the grid of example pasture images (loaded from `data/edu_images`).
     Click on any image to run the comparison.
   * **Custom image**
     Check **“Use your own image”**, upload an image file, then click **“Compare on this image”**.

3. **Inspect predictions**

   After a comparison runs (via `POST /compare`):

   * A **“Biomass predictions comparison”** bar chart shows:

     * Ground truth (when available from `data/wide.csv`)
     * Model 1 predictions
     * Model 2 predictions
   * A **summary table** lists, for core biomass components:

     * Ground truth value
     * Signed relative error for each model
       (e.g. `(Ground truth - prediction) / Ground truth`), colour-coded.

4. **Explore Grad-CAM explanations**

   Lower on the page, the **Explainability (Grad-CAM)** section lets you choose a **biomass focus**:

   * Green biomass
   * Clover biomass
   * Dead biomass

   For the selected focus, the app displays **pre-rendered Grad-CAM overlays** returned by the backend for:

   * Model 1 (left)
   * Model 2 (right)

   These overlays illustrate which regions of the image contributed most to the prediction for that biomass component.

---

### About & Credits

The **“About & Credits”** tab is a static page implemented in `ui_about.py`.
It typically contains:

* A short explanation of what the project is about.
* High-level architecture overview (Streamlit UI + FastAPI backend).
* Links to related papers / blog posts / model training repos.
* Credits and acknowledgements.

You can freely customise this content.

---

## Backend API

The UI communicates with a FastAPI service, using the endpoints exposed in its OpenAPI docs (`/docs`).

At minimum, the backend is expected to provide:

* `GET /` – health / landing endpoint (not strictly required by the UI).
* `GET /health` – optional lightweight health check.
* `GET /models` – list available models and metadata.
* `POST /predict` – run predictions on 1-N images.
* `POST /compare` – run a joint prediction + explainability pass for two models on a single image.

The corresponding **Pydantic schemas** on the backend side typically include:

* `ModelMetadata`, `ModelsListResponse`
* `Biomass`, `BiomassUncertainty`
* `Prediction`, `PredictResponse`
* `Explanation`, `HeatmapOverlay`, `ExplainedPrediction`
* `CompareResponse`

The frontend does **not** hard-code the exact schema details; instead:

* `api_client.py` is the single place where HTTP calls and payload parsing happen.
* If you change the backend contract, you only need to update `api_client.py` and, if needed, the display logic in the UI modules.

> 💡 To see the precise request/response schema, run your FastAPI server and open its `/docs` page.

---

## Data folder

The `data/` directory provides static assets used by the UI (no training data):

* `data/edu_images/` – sample images shown on the **Educational Lab** page.

  * Filenames (e.g. `ID383231615.jpg`) may correspond to rows in `wide.csv`.
* `data/wide.csv` – a “wide” table of per-image ground-truth biomass measurements.

  * Used for:

    * Looking up ground truth for the current image.
    * Computing error metrics shown in the comparison tables.

If you replace these with your own dataset:

1. Keep the **same column names** expected by the ground-truth lookup helper.
2. Keep filenames consistent between `edu_images` and `wide.csv`.

---

## Development notes

### Code layout

The UI code is split into smaller modules:

* `app.py`

  * Sets up overall Streamlit configuration and navigation.
  * Delegates rendering of each tab to the `ui_*` modules.

* `ui_predict.py`

  * All UI components for the **Predict** page.
  * Handles file uploads, calls to `/predict` and rendering of tables/plots.

* `ui_educational.py`

  * UI for the **Educational Lab**.
  * Handles dataset example grid, custom image upload, calls to `/compare`.
  * Manages Grad-CAM focus selection and display of overlays.

* `ui_about.py`

  * Static content for the **About & Credits** page.

* `api_client.py`

  * Wraps HTTP calls to the FastAPI backend.
  * Helps keep the UI logic clean and isolated from networking details.

* `config.py`

  * Central place for constants and configuration.

  Adjust values here instead of scattering magic numbers / URLs through the code.

### Tests

A simple structure check lives in `tests/test_structure.sh`.

Run it from the repo root:

```bash
bash tests/test_structure.sh
```

This script is mainly intended for automated grading / CI in teaching / competition settings.

You can add your own tests (e.g. for `api_client.py` or for data-loading helpers) under the `tests/` directory.

---

## Troubleshooting

* **I get an error about connecting to the API server**

  * Ensure your FastAPI backend is running.
  * Check the base URL configured in `config.py`.
  * Confirm you can reach `<API_BASE_URL>/health` or `/docs` in your browser.

* **The model dropdowns are empty**

  * The UI populates them from `GET /models`.
  * Verify the backend implements this endpoint and returns a non-empty list.

* **Grad-CAM images don’t appear**

  * The Educational Lab calls `POST /compare` with `method="grad_cam"`.
  * Check that this endpoint is implemented and that it returns:

    * Predictions for both models.
    * Explanation objects with an appropriate `heatmaps` dict keyed by focus name.

* **Ground truth values show as “N/A”**

  * The app looks up each image in `data/wide.csv` by filename.
  * Check that the image name exists as expected in the CSV.

---
