# ui_about.py
import streamlit as st
from config import APP_TITLE


def render_about_page() -> None:
    st.subheader("Project overview")
    st.markdown(
        """
PastureVision is a computer vision prototype that estimates pasture biomass
from simple images. Instead of manually cutting and weighing grass samples,
users can upload photos and obtain estimates of green, clover, dead, GDM and
total biomass.

This app was developed as a capstone project in a data science bootcamp to
explore how AI can support grazing management and farm decision-making.
"""
    )

    st.markdown("---")
    _render_team()
    st.markdown("---")
    _render_tech_stack()
    st.markdown("---")
    _render_links_and_acknowledgements()


def _render_team() -> None:
    st.subheader("Team")

    st.markdown(
        """
- **I. Moujoud** – project lead; led data exploration and analysis, designed and trained the initial baseline models, created the shared model backbone used across all experiments and model families, set up the project packaging and MLOps, and led the production-ready cloud deployment and polishing of the Streamlit interface (frontend & backend).
- **Kamil Laroui** – co-developer; contributed to baseline modelling, explored alternative computer-vision model families, and implemented the initial MLOps-focused cloud deployment prototype (adapting the codebase for Docker, API serving, and Streamlit integration).
- **Iz** – model evaluation and testing; focused on systematically testing model families, comparing performance, and providing feedback that guided model and UX refinements.
"""
    )


def _render_tech_stack() -> None:
    st.subheader("Tech stack & architecture")

    st.markdown(
        """
**Models & training**

- PyTorch / deep learning models (e.g. ResNet, EfficientNet, AUX heads)
- Custom training scripts & notebooks
- Data augmentation for robustness (color jitter, rotations, crops)

**Serving / backend**

- Python API (e.g. FastAPI) exposing a `/predict` endpoint
- Dockerized service
- Deployed on Google Cloud (e.g. Cloud Run / VM / GKE)

**Frontend**

- Streamlit app (this interface)
- Python client that sends uploaded images to the `/predict` endpoint
- Interactive tables, bar charts, radar charts, and explainability visualizations
"""
    )

    st.markdown(
        """
When a user uploads pasture images in the Streamlit app, the frontend sends
them to the backend `/predict` endpoint hosted on Google Cloud. The backend
loads the selected deep learning model (with or without AUX heads), runs
inference on the images, and returns biomass estimates in JSON format. The
app then visualizes these estimates and, in the Educational Lab, shows
precomputed explainability maps on validation images.
"""
    )


def _render_links_and_acknowledgements() -> None:
    st.subheader("Links & acknowledgements")

    st.markdown(
        """
**Repositories**

- Backend (models & API): `https://github.com/your-org/your-backend-repo`
- Frontend (this Streamlit app): `https://github.com/your-org/your-frontend-repo`

_Replace these links with your actual GitHub repos._
"""
    )

    st.markdown(
        """
**Data & references**

- Pasture imagery and biomass labels from [your data source here].
- Scientific papers and methods on pasture biomass estimation and deep
  learning for agriculture (list any key references if you like).

**Acknowledgements**

- Bootcamp organizers and instructors
- Any agronomists or domain experts who helped with interpretation
"""
    )
