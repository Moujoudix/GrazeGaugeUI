# ui_about.py
import streamlit as st
from config import APP_TITLE


def render_about_page() -> None:
    st.subheader("Project overview")
    st.markdown(
        f"""
**{APP_TITLE}** is a computer-vision prototype that estimates pasture biomass from simple top-view pasture photos.

Instead of relying only on manual “clip-and-weigh” sampling, users can upload images and obtain approximate estimates of:

- green biomass  
- dead biomass  
- clover biomass  
- gross dry matter (GDM)  
- total biomass  

The models are trained on the public **CSIRO Image2Biomass** pasture dataset, using labelled pasture images to learn how visual patterns relate to biomass measurements. The goal is to explore whether deep learning can provide fast, non-destructive estimates that are “good enough” to compare paddocks, understand pasture condition, and illustrate how AI can support grazing decisions.

This app was developed as a capstone project during the Le Wagon Data Science bootcamp. It is a research and educational prototype: estimates can be noisy and should not replace expert agronomic advice or on-farm measurements, but they showcase how modern computer vision and MLOps can be turned into a usable tool.
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

- PyTorch + `timm` convolutional backbones (e.g. ConvNeXt, EfficientNetV2, RegNet, DenseNet).
- A shared regression head that predicts the five biomass components from image features.
- Data augmentation for robustness (colour jitter, flips, random crops, etc.).
- Training and experimentation carried out in notebooks, iterating over several model families and configurations.
"""
    )

    st.markdown(
        """
**Serving / backend**

- FastAPI backend exposing a `/predict` endpoint for image-based biomass estimation.
- Dockerised service that bundles the trained model weights, preprocessing, and inference code.
- Deployed as a stateless HTTP service on **Google Cloud Run**.
"""
    )

    st.markdown(
        """
**Frontend**

- Streamlit app (this interface) for uploading one or more pasture images.
- The app calls the Cloud Run `/predict` endpoint and displays the five biomass components per image using interactive charts and comparison views.
- An **Educational Lab** page that allows users to select two trained models, run them on the same image, and compare their predictions side by side with visualisations.
"""
    )

    st.markdown(
        """
When a user uploads pasture images in the Streamlit app, the frontend sends them to the FastAPI `/predict` endpoint running on Google Cloud Run. The backend loads the selected deep-learning model, applies the same preprocessing as during training, and returns biomass estimates in JSON format.

The app then visualises these estimates so users can compare components (green vs dead vs clover, etc.), and in the Educational Lab it highlights how different model families behave on the same image.
"""
    )


def _render_links_and_acknowledgements() -> None:
    st.subheader("Links & acknowledgements")

    st.markdown(
        """
**Codebase**

- The backend (models & API) and the Streamlit frontend are maintained in private repositories as part of the GrazeGauge / PastureVision project.
"""
    )

    st.markdown(
        """
**Data & references**

- Pasture imagery and biomass labels from the **CSIRO Image2Biomass** dataset.
- Additional inspiration from research on pasture biomass estimation and computer vision in agriculture.
"""
    )

    st.markdown(
        """
**Acknowledgements**

- CSIRO and partners for releasing the Image2Biomass dataset.
- The **Le Wagon Data Science** teaching team and community for guidance, feedback, and code reviews during the project.
- Friends, classmates, and early testers who tried the app and shared ideas on how it could be useful in real grazing contexts.
"""
    )
