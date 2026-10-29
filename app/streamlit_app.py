import base64
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODES = {"Classification": "/predict/classification", "Détection": "/predict/detection"}

st.set_page_config(page_title="Diagnostic Multimodal", page_icon="🧠", layout="wide")


def api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_api(endpoint, file, params=None):
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_URL}{endpoint}", files=files, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def show_classification(result):
    st.metric("Classe prédite", result["class"], f"Confiance : {result['confidence']:.1%}", delta_color="off")
    probabilities = pd.DataFrame(
        {"Classe": list(result["probabilities"]), "Probabilité": list(result["probabilities"].values())}
    )
    st.bar_chart(probabilities, x="Classe", y="Probabilité", horizontal=True)


def show_detection(result):
    if result["count"] == 0:
        st.info("Aucune tumeur détectée avec ce seuil de confiance.")
        return
    if all(d["class"] == "No Tumor" for d in result["detections"]):
        st.success("Aucune tumeur détectée : le modèle reconnaît un cerveau sain (No Tumor).")
    st.image(base64.b64decode(result["annotated_image"]), caption="Image annotée", width="stretch")
    st.metric("Détections", result["count"])
    detections = pd.DataFrame(
        [
            {"Classe": d["class"], "Confiance": f"{d['confidence']:.1%}", "x1": d["bbox"][0], "y1": d["bbox"][1],
             "x2": d["bbox"][2], "y2": d["bbox"][3]}
            for d in result["detections"]
        ]
    )
    st.dataframe(detections, hide_index=True, width="stretch")


with st.sidebar:
    st.header("API")
    st.caption(API_URL)
    health = api_health()
    if health is None:
        st.error("API injoignable")
    elif health["status"] == "ok":
        st.success("API disponible")
    else:
        st.warning("API dégradée")
    if health:
        for name, loaded in health["models_loaded"].items():
            st.write(f"{'✅' if loaded else '❌'} {name}")

st.title("🧠 Diagnostic Multimodal")
st.write("Analyse d'IRM cérébrales : classification du type de tumeur et détection de sa localisation.")

mode = st.radio("Mode", list(MODES), horizontal=True)
conf = st.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05) if mode == "Détection" else None
uploaded = st.file_uploader("Image IRM", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Chargez une image IRM (JPEG ou PNG) pour commencer.")
    st.stop()

preview, results = st.columns(2)
with preview:
    st.subheader("Image")
    st.image(Image.open(uploaded), width="stretch")

with results:
    st.subheader(mode)
    if st.button("Analyser", type="primary", width="stretch"):
        params = {"conf": conf, "annotated": True} if mode == "Détection" else None
        with st.spinner("Analyse en cours..."):
            result = call_api(MODES[mode], uploaded, params)
        if mode == "Classification":
            show_classification(result)
        else:
            show_detection(result)
