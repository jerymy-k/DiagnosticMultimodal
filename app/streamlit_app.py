import os

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
    st.button("Analyser", type="primary", width="stretch")
