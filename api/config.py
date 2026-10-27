import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))

CLASSIFIER_PATH = MODELS_DIR / "classifier.keras"
CLASSIFIER_META_PATH = MODELS_DIR / "classifier_meta.json"
DETECTOR_PATH = MODELS_DIR / "detector.pt"
DETECTOR_META_PATH = MODELS_DIR / "detector_meta.json"
