import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def glioma_image():
    return ("glioma.jpg", (DATA_DIR / "glioma.jpg").read_bytes(), "image/jpeg")


@pytest.fixture
def no_tumor_image():
    return ("no_tumor.jpg", (DATA_DIR / "no_tumor.jpg").read_bytes(), "image/jpeg")


@pytest.fixture
def blank_image():
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512)).save(buffer, format="PNG")
    return ("blank.png", buffer.getvalue(), "image/png")


@pytest.fixture
def text_file():
    return ("notes.txt", b"pas une image", "text/plain")


@pytest.fixture
def corrupted_image(glioma_image):
    name, data, content_type = glioma_image
    return ("corrupted.jpg", data[: len(data) // 2], content_type)
