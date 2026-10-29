import pytest

CLASS_NAMES = {"Glioma", "Meningioma", "No Tumor", "Pituitary"}
ENDPOINTS = ["/predict/classification", "/predict/detection"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": {"classifier": True, "detector": True}}


def test_classification(client, glioma_image):
    response = client.post("/predict/classification", files={"file": glioma_image})
    assert response.status_code == 200
    body = response.json()
    assert body["class"] == "Glioma"
    assert set(body["probabilities"]) == CLASS_NAMES
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert body["confidence"] == max(body["probabilities"].values())


def test_classification_no_tumor(client, no_tumor_image):
    response = client.post("/predict/classification", files={"file": no_tumor_image})
    assert response.status_code == 200
    assert response.json()["class"] == "No Tumor"


def test_detection(client, glioma_image):
    response = client.post("/predict/detection", files={"file": glioma_image})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["detections"]) > 0
    for detection in body["detections"]:
        assert detection["class"] in CLASS_NAMES
        assert 0 <= detection["confidence"] <= 1
        x1, y1, x2, y2 = detection["bbox"]
        assert x1 < x2 and y1 < y2
    assert "annotated_image" not in body


def test_detection_annotated(client, glioma_image):
    response = client.post("/predict/detection", params={"annotated": True}, files={"file": glioma_image})
    assert response.status_code == 200
    assert response.json()["annotated_image"]


def test_detection_empty(client, blank_image):
    response = client.post("/predict/detection", files={"file": blank_image})
    assert response.status_code == 200
    assert response.json() == {"detections": [], "count": 0}


def test_detection_invalid_conf(client, glioma_image):
    response = client.post("/predict/detection", params={"conf": 1.5}, files={"file": glioma_image})
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unsupported_file(client, text_file, endpoint):
    response = client.post(endpoint, files={"file": text_file})
    assert response.status_code == 415


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_corrupted_image(client, corrupted_image, endpoint):
    response = client.post(endpoint, files={"file": corrupted_image})
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint, model_name", [(ENDPOINTS[0], "classifier"), (ENDPOINTS[1], "detector")])
def test_model_not_loaded(client, glioma_image, monkeypatch, endpoint, model_name):
    monkeypatch.setitem(client.app.state.models, model_name, None)
    response = client.post(endpoint, files={"file": glioma_image})
    assert response.status_code == 503
    assert model_name in response.json()["detail"]


def test_internal_error(client, glioma_image, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(client.app.state.models["detector"], "predict", fail)
    response = client.post("/predict/detection", files={"file": glioma_image})
    assert response.status_code == 500
    assert response.json() == {"detail": "Erreur interne du serveur."}
