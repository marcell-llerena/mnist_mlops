"""Tests de la API: simulan peticiones sin necesidad de levantar el servidor."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from mnist_mlops.api import app
from mnist_mlops.config import EXAMPLES_DIR

client = TestClient(app)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "accuracy" in response.json()


def test_predict_returns_digit():
    image = (EXAMPLES_DIR / "digito_7.png").read_bytes()
    response = client.post("/predict", files={"file": ("digito_7.png", image, "image/png")})
    assert response.status_code == 200
    assert response.json()["digit"] == 7


def test_predict_rejects_non_image():
    # Un archivo de texto no es una imagen -> 422 (datos inválidos)
    response = client.post("/predict", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert response.status_code == 422


def test_predict_rejects_empty_canvas():
    blank = png_bytes(Image.new("L", (400, 400), "white"))
    response = client.post("/predict", files={"file": ("vacio.png", blank, "image/png")})
    assert response.status_code == 422


def test_predict_requires_a_file():
    response = client.post("/predict")
    assert response.status_code == 422
