"""Tests del modelo: comprueban que aprende bien y predice algo con sentido."""

import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps

from mnist_mlops.config import CLASSES, EXAMPLES_DIR, METRICS_PATH, MIN_ACCURACY
from mnist_mlops.data import load_data, make_loaders
from mnist_mlops.model import DigitCNN
from mnist_mlops.predict import predict
from mnist_mlops.preprocess import preprocess
from mnist_mlops.train import evaluate_onnx


def test_network_outputs_ten_scores():
    images = torch.zeros(5, 1, 28, 28)  # 5 imágenes vacías
    assert DigitCNN()(images).shape == (5, 10)


def test_onnx_model_reaches_minimum_accuracy():
    # Se evalúa el archivo ONNX guardado: el mismo que usa la API
    _, test_loader = make_loaders(*load_data())
    metrics = evaluate_onnx(test_loader)
    assert metrics["accuracy"] >= MIN_ACCURACY


def test_metrics_file_is_complete():
    card = json.loads(METRICS_PATH.read_text())
    assert card["accuracy"] >= MIN_ACCURACY
    assert card["classes"] == CLASSES


def test_predict_mnist_example():
    result = predict(Image.open(EXAMPLES_DIR / "digito_7.png"))
    assert result["digit"] == 7


def test_predict_drawing_black_on_white():
    # Como en la app: trazo negro sobre fondo blanco, grande y fuera del centro
    seven = ImageOps.invert(Image.open(EXAMPLES_DIR / "digito_7.png")).resize((300, 300))
    canvas = Image.new("L", (800, 800), "white")
    canvas.paste(seven, (450, 50))
    assert predict(canvas)["digit"] == 7


def test_probabilities_sum_to_one():
    result = predict(Image.open(EXAMPLES_DIR / "digito_0.png"))
    assert set(result["probabilities"]) == set(CLASSES)
    assert abs(sum(result["probabilities"].values()) - 1) < 0.01


def test_preprocess_gives_white_digit_on_black_28x28():
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))  # lienzo transparente
    ImageDraw.Draw(canvas).line([(200, 50), (200, 350)], fill="black", width=30)
    pixels = preprocess(canvas)
    assert pixels.shape == (1, 1, 28, 28)
    assert pixels.dtype == np.float32
    assert pixels[0, 0, 0, 0] == 0  # esquina: fondo negro
    assert pixels[0, 0, 14, 14] > 0.5  # centro: trazo blanco


def test_preprocess_rejects_empty_image():
    with pytest.raises(ValueError):
        preprocess(Image.new("L", (400, 400), "white"))
