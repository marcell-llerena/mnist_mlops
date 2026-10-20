"""
Paso 3b: PREDICCIÓN

Funciones para cargar el modelo ONNX y hacer predicciones.
Las usan la API y los tests.

Aquí NO se usa PyTorch: el modelo se ejecuta con onnxruntime, un motor
liviano hecho solo para predecir. Por eso la imagen de producción es pequeña
y entra cómoda en los 512 MB de RAM del plan gratis de Render.

Ejecutar (ejemplo rápido):  uv run predecir
             (otra imagen):  uv run predecir ruta/a/mi_digito.png
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from mnist_mlops.config import CLASSES, EXAMPLES_DIR, METRICS_PATH, MODEL_PATH
from mnist_mlops.preprocess import preprocess


@lru_cache  # carga el modelo una sola vez y lo reutiliza (más rápido)
def load_model() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo. Primero ejecuta: uv run entrenar"
        )
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def load_model_info() -> dict:
    """Devuelve la ficha técnica del modelo (versión, fecha, accuracy...)."""
    return json.loads(METRICS_PATH.read_text())


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convierte los 10 puntajes del modelo en probabilidades que suman 1."""
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict(image: Image.Image) -> dict:
    """Recibe una imagen con un dígito y devuelve el dígito predicho."""
    session = load_model()
    (logits,) = session.run(None, {"image": preprocess(image)})
    probabilities = softmax(logits[0])

    return {
        "digit": int(probabilities.argmax()),
        "probabilities": {
            digit: round(float(p), 4) for digit, p in zip(CLASSES, probabilities)
        },
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXAMPLES_DIR / "digito_7.png"
    result = predict(Image.open(path))
    print(f"Imagen:      {path}")
    print(f"Predicción:  {result['digit']}")
    print("Top 3:")
    top = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    for digit, p in top:
        print(f"  {digit}: {p:.2%}")


if __name__ == "__main__":
    main()
