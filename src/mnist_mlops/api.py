"""
Paso 4: SERVIR EL MODELO CON UNA API (FastAPI)

Una API permite que cualquier programa (una web, una app móvil, Gradio...)
le pida predicciones al modelo a través de internet.

A diferencia de Iris (que recibía 4 números en JSON), aquí se recibe un
ARCHIVO de imagen (PNG, JPG...), igual que cuando subes una foto a una web.

Ejecutar:  uv run fastapi dev src/mnist_mlops/api.py
Documentación interactiva:  http://127.0.0.1:8000/docs
  (en /predict → "Try it out" → elige una imagen de examples/ → "Execute")
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from mnist_mlops import __version__
from mnist_mlops.config import MAX_IMAGE_BYTES
from mnist_mlops.predict import load_model_info, predict

app = FastAPI(
    title="MNIST API",
    description="API para reconocer un dígito escrito a mano (0-9) en una imagen.",
    version=__version__,
)


# --- Esquema de salida: define qué datos devuelve la API --------------------
# (La entrada es un archivo, así que no necesita un esquema de Pydantic)
class PredictionOutput(BaseModel):
    digit: int
    probabilities: dict[str, float]


# --- Endpoints: las "puertas" de la API -------------------------------------
@app.get("/")
def health():
    """Comprueba que la API está funcionando."""
    return {"status": "ok", "message": "MNIST API funcionando"}


@app.get("/model-info")
def model_info():
    """Muestra la información del modelo que está en uso."""
    try:
        return load_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no entrenado todavía")


@app.post("/predict", response_model=PredictionOutput)
def predict_digit(file: UploadFile = File(description="Imagen con un dígito (PNG, JPG...)")):
    """Recibe una imagen con un dígito y devuelve cuál es."""
    content = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="La imagen pesa más de 5 MB")

    # Abrir la imagen. Si el archivo no es una imagen (un PDF, un texto...), error 422
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="El archivo no es una imagen válida")

    try:
        return predict(image)
    except ValueError as error:  # imagen vacía (por ejemplo, un lienzo sin dibujar)
        raise HTTPException(status_code=422, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
