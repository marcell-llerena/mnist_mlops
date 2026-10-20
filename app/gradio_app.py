"""
Paso 5: INTERFAZ WEB con Gradio

Dos pestañas:
  ✏️ Dibujar → un lienzo (Sketchpad) para dibujar un dígito con el mouse o el dedo
  🖼️ Subir   → subir una imagen (o usar los ejemplos de examples/)

La interfaz NO carga el modelo: le envía la imagen a la API y muestra
lo que responde. Así la interfaz y el modelo se pueden actualizar por separado.

Antes, deja la API corriendo en otra terminal:
    uv run fastapi dev src/mnist_mlops/api.py

Ejecutar:  uv run app/gradio_app.py
Se abre en: http://127.0.0.1:7860
"""

import io
import os
from pathlib import Path

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def enviar_a_la_api(imagen):
    """Envía la imagen a la API (como archivo PNG) y devuelve las probabilidades."""
    if imagen is None:
        raise gr.Error("Primero dibuja o sube un dígito")

    # Guardamos la imagen como PNG "en memoria" (sin crear un archivo en disco)
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    archivo = {"file": ("digito.png", buffer.getvalue(), "image/png")}

    try:
        response = requests.post(f"{API_URL}/predict", files=archivo, timeout=10)
    except requests.RequestException:  # API apagada, colgada o inalcanzable
        raise gr.Error(
            "No se pudo conectar con la API. ¿La iniciaste con "
            "`uv run fastapi dev src/mnist_mlops/api.py`?"
        )

    if not response.ok:
        raise gr.Error(f"La API respondió con un error: {response.json().get('detail')}")

    # gr.Label recibe un diccionario {dígito: probabilidad} y dibuja las barras
    return response.json()["probabilities"]


def clasificar_dibujo(dibujo):
    # El Sketchpad entrega un diccionario con capas; "composite" es el dibujo completo
    return enviar_a_la_api(dibujo["composite"] if dibujo else None)


dibujar = gr.Interface(
    fn=clasificar_dibujo,
    inputs=gr.Sketchpad(
        type="pil",
        label="Dibuja un dígito (0-9)",
        canvas_size=(400, 400),
        # Pincel grueso y negro, como un plumón: se parece más a los trazos de MNIST
        brush=gr.Brush(default_size=24, colors=["#000000"], color_mode="fixed"),
    ),
    outputs=gr.Label(num_top_classes=3, label="Dígito predicho"),
    description="Dibuja un dígito grande y centrado, y presiona **Submit**. "
    "Para borrar, usa el ícono de la papelera.",
    flagging_mode="never",
)

subir = gr.Interface(
    fn=enviar_a_la_api,
    inputs=gr.Image(type="pil", label="Imagen con un dígito", sources=["upload", "clipboard"]),
    outputs=gr.Label(num_top_classes=3, label="Dígito predicho"),
    # Ejemplos listos para probar: una imagen real de MNIST de cada dígito
    examples=sorted(str(path) for path in EXAMPLES_DIR.glob("digito_*.png")),
    description="Sube una foto o imagen de un dígito, o elige un ejemplo de MNIST.",
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [dibujar, subir],
    tab_names=["✏️ Dibujar", "🖼️ Subir imagen"],
    title="✍️ Reconocedor de dígitos MNIST",
)

if __name__ == "__main__":
    demo.launch()
