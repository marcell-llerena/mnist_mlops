"""
Paso 2: ENTRENAMIENTO, EXPORTACIÓN Y EVALUACIÓN

- Carga los datos.
- Entrena la red neuronal (DigitCNN) con PyTorch.
- La exporta a ONNX: un formato estándar que se puede ejecutar SIN PyTorch.
- Evalúa el archivo ONNX (el mismo que irá a producción) con imágenes que nunca vio.
- Si es suficientemente bueno, guarda el modelo y sus métricas en models/.

¿Dónde entrena? Elige solo el dispositivo más rápido disponible:
  - Mac con chip Apple (M1, M2...) → "mps" (la GPU de Apple)
  - PC con GPU NVIDIA              → "cuda"
  - Docker, CI, Render             → "cpu" (más lento, pero con este modelo basta)

Ejecutar:  uv run entrenar
Forzar la CPU (para comparar tiempos):  DEVICE=cpu uv run entrenar
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from mnist_mlops import __version__
from mnist_mlops.config import (
    CLASSES,
    IMAGE_SIZE,
    METRICS_PATH,
    MIN_ACCURACY,
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_PARAMS,
)
from mnist_mlops.data import load_data, make_loaders
from mnist_mlops.model import DigitCNN


def get_device() -> str:
    """Devuelve el dispositivo más rápido disponible para entrenar."""
    if os.getenv("DEVICE"):  # si se indica a mano, se respeta
        return os.environ["DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(train_loader, device: str) -> DigitCNN:
    """Crea la red y la entrena durante TRAIN_PARAMS['epochs'] épocas."""
    torch.manual_seed(RANDOM_STATE)
    model = DigitCNN().to(device)  # .to(device) = mover el modelo a la GPU (o dejarlo en CPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()  # mide qué tan equivocada está cada predicción

    model.train()  # modo entrenamiento (activa el Dropout)
    for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 1. borrar las correcciones del lote anterior
            loss = loss_fn(model(images), labels)  # 2. predecir y medir el error
            loss.backward()  # 3. calcular cómo corregir cada parámetro
            optimizer.step()  # 4. aplicar la corrección
            total_loss += loss.item()
        print(f"   Época {epoch}: error promedio = {total_loss / len(train_loader):.4f}")

    return model


def export_onnx(model: DigitCNN) -> None:
    """Exporta el modelo entrenado a models/model.onnx."""
    model = model.to("cpu").eval()  # ONNX se exporta desde la CPU, en modo evaluación
    example = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)  # una imagen "de muestra" con la forma de entrada
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        MODEL_PATH,
        input_names=["image"],
        output_names=["logits"],
        # La primera dimensión (cuántas imágenes) queda libre: 1 desde la API,
        # 1000 de golpe al evaluar. "x" es el nombre del argumento de forward().
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        external_data=False,  # todo en un solo archivo .onnx
        verbose=False,
    )


def evaluate_onnx(test_loader) -> dict:
    """Mide el accuracy del archivo ONNX con onnxruntime: el mismo motor que usa la API."""
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    correct = 0
    per_class = np.zeros((len(CLASSES), 2))  # por dígito: [aciertos, total]
    for images, labels in test_loader:
        (logits,) = session.run(None, {"image": images.numpy()})
        predictions = logits.argmax(axis=1)
        labels = labels.numpy()
        correct += int((predictions == labels).sum())
        np.add.at(per_class, labels, np.stack([predictions == labels, np.ones_like(labels)], axis=1))

    return {
        "accuracy": round(correct / len(test_loader.dataset), 4),
        "accuracy_per_digit": {
            digit: round(hits / total, 4) for digit, (hits, total) in zip(CLASSES, per_class)
        },
    }


def save_metrics(metrics: dict, device: str, seconds: float) -> None:
    """Guarda un archivo JSON con la 'ficha técnica' del modelo."""
    model_card = {
        "model_version": __version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": DigitCNN.__name__,
        "format": "onnx",
        "trained_on": device,
        "training_seconds": round(seconds, 1),
        "params": TRAIN_PARAMS,
        "n_parameters": sum(p.numel() for p in DigitCNN().parameters()),
        "input": f"imagen 1x{IMAGE_SIZE}x{IMAGE_SIZE} en escala de grises (0 a 1)",
        "classes": CLASSES,
        **metrics,
    }
    METRICS_PATH.write_text(json.dumps(model_card, indent=2, ensure_ascii=False))


def main() -> None:
    print("1) Cargando datos...")
    train, test = load_data()
    train_loader, test_loader = make_loaders(train, test)
    print(f"   Entrenamiento: {len(train)} imágenes | Prueba: {len(test)} imágenes")

    device = get_device()
    print(f"2) Entrenando en '{device}'...")
    start = time.perf_counter()
    model = train_model(train_loader, device)
    seconds = time.perf_counter() - start
    print(f"   Tiempo: {seconds:.0f} s")

    print("3) Exportando a ONNX...")
    export_onnx(model)

    print("4) Evaluando el modelo ONNX...")
    metrics = evaluate_onnx(test_loader)
    print(f"   Accuracy: {metrics['accuracy']:.2%}")

    # "Puerta de calidad": solo se publica el modelo si cumple el mínimo.
    # Se evalúa el ONNX (no el modelo de PyTorch) porque es el archivo que
    # realmente llega a producción: si algo se rompió al exportar, aquí se nota.
    if metrics["accuracy"] < MIN_ACCURACY:
        MODEL_PATH.unlink()  # se borra: un modelo malo no debe quedar en models/
        raise SystemExit(
            f"El modelo no alcanzó el mínimo de {MIN_ACCURACY:.0%}. No se guardará."
        )

    print("5) Guardando métricas...")
    save_metrics(metrics, device, seconds)
    print(f"   Modelo:   {MODEL_PATH}")
    print(f"   Métricas: {METRICS_PATH}")
    print("Listo!")


if __name__ == "__main__":
    main()
