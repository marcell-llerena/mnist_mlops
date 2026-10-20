"""
Paso 3a: PREPROCESAMIENTO

Convierte CUALQUIER imagen (un dibujo en la app, una foto, un PNG de MNIST...)
al formato exacto con el que se entrenó el modelo.

¿Por qué hace falta? Las imágenes de MNIST tienen reglas muy concretas:
  - 28x28 píxeles, escala de grises
  - dígito BLANCO sobre fondo NEGRO
  - el dígito ocupa un cuadro de ~20x20 y está centrado

Un dibujo en la app suele ser negro sobre blanco, de 800x800 y en una esquina.
Si se lo pasamos tal cual, el modelo falla aunque sea muy bueno: nunca vio
algo así. Este desajuste entre los datos de entrenamiento y los de producción
es uno de los problemas más comunes en MLOps (se llama "training-serving skew").

Solo usa numpy y Pillow: funciona en producción, donde no hay torch.
"""

import numpy as np
from PIL import Image

from mnist_mlops.config import IMAGE_SIZE

DIGIT_BOX = 20  # en MNIST el dígito cabe en un cuadro de 20x20 dentro de los 28x28


def preprocess(image: Image.Image) -> np.ndarray:
    """Devuelve un arreglo de forma (1, 1, 28, 28) con valores entre 0 y 1."""
    # 1. Fondo transparente → blanco (los dibujos de la app vienen con transparencia)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image)

    # 2. Escala de grises, con valores entre 0 (negro) y 1 (blanco)
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255

    # 3. Si el fondo es claro, invertir colores: MNIST es blanco sobre negro.
    #    El "fondo" se estima con los bordes de la imagen.
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if border.mean() > 0.5:
        pixels = 1 - pixels

    # 4. Recortar el dígito: quedarse solo con el rectángulo donde hay trazo
    rows, cols = np.nonzero(pixels > 0.2)
    if len(rows) == 0:
        raise ValueError("La imagen está vacía: no se encontró ningún dígito")
    digit = pixels[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    # 5. Achicar para que quepa en 20x20, sin deformarlo (manteniendo la proporción)
    height, width = digit.shape
    scale = DIGIT_BOX / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit_image = Image.fromarray((digit * 255).astype(np.uint8))
    digit = np.asarray(digit_image.resize(new_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255

    # 6. Pegarlo en el centro de un lienzo negro de 28x28
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    height, width = digit.shape
    top, left = (IMAGE_SIZE - height) // 2, (IMAGE_SIZE - width) // 2
    canvas[top : top + height, left : left + width] = digit

    # 7. Centrar por "centro de masa" (donde se concentra la tinta), como en MNIST.
    #    Por ejemplo, un 7 tiene más tinta arriba: se baja un poco.
    canvas = _center_by_mass(canvas, top, left, height, width)

    return canvas.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)


def _center_by_mass(canvas, top, left, height, width) -> np.ndarray:
    ys, xs = np.indices(canvas.shape)
    total = canvas.sum()
    center_y, center_x = (ys * canvas).sum() / total, (xs * canvas).sum() / total
    middle = (IMAGE_SIZE - 1) / 2
    # Cuánto mover, limitado para que el dígito no se salga del lienzo
    shift_y = int(np.clip(round(middle - center_y), -top, IMAGE_SIZE - top - height))
    shift_x = int(np.clip(round(middle - center_x), -left, IMAGE_SIZE - left - width))
    return np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
