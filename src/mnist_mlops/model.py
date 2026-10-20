"""
La ARQUITECTURA del modelo: una red neuronal convolucional (CNN) pequeña.

Una CNN mira la imagen "por partes", con filtros de 3x3 píxeles que aprenden
a detectar bordes, curvas y trazos. Capa a capa combina esas piezas
hasta reconocer el dígito completo.

  imagen 1x28x28
     │  Conv 3x3 (16 filtros) + ReLU + MaxPool  → 16x14x14   (bordes, trazos)
     │  Conv 3x3 (32 filtros) + ReLU + MaxPool  → 32x7x7     (curvas, partes de dígitos)
     │  Aplanar                                 → 1568 números
     │  Lineal (64) + ReLU                      → 64          (combina lo anterior)
     ▼  Lineal (10)                             → 10 puntajes, uno por dígito (0-9)

Son ~106,000 parámetros: el archivo ONNX pesa ~430 KB.
"""

import torch
from torch import nn

from mnist_mlops.config import MNIST_MEAN, MNIST_STD


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce la imagen a la mitad: 28 → 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 → 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # apaga 20% de neuronas al azar al entrenar: evita "memorizar"
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # La normalización va DENTRO del modelo: así queda guardada en el ONNX
        # y la API no puede olvidarse de aplicarla (ni aplicarla distinto).
        x = (x - MNIST_MEAN) / MNIST_STD
        return self.classifier(self.features(x))
