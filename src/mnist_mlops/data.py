"""
Paso 1: DATOS

- Descarga MNIST: 70,000 imágenes de dígitos escritos a mano (0 al 9), de 28x28 píxeles.
- Valida que los datos tengan la forma esperada.
- Guarda un PNG de ejemplo de cada dígito en examples/ (para probar la app).
- Prepara los "DataLoaders": los que le entregan las imágenes al modelo por lotes.

MNIST ya viene separado en dos partes:
  - train: 60,000 imágenes para entrenar
  - test:  10,000 imágenes para evaluar (el modelo nunca las ve al entrenar)

Ejecutar:  uv run preparar-datos
"""

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_mlops.config import (
    CLASSES,
    DATA_DIR,
    EXAMPLES_DIR,
    IMAGE_SIZE,
    N_TEST,
    N_TRAIN,
    TRAIN_PARAMS,
)

# Convierte cada imagen en un "tensor" (una tabla de números) con valores entre 0 y 1
TO_TENSOR = transforms.ToTensor()

# "Aumento de datos" (data augmentation): en cada época, cada imagen de entrenamiento
# se gira, mueve y escala un poquito al azar. Así el modelo ve más variedad y se
# acostumbra a dígitos torcidos o descentrados, como los que dibuja la gente en la app.
# Solo se aplica al entrenar: la evaluación usa las imágenes tal cual.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        TO_TENSOR,
    ]
)


def download_data() -> tuple[datasets.MNIST, datasets.MNIST]:
    """Descarga MNIST en data/raw/MNIST/ (si ya está descargado, no lo vuelve a bajar)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)
    return train, test


def validate_data(dataset: datasets.MNIST, expected_size: int) -> None:
    """Revisiones básicas de calidad de datos. Si algo falla, se detiene todo."""
    images, labels = dataset.data, dataset.targets  # tensores con TODAS las imágenes y etiquetas

    if len(images) != expected_size:
        raise ValueError(f"Se esperaban {expected_size} imágenes y hay {len(images)}")

    if tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Las imágenes deben medir {IMAGE_SIZE}x{IMAGE_SIZE}")

    if len(images) != len(labels):
        raise ValueError("Hay distinta cantidad de imágenes que de etiquetas")

    if labels.min() < 0 or labels.max() >= len(CLASSES):
        raise ValueError("Hay etiquetas fuera del rango 0-9")

    if len(labels.unique()) != len(CLASSES):
        raise ValueError("Falta algún dígito: deben estar los 10 (0-9)")


def load_data() -> tuple[datasets.MNIST, datasets.MNIST]:
    """Devuelve los datos de entrenamiento y prueba, ya validados y listos para el modelo."""
    download_data()
    train = datasets.MNIST(DATA_DIR, train=True, transform=TRAIN_TRANSFORM)
    test = datasets.MNIST(DATA_DIR, train=False, transform=TO_TENSOR)
    validate_data(train, N_TRAIN)
    validate_data(test, N_TEST)
    return train, test


def make_loaders(train, test) -> tuple[DataLoader, DataLoader]:
    """Arma los DataLoaders: entregan las imágenes al modelo de a `batch_size`."""
    batch_size = TRAIN_PARAMS["batch_size"]
    return (
        # shuffle=True: en cada época las imágenes llegan en otro orden (aprende mejor)
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1000),
    )


def save_examples(test: datasets.MNIST) -> None:
    """Guarda la primera imagen de prueba de cada dígito como examples/digito_N.png."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for digit in range(len(CLASSES)):
        index = int((test.targets == digit).nonzero()[0])  # primera imagen con ese dígito
        image = Image.fromarray(test.data[index].numpy())
        image.save(EXAMPLES_DIR / f"digito_{digit}.png")


def main() -> None:
    train, test = download_data()
    validate_data(train, N_TRAIN)
    validate_data(test, N_TEST)
    save_examples(test)
    print(f"Datos guardados en: {DATA_DIR / 'MNIST'}")
    print(f"Entrenamiento: {len(train)} imágenes | Prueba: {len(test)} imágenes")
    print(f"Tamaño de cada imagen: {IMAGE_SIZE}x{IMAGE_SIZE} píxeles (escala de grises)")
    print(f"Ejemplos guardados en: {EXAMPLES_DIR}")
    print("\nCantidad de imágenes de entrenamiento por dígito:")
    for digit, count in zip(*train.targets.unique(return_counts=True)):
        print(f"  {int(digit)}: {int(count)}")


if __name__ == "__main__":
    main()
