"""
Configuración central del proyecto.

Aquí guardamos en UN SOLO LUGAR todas las rutas y parámetros.
Si queremos cambiar algo (por ejemplo, el número de épocas),
solo lo cambiamos aquí y todo el proyecto se entera.

Ojo: este archivo NO importa torch. Lo usa también la API, que en
producción no tiene torch instalado.
"""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------
# Carpeta raíz del proyecto (sube 3 niveles desde este archivo:
# config.py -> mnist_mlops/ -> src/ -> raíz)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"  # torchvision guarda MNIST en data/raw/MNIST/
EXAMPLES_DIR = ROOT_DIR / "examples"  # un PNG de cada dígito, para probar la app

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.onnx"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --- Datos -----------------------------------------------------------------
IMAGE_SIZE = 28  # las imágenes de MNIST miden 28x28 píxeles, en escala de grises
CLASSES = [str(d) for d in range(10)]  # lo que el modelo puede responder: "0" ... "9"
N_TRAIN = 60_000  # imágenes de entrenamiento que trae MNIST
N_TEST = 10_000  # imágenes de prueba (el modelo nunca las ve al entrenar)

# Promedio y desviación de los píxeles de MNIST. El modelo los usa para
# "normalizar" la imagen (centrarla en 0), lo que ayuda a que aprenda mejor.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# --- Entrenamiento ---------------------------------------------------------
RANDOM_STATE = 42  # "semilla": hace que los resultados sean reproducibles

TRAIN_PARAMS = {
    "epochs": 3,  # cuántas veces el modelo recorre las 60,000 imágenes
    "batch_size": 128,  # cuántas imágenes ve antes de corregirse una vez
    "learning_rate": 0.001,  # qué tan grandes son esas correcciones
}

# Calidad mínima aceptable: si el modelo no la alcanza, no se guarda
MIN_ACCURACY = 0.97

# --- API -------------------------------------------------------------------
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB: imágenes más grandes se rechazan
