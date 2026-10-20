# ✍️ MNIST MLOps

Proyecto de bootcamp: una **red neuronal** que reconoce **dígitos escritos a mano** (0 al 9),
llevada de punta a punta: **datos → entrenamiento → ONNX → API → interfaz web → nube**.

Es el "hermano mayor" de [Iris MLOps](../iris_mlops): la misma organización, pero con
**imágenes** y **deep learning** (PyTorch). Eso trae problemas nuevos de MLOps que Iris no tenía:

| Problema nuevo | Cómo se resuelve aquí |
|---|---|
| PyTorch pesa ~600 MB y Render Free solo da 512 MB de RAM | Se entrena con PyTorch, pero se sirve con **ONNX + onnxruntime** (sin torch) |
| La imagen de Docker no debe llevar herramientas de entrenamiento | **Dockerfile multi-stage**: una etapa entrena, otra sirve |
| Entrenar es más rápido con GPU | El código elige solo: **MPS** (Mac), **CUDA** (NVIDIA) o **CPU** |
| Un dibujo en la web no se parece a una imagen de MNIST | **Preprocesamiento** que convierte cualquier imagen al formato de MNIST |

---

## 🧭 ¿Qué hace el proyecto?

```
 ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
 │ 1. DATOS   │ → │ 2. ENTRENAR  │ → │ 3. EXPORTAR│ → │  4. API    │ → │5. INTERFAZ │
 │ MNIST      │   │ CNN (PyTorch)│   │ model.onnx │   │  FastAPI   │   │  Gradio    │
 │ 70k imgs   │   │ MPS/CUDA/CPU │   │ + puerta de│   │  /predict  │   │ (dibujar)  │
 │ (validado) │   │              │   │   calidad  │   │ onnxruntime│   │            │
 └────────────┘   └──────────────┘   └────────────┘   └────────────┘   └────────────┘
   data.py          model.py           train.py         preprocess.py    gradio_app.py
                    train.py                            predict.py
                                                        api.py
 ◄────────── con torch (entrenar) ──────────►  ◄──────── sin torch (producción) ───────►
```

1. **Datos**: se descarga MNIST (60,000 imágenes para entrenar y 10,000 para evaluar) y se valida.
2. **Entrenamiento**: una red convolucional pequeña (~106,000 parámetros) aprende durante 3 épocas.
3. **Exportación**: el modelo se guarda como `model.onnx`, un formato estándar que **no necesita
   PyTorch** para funcionar. Se evalúa ese archivo y, **solo si llega a 97% de accuracy**, se guarda.
4. **API**: recibe una imagen, la convierte al formato de MNIST y devuelve el dígito.
5. **Interfaz**: una página para **dibujar** un dígito con el mouse (o subir una imagen).

---

## 📁 Estructura del proyecto

```
mnist_mlops/
├── pyproject.toml          # Dependencias: las de servir + grupos "train" (torch) y "dev" (pytest)
├── uv.lock                 # Versiones exactas (reproducibilidad)
├── .python-version         # Python 3.14
├── .github/workflows/
│   ├── ci.yml              # 🤖 En cada push: datos → entrenar → pruebas → API sin torch
│   └── cd.yml              # 🚀 Si el CI pasa: despliega en Render
├── Dockerfile              # 🐳 Multi-stage: etapa "entrenador" + etapa de producción
├── compose.yaml            # 🐳 Levanta API + Gradio con un comando
├── .dockerignore
│
├── data/raw/MNIST/         # Se descarga con `uv run preparar-datos` (no se sube a git)
├── examples/               # Un PNG real de MNIST por dígito (para probar)
├── models/                 # model.onnx + metrics.json (se generan al entrenar)
│
├── src/mnist_mlops/
│   ├── config.py           # ⚙️  Rutas y parámetros en un solo lugar
│   ├── data.py             # 1️⃣  Descargar, validar y preparar los datos
│   ├── model.py            # 🧠 La arquitectura de la red (CNN)
│   ├── train.py            # 2️⃣  Entrenar, exportar a ONNX, evaluar y guardar
│   ├── preprocess.py       # 3️⃣  Convertir cualquier imagen al formato MNIST
│   ├── predict.py          # 3️⃣  Cargar el ONNX y predecir (sin torch)
│   └── api.py              # 4️⃣  API web con FastAPI
├── app/
│   └── gradio_app.py       # 5️⃣  Interfaz: dibujar o subir un dígito
├── tests/                  # ✅ Pruebas automáticas (datos, modelo, API)
└── docs/
    ├── docker.md           # 🐳 Guía de Docker (y del multi-stage)
    └── cd.md               # 🚀 Guía de despliegue en Render
```

---

## 🚀 Cómo ejecutarlo (paso a paso)

**Requisito:** tener [uv](https://docs.astral.sh/uv/) instalado.

```bash
# 0. Instalar todo (Python 3.14 + librerías, incluido torch)
uv sync

# 1. Descargar y validar MNIST (~12 MB de descarga)
uv run preparar-datos

# 2. Entrenar, exportar a ONNX y evaluar (crea models/model.onnx y models/metrics.json)
uv run entrenar

# 3. Probar una predicción rápida (o con tu propia imagen: uv run predecir mi_dibujo.png)
uv run predecir

# 4. Ejecutar las pruebas
uv run pytest
```

### 🍎 ¿MPS, CUDA o CPU?

`entrenar` usa el dispositivo más rápido que encuentre. Medido en una Mac con chip Apple:

| Dispositivo | Dónde se usa | Tiempo (3 épocas) | Accuracy |
|---|---|---|---|
| `mps` (GPU de Apple) | Tu Mac | ~15 s | ~98.8% |
| `cpu` | Docker, CI, Render (y tu Mac si lo fuerzas) | ~45 s | ~98.5% |

```bash
DEVICE=cpu uv run entrenar   # forzar la CPU para comparar
```

El archivo `model.onnx` es el mismo sin importar dónde se entrenó: ONNX no depende del dispositivo.
(Los resultados varían un poco entre MPS y CPU, pero eso no importa: la puerta de calidad mide
el accuracy contra un mínimo, no un número exacto).

### Levantar la API y la interfaz

Se necesitan **dos terminales**:

```bash
# Terminal 1: la API
uv run fastapi dev src/mnist_mlops/api.py
#   → Documentación interactiva: http://127.0.0.1:8000/docs

# Terminal 2: la interfaz (le pide las predicciones a la API)
uv run app/gradio_app.py
#   → http://127.0.0.1:7860
```

---

## 🐳 Correrlo con Docker en tu computadora

Con Docker **no necesitas uv, Python ni torch instalados**: todo se construye dentro de la imagen,
igual que en Render. Es la mejor forma de probar "como en producción" antes de hacer `git push`.

### Requisitos

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado.
2. Docker Desktop **abierto** y listo (el ícono de la ballena quieto). Para comprobarlo:

   ```bash
   docker version      # debe mostrar una sección "Server:" sin errores
   ```

### Opción A: todo junto con Docker Compose (recomendada)

Desde la carpeta `mnist_mlops/`:

```bash
# 1. Construir la imagen (las 2 etapas) y encender la API + Gradio
docker compose up --build
```

Verás en la terminal cómo la etapa "entrenador" descarga MNIST y entrena en CPU
(`Época 1... Época 3... Accuracy: 98.xx%`). Cuando aparezca que los servicios están corriendo, abre:

| Servicio | Dirección |
|---|---|
| ✍️ Gradio (dibujar un dígito) | http://localhost:7860 |
| 🔌 API (documentación interactiva) | http://localhost:8000/docs |

```bash
# 2. Probar la API desde la terminal (en otra terminal)
curl -F "file=@examples/digito_7.png" http://localhost:8000/predict

# 3. Apagar: Ctrl + C en la terminal de compose, y luego
docker compose down
```

> 💡 Con `docker compose up --build -d` los servicios quedan corriendo "de fondo" y la terminal
> queda libre. Se ven sus mensajes con `docker compose logs -f` y se apagan con `docker compose down`.

### Opción B: solo la API, sin Compose

```bash
docker build -t mnist-mlops .                  # construir la imagen
docker run --rm -p 8000:8000 mnist-mlops       # encender la API → http://localhost:8000/docs
```

`--rm` borra el contenedor al apagarlo (Ctrl + C) y `-p 8000:8000` conecta el puerto de tu
computadora con el del contenedor.

### ⏱️ ¿Cuánto tarda? (medido en una Mac con chip Apple)

| Situación | Tiempo |
|---|---|
| Primera construcción (descarga torch y MNIST, entrena en CPU) | ~1 a 2 minutos |
| Reconstruir después de cambiar código | ~40 segundos (reutiliza las librerías, pero **vuelve a entrenar**) |
| Volver a encender sin cambios (`docker compose up`) | segundos |

Dentro de Docker **no hay MPS**: siempre entrena en CPU (~25 s, ~98.6% de accuracy).

### 🔍 Comprobar que el multi-stage funcionó

```bash
docker image ls | grep mnist                                    # tamaño de la imagen final
docker run --rm mnist-mlops python -c "import torch"            # debe fallar: No module named 'torch'
docker run --rm mnist-mlops predecir                            # predice examples/digito_7.png
docker run --rm mnist-mlops cat models/metrics.json             # ficha del modelo entrenado en el build
docker build --target entrenador -t mnist-entrenador .          # construir solo la etapa 1, para comparar
```

| Imagen | Tamaño medido | ¿Tiene torch? |
|---|---|---|
| `mnist-entrenador` (etapa 1, se descarta) | ~3.4 GB | ✅ |
| `mnist-mlops` (etapa 2, la que se usa) | ~1.35 GB | ❌ |

Uso de RAM medido con `docker stats`: API ~90 MB, Gradio ~160 MB (Render Free da 512 MB por servicio).

### 🧯 Si algo falla

| Síntoma | Solución |
|---|---|
| `Cannot connect to the Docker daemon` o `500 Internal Server Error ... docker.sock` | Docker Desktop está cerrado o trabado: ábrelo o reinícialo (ballena → **Restart**) |
| `port is already allocated` | Ya hay algo en ese puerto (¿`uv run fastapi dev` o `gradio_app.py` corriendo?). Apágalo o usa `-p 8001:8000` |
| Cambié el código y no veo el cambio | Faltó `--build`: `docker compose up --build` |
| El build falla en `entrenar` | El modelo no superó `MIN_ACCURACY`: ¡la puerta de calidad funcionó! |
| Gradio dice "No se pudo conectar con la API" | Revisa `docker compose logs api` |

Guía completa (cómo funciona el multi-stage línea por línea): [`docs/docker.md`](docs/docker.md).

---

## 🔌 Usar la API

A diferencia de Iris (4 números en JSON), aquí se envía un **archivo de imagen**:

```bash
curl -F "file=@examples/digito_7.png" http://127.0.0.1:8000/predict
```

```json
{"digit": 7, "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0001, "...": "...", "7": 0.9999}}
```

| Endpoint | Método | ¿Qué hace? |
|---|---|---|
| `/` | GET | Comprueba que la API está viva |
| `/model-info` | GET | Ficha técnica del modelo (accuracy, dónde se entrenó, fecha...) |
| `/predict` | POST | Recibe una imagen (`file`) y devuelve el dígito y sus probabilidades |

Errores: `422` si el archivo no es una imagen o está vacía, `413` si pesa más de 5 MB,
`503` si todavía no hay modelo.

---

## 🧠 Conceptos clave

### ONNX: entrenar con una herramienta, servir con otra

**ONNX** (*Open Neural Network Exchange*) es un formato estándar para guardar redes neuronales,
como un PDF para modelos: lo crea una herramienta (PyTorch) y lo lee otra (onnxruntime).

| | PyTorch | onnxruntime |
|---|---|---|
| Sirve para | Entrenar y predecir | **Solo** predecir |
| Tamaño instalado | ~600 MB | ~75 MB |
| Dónde se usa aquí | Tu computadora, CI, etapa 1 del Dockerfile | API en producción |

La **puerta de calidad evalúa el archivo ONNX**, no el modelo de PyTorch: así se comprueba el
archivo que realmente llega a producción (si algo se rompió al exportar, se detecta).

### La normalización va dentro del modelo

Las redes aprenden mejor con datos "centrados" (`(pixel - promedio) / desviación`). Ese cálculo
está **dentro de `DigitCNN.forward()`**, así que queda guardado en el ONNX. La API no tiene que
recordarlo: un error menos posible entre entrenamiento y producción.

### Training-serving skew: por qué existe `preprocess.py`

El modelo aprendió con imágenes de MNIST: **28x28, dígito blanco sobre negro, centrado**.
Un dibujo en Gradio es **400x400, negro sobre blanco, en cualquier parte del lienzo**. Si se lo
pasáramos tal cual, fallaría aunque el modelo sea excelente.

`preprocess.py` lo transforma paso a paso: fondo transparente → blanco, escala de grises,
invertir colores, recortar el dígito, achicarlo a 20x20, centrarlo en 28x28 por su "centro de masa".
A este desajuste entre los datos de entrenamiento y los de producción se le llama
**training-serving skew**, y es una de las fallas más comunes de los modelos en producción.

### ¿98.5% significa que nunca se equivoca?

No: falla en ~150 de las 10,000 imágenes de prueba. Mira `examples/digito_3.png` en la pestaña
"Subir imagen": es un 3 real de MNIST que el modelo confunde con un 8. ¿Tú lo reconocerías?
El accuracy por dígito está en `models/metrics.json` (`accuracy_per_digit`).

---

## ✅ Las pruebas

| Archivo | ¿Qué comprueba? |
|---|---|
| `test_data.py` | MNIST tiene 60k/10k imágenes de 28x28, los 10 dígitos, y la validación detecta etiquetas inválidas |
| `test_model.py` | La red da 10 puntajes, el **ONNX** supera el mínimo, predice bien un ejemplo y un "dibujo" negro sobre blanco, y el preprocesamiento funciona |
| `test_api.py` | La API responde, predice desde un PNG, y rechaza archivos que no son imágenes o lienzos vacíos |

El CI además comprueba que **la API funciona sin torch instalado**, igual que en producción.

---

## ☁️ Despliegue (CI/CD con Render)

Igual que en Iris: `git push` → CI ✅ → CD llama al deploy hook → Render construye la imagen
(multi-stage: entrena en la etapa 1) y publica la API y Gradio. Guía: [`docs/cd.md`](docs/cd.md).

---

## 🧰 Tecnologías

| Herramienta | ¿Para qué? |
|---|---|
| **uv** | Instalar Python y librerías, con versiones exactas (`uv.lock`) |
| **PyTorch + torchvision** | Descargar MNIST y entrenar la red neuronal |
| **ONNX + onnxruntime** | Guardar el modelo en un formato estándar y ejecutarlo sin PyTorch |
| **Pillow + NumPy** | Preprocesar las imágenes |
| **FastAPI** | Servir el modelo como API web |
| **Gradio** | Interfaz web para dibujar dígitos |
| **pytest** | Pruebas automáticas |
| **Docker (multi-stage)** | Empaquetar: entrenar en una etapa, servir liviano en otra |
| **GitHub Actions + Render** | CI/CD y publicación en internet |
