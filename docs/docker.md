# 🐳 Docker en el proyecto MNIST MLOps

Esta guía se enfoca en **lo nuevo respecto a Iris MLOps: el Dockerfile multi-stage**.
Los conceptos básicos (imagen, contenedor, capas, caché, compose, `0.0.0.0`, `API_URL`...)
son los mismos y están explicados en la [guía de Docker de Iris](../../iris_mlops/docs/docker.md).

---

## 🚀 Uso rápido

**Requisito:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y **abierto**.

```bash
docker compose up --build      # construir (las 2 etapas) y encender API + Gradio
```

La primera vez tarda **1 a 2 minutos**: descarga torch, descarga MNIST y **entrena en CPU**
(dentro de Docker no hay MPS ni GPU). Después de cambiar código, reconstruir tarda ~40 s:
reutiliza las librerías, pero vuelve a entrenar (el código está copiado antes de `entrenar`). Cuando termine:

| Servicio | Dirección |
|---|---|
| 🔌 API (documentación interactiva) | http://localhost:8000/docs |
| ✍️ Interfaz con Gradio (dibujar) | http://localhost:7860 |

Para apagar: **Ctrl + C** y luego `docker compose down`.

---

## 🍳 ¿Qué es un Dockerfile multi-stage?

Un Dockerfile con **varios `FROM`**. Cada `FROM` empieza una **etapa** nueva, desde cero.
La imagen final es **solo la última etapa**: de las anteriores únicamente se lleva lo que
copiamos a propósito con `COPY --from=<etapa>`.

**Analogía:** la etapa 1 es la **cocina** (ollas, horno, harina, todo el desorden) y la etapa 2 es
el **plato servido**. Las ollas no van a la mesa.

```
┌──────────────────── ETAPA 1: "entrenador" ────────────────────┐
│ FROM python:3.14-slim AS entrenador                            │
│ uv sync --group train          → torch (CPU), torchvision, onnx│
│ preparar-datos                 → descarga MNIST                │
│ entrenar                       → entrena, exporta, evalúa ONNX │
│                                   (puerta de calidad)          │
│                       models/model.onnx  +  metrics.json       │
└──────────────────────────────────┬─────────────────────────────┘
                                   │  COPY --from=entrenador
                                   │  (lo ÚNICO que pasa a la etapa 2)
┌──────────────────── ETAPA 2: producción ─┼────────────────────┐
│ FROM python:3.14-slim                    ▼                     │
│ uv sync --no-default-groups    → onnxruntime, FastAPI, Gradio  │
│ models/model.onnx              ← copiado de la etapa 1         │
│ CMD fastapi run ...                                            │
│                                                                │
│ ← ESTA es la imagen "mnist-mlops" (la que corre en Render)     │
└────────────────────────────────────────────────────────────────┘
```

### ¿Por qué no hacerlo en una sola etapa, como en Iris?

En Iris, scikit-learn sirve para entrenar **y** para predecir, y es liviano. Aquí no:

| | Una sola etapa (con torch) | Multi-stage (sin torch) |
|---|---|---|
| Tamaño de la imagen (medido con `docker image ls`) | ~3.4 GB (la etapa "entrenador") | ~1.35 GB |
| RAM de la API (medido con `docker stats`) | Más de lo necesario (torch se carga entero) | ~90 MB |
| Riesgo con 512 MB de RAM en Render | Alto | Bajo |
| Datos de MNIST dentro de la imagen | Sí (~55 MB que nadie usa) | No |
| Puerta de calidad en el build | ✅ | ✅ (si `entrenar` falla, la etapa 1 falla y no hay imagen) |

(El build completo tardó ~70 s en una Mac con chip Apple; el entrenamiento en CPU, ~25 s con 98.6% de accuracy).

### Las líneas clave

```dockerfile
FROM python:3.14-slim AS entrenador       # 1. ponerle nombre a la etapa
...
RUN uv sync --locked --no-default-groups --group train   # con torch
RUN uv run --no-sync preparar-datos && uv run --no-sync entrenar

FROM python:3.14-slim                     # 2. empezar de cero otra vez
...
RUN uv sync --locked --no-default-groups  # sin torch
COPY --from=entrenador /app/models/model.onnx /app/models/metrics.json ./models/   # 3. traer solo el modelo
```

### ¿Cómo sabe uv qué instalar en cada etapa?

Por los **grupos de dependencias** de `pyproject.toml`:

| Grupo | Contiene | Etapa 1 (entrenador) | Etapa 2 (producción) | CI / tu computadora |
|---|---|---|---|---|
| `[project] dependencies` | onnxruntime, FastAPI, Gradio, Pillow, NumPy | ✅ | ✅ | ✅ |
| `train` | torch, torchvision, onnx, onnxscript | ✅ | ❌ | ✅ |
| `dev` | pytest, httpx | ❌ | ❌ | ✅ |

- `uv sync` (sin opciones) instala los tres, porque `default-groups = ["dev", "train"]`.
- `--no-default-groups` = solo `[project]`. `--group train` = agregar ese grupo.

### torch "solo CPU" en Linux

El torch de PyPI para Linux incluye **CUDA** (para GPUs NVIDIA): ~3 GB extra que en Docker y en
Render no sirven (no hay GPU). Por eso `pyproject.toml` le indica a uv que, **solo en Linux**, lo
descargue del índice CPU de PyTorch:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

En tu Mac se usa el torch de PyPI, que trae soporte para **MPS**. Las dos variantes quedan
registradas en `uv.lock`.

---

## 🧩 compose.yaml: dos servicios, una imagen

```
            TU COMPUTADORA (navegador)
        localhost:7860          localhost:8000
              │                       │
 ┌────────────┼───────────────────────┼──────────┐
 │            ▼                       ▼          │
 │     ┌─────────────┐  API_URL  ┌────────────┐  │
 │     │   gradio    │ ────────► │    api     │  │
 │     │    :7860    │  http://  │   :8000    │  │
 │     └─────────────┘  api:8000 │ model.onnx │  │
 │                               └────────────┘  │
 │              RED INTERNA DE DOCKER            │
 └───────────────────────────────────────────────┘
```

Igual que en Iris, pero sin Streamlit. Gradio **no carga el modelo**: le envía la imagen a la API.

---

## 🛠️ Comandos útiles

| Comando | ¿Para qué sirve? |
|---|---|
| `docker compose up --build` | Construir (si hay cambios) y encender todo |
| `docker compose logs -f api` | Ver los mensajes de la API en vivo |
| `docker compose down` | Apagar y borrar los contenedores |
| `docker build -t mnist-mlops .` | Construir solo la imagen final |
| `docker build --target entrenador -t mnist-entrenador .` | Construir **solo la etapa 1** (para ver qué tiene) |
| `docker image ls` | Comparar cuánto pesa cada imagen |
| `docker run --rm mnist-mlops python -c "import torch"` | Comprobar que la imagen final **no** tiene torch (debe dar error) |
| `docker run --rm mnist-mlops cat models/metrics.json` | Ver la ficha del modelo que se entrenó en el build |

> 💡 Ejercicio: construye las dos etapas con los comandos de arriba y compara sus tamaños con
> `docker image ls`. Esa diferencia es lo que ahorra el multi-stage.

---

## 🧯 Problemas comunes

| Mensaje / síntoma | Causa | Solución |
|---|---|---|
| `Cannot connect to the Docker daemon` o `500 Internal Server Error ... docker.sock` | Docker Desktop cerrado o trabado | Ábrelo (o reinícialo: ícono de la ballena → **Restart**) y espera a que quede quieto |
| El build tarda mucho en `uv sync ... --group train` | Descarga torch (~200 MB comprimido) | Solo la primera vez: después usa la caché |
| El build tarda en `entrenar` | Entrena en CPU (en Docker no hay MPS) | Normal: ~25 s en una Mac con chip Apple; más en computadoras lentas |
| El build falla en `preparar-datos` | No se pudo descargar MNIST | Revisa tu internet y vuelve a construir |
| El build falla en `entrenar` | El modelo no superó `MIN_ACCURACY` | ¡La puerta de calidad funcionó! Revisa el accuracy en el mensaje |
| `port is already allocated` | Otro programa usa el puerto (¿`uv run fastapi dev`?) | Apágalo, o usa `"8001:8000"` en `compose.yaml` |
