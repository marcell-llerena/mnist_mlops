# =============================================================================
# Dockerfile MULTI-STAGE: la "receta" para construir la imagen de MNIST MLOps
# =============================================================================
# IMAGEN     = una caja sellada con todo lo necesario para ejecutar el proyecto.
# CONTENEDOR = esa caja funcionando (una imagen "encendida").
#
# ¿Qué es "multi-stage"? Un Dockerfile con VARIOS "FROM". Cada FROM empieza una
# etapa nueva, desde cero. La imagen final es SOLO la última etapa: de las
# anteriores únicamente se lleva lo que copiamos a propósito con COPY --from.
#
# Analogía: la etapa 1 es la COCINA (ollas, horno, harina: todo el desorden)
# y la etapa 2 es el PLATO SERVIDO. Las ollas no van a la mesa.
#
#   ┌───────────────── ETAPA 1: "entrenador" ─────────────────┐
#   │ Python + torch (~600 MB)                                 │
#   │ preparar-datos → entrenar → puerta de calidad            │
#   │                     │                                    │
#   │                     ▼ models/model.onnx (~430 KB)        │
#   └─────────────────────┼────────────────────────────────────┘
#                         │  COPY --from=entrenador (lo único que pasa)
#   ┌─────────────────────▼─── ETAPA 2: producción ───────────┐
#   │ Python + onnxruntime + FastAPI + Gradio (SIN torch)      │
#   │ ← esta es la imagen que corre en Render                  │
#   └──────────────────────────────────────────────────────────┘
#
# ¿Por qué? Render Free da 512 MB de RAM. Con torch en producción la imagen
# pesaría ~1.5 GB y la API arrancaría lenta y gastando mucha memoria.
# Para PREDECIR no hace falta torch: basta con onnxruntime.
#
# Una sola imagen sirve para los dos servicios (API y Gradio).
# El archivo compose.yaml decide qué comando ejecuta cada uno.
#
# Usarlo sin compose (desde la carpeta del proyecto):
#   docker build -t mnist-mlops .           → construir la imagen
#   docker run -p 8000:8000 mnist-mlops     → encender un contenedor con la API
#
# Guía paso a paso: docs/docker.md
# =============================================================================


# #############################################################################
# ETAPA 1: ENTRENADOR (se descarta al terminar)
# #############################################################################

# "AS entrenador" le pone nombre a la etapa, para luego copiar desde ella
FROM python:3.14-slim AS entrenador

# Traer uv desde su imagen oficial (versión fija, igual que en el CI)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

# UV_COMPILE_BYTECODE=1 → precompila los .py: arranca más rápido
# UV_LINK_MODE=copy     → copia los archivos en vez de enlazarlos (evita avisos)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Instalar las librerías primero (aprovecha la caché de Docker, igual que en Iris)
#   --no-default-groups --group train → dependencias normales + grupo "train"
#                                        (torch), pero SIN el grupo dev (pytest)
# En Linux, torch viene del índice "solo CPU" (ver [tool.uv.sources] en pyproject.toml)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --group train --no-install-project

COPY . .
RUN uv sync --locked --no-default-groups --group train

# Descargar MNIST, entrenar (en CPU: aquí no hay GPU) y exportar a ONNX.
# Si el accuracy no supera MIN_ACCURACY, "entrenar" falla y la imagen NO se
# construye: la puerta de calidad también protege a Docker.
RUN uv run --no-sync preparar-datos && uv run --no-sync entrenar


# #############################################################################
# ETAPA 2: PRODUCCIÓN (esta es la imagen final)
# #############################################################################

# Un FROM nuevo = empezar otra vez desde una imagen limpia.
# Nada de la etapa 1 (torch, MNIST, cachés) llega aquí si no lo copiamos.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# --no-default-groups → solo las dependencias de [project]: onnxruntime,
# FastAPI, Gradio, Pillow... SIN torch y SIN pytest
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY . .
RUN uv sync --locked --no-default-groups

# ⭐ La línea clave del multi-stage: traer de la etapa "entrenador" SOLO el
# modelo entrenado y su ficha técnica. Todo lo demás de esa etapa se descarta.
COPY --from=entrenador /app/models/model.onnx /app/models/metrics.json ./models/

# Usar el entorno virtual por defecto: "python", "fastapi", "predecir"...
# funcionan sin escribir "uv run"
ENV PATH="/app/.venv/bin:$PATH"

# No ejecutar como administrador (si alguien encontrara una falla, tendría
# menos permisos)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Comando por defecto: la API. compose.yaml lo reemplaza para Gradio
# (en Render, lo mismo hace el campo "Docker Command").
# ${PORT:-8000} = "el valor de PORT; si no existe, 8000" (Render define PORT).
CMD ["sh", "-c", "exec fastapi run src/mnist_mlops/api.py --port ${PORT:-8000}"]
