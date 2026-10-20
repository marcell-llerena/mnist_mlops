# 🚀 Despliegue continuo (CD) en el proyecto MNIST MLOps

Funciona **igual que en Iris MLOps**: si ya lo configuraste allá, aquí son los mismos pasos con
otros nombres. Lo nuevo es lo que pasa **dentro de Render** al construir la imagen (multi-stage).

---

## 🧭 El recorrido completo

```
 git push a main
      │
      ▼
 CI (ci.yml): datos → entrenar (MIN_ACCURACY) → pytest → API sin torch ── ❌ falla → no se despliega
      │ ✅
      ▼
 CD (cd.yml): llama a los deploy hooks de Render (API y Gradio)
      │
      ▼
 Render: descarga el código → docker build
            ├─ etapa 1 "entrenador": instala torch (CPU), descarga MNIST, entrena, exporta model.onnx
            └─ etapa 2 producción:   onnxruntime + FastAPI + Gradio + model.onnx  (SIN torch)
         → enciende el contenedor con la etapa 2
      │
      ▼
 https://<tu-servicio>.onrender.com/docs
```

> 💡 El modelo se entrena **dos veces**: en el CI (para probarlo) y en el build de Render (el que
> se publica). En ambos casos con la misma semilla, el mismo código y en CPU, y en ambos pasa por
> la puerta de calidad.

---

## 💸 ¿Cuánto cuesta? Nada (dentro de estos límites)

| Servicio | Gratis | Límites a tener en cuenta |
|---|---|---|
| **GitHub Actions** | Ilimitado en repos públicos | Repos privados: 2,000 minutos/mes. Este CI tarda más que el de Iris (instala torch y entrena) |
| **Render** (plan Free) | 750 horas/mes por cuenta, sin tarjeta | **512 MB de RAM**. Se **duerme** tras 15 min sin visitas; despertar tarda ~1 minuto |

### ¿Por qué cabe en 512 MB de RAM?

Por el multi-stage: en producción no hay torch. Medido en local:

| Proceso | RAM |
|---|---|
| API (FastAPI + onnxruntime + modelo) | ~120 MB |
| Gradio | ~185 MB |

Cada uno es un servicio aparte en Render, con sus propios 512 MB.

---

## 🛠️ Configurarlo paso a paso

Solo se hace **una vez**. Después todo es automático.

### Paso 1: Crear el servicio de la API en Render

1. En [render.com](https://render.com): **+ New** → **Web Service** → elige tu repositorio.
2. Configura:
   - **Language:** `Docker`
   - **Branch:** `main`
   - **Root Directory:** si el repositorio contiene varias carpetas (por ejemplo `iris_mlops/` y
     `mnist_mlops/`), escribe `mnist_mlops`. Si el repositorio es solo este proyecto, déjalo vacío.
   - **Name:** por ejemplo `mnist-api`. **Instance type:** **Free**
3. En **Advanced**:
   - **Health Check Path:** `/`
   - **Auto-Deploy:** **Off** ⚠️ (si no, Render desplegaría aunque el CI falle)
4. **Deploy Web Service**. El primer build tarda varios minutos (descarga torch y entrena en CPU).
   Puedes seguirlo en **Logs**: verás las épocas y el accuracy, igual que en tu terminal.

### Paso 2: Crear el servicio de Gradio

Otro **Web Service** con el mismo repositorio (y el mismo Root Directory). No edites el de la API.

| Campo | Valor |
|---|---|
| **Name** | por ejemplo `mnist-gradio` |
| **Docker Command** | `python app/gradio_app.py` |
| **Health Check Path** | `/` |
| **Auto-Deploy** | **Off** |

Variables de entorno (**Environment Variables**):

| KEY | VALUE | ¿Para qué? |
|---|---|---|
| `API_URL` | `https://mnist-api-xxxx.onrender.com` | Dónde está la API. Sin `/docs` ni `/` al final |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Aceptar visitas desde fuera del contenedor |
| `PORT` | `7860` | Le dice a Render en qué puerto escucha Gradio |

### Paso 3: Conectar GitHub con Render (los deploy hooks)

1. En Render, en cada servicio: **Settings** → **Deploy Hook** → copia la URL.
2. En GitHub: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Name | Secret |
|---|---|
| `RENDER_DEPLOY_HOOK_URL` | El deploy hook de la API |
| `RENDER_DEPLOY_HOOK_URL_GRADIO` | El deploy hook de Gradio |

> ⚠️ Un deploy hook es una **contraseña**: quien tenga la URL puede redesplegar tu servicio.
> Nunca va dentro del código. Si se filtra, en Render puedes regenerarla.

> 📁 Los workflows (`.github/workflows/`) **tienen que estar en la raíz del repositorio** para que
> GitHub los encuentre. Si `mnist_mlops/` es una subcarpeta de un repo más grande, hay que moverlos
> a la raíz y agregar `working-directory: mnist_mlops` en el CI. Lo más simple: un repositorio
> propio para este proyecto.

### Paso 4: Comprobarlo

Haz un cambio pequeño (por ejemplo, el `message` de `GET /` en `api.py`), súbelo y observa:

1. **GitHub → Actions**: CI ✅ → CD ✅.
2. **Render → Events**: aparece un despliegue nuevo en cada servicio.
3. Abre primero la URL de la API (para despertarla) y luego la de Gradio, y dibuja un dígito.

---

## ⏪ Volver a una versión anterior (rollback)

**Render → Events** → busca el despliegue anterior → **Rollback**.

---

## 🧯 Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| El workflow CD no arranca | El CI falló, o el push no fue a `main` | El CD solo corre tras un CI ✅ en `main` |
| CD falla con `curl: (3) URL rejected` | Falta un secreto de deploy hook | Paso 3 |
| Render despliega aunque el CI falle | **Auto-Deploy** quedó encendido | Settings → **Auto-Deploy: Off** |
| El build falla en `entrenar` | El modelo no superó `MIN_ACCURACY` | La puerta de calidad funcionó: revisa los Logs |
| El build falla al descargar MNIST | Un servidor de MNIST no respondió | Vuelve a desplegar (**Manual Deploy**). torchvision prueba varios servidores |
| El build tarda mucho | Primera vez: descarga torch y entrena en CPU | Normal. Los siguientes reutilizan capas en caché (cuando Render la conserva) |
| Gradio: *No se pudo conectar con la API* | La API está dormida o `API_URL` está mal | Abre primero la URL de la API; revisa `API_URL` |
| Error *Out of memory* en la API | Alguien agregó torch a las dependencias de `[project]` | torch va solo en el grupo `train`. El paso 8 del CI lo detecta |
