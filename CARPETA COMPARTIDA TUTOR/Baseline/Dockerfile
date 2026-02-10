# =============================================================================
# Dockerfile - TFM Detección de Fraude
# =============================================================================
# Entorno Docker para ejecutar los experimentos y cuadernos unificados
# del Fraud Detection Handbook.
# =============================================================================

FROM python:3.8-slim

LABEL maintainer="TFM Detección de Fraude - Sergio Moya"
LABEL description="Entorno Docker para los experimentos del TFM"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install nbconvert papermill jupyter-client ipykernel lxml_html_clean jupyterlab && \
    python -m ipykernel install --user --name python3

COPY . .

# Puerto Jupyter (para desarrollo interactivo)
EXPOSE 8888

# Por defecto: shell interactiva (sobrescribir con docker-compose)
CMD ["bash"]
