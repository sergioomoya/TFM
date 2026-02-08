# Dockerfile para ejecutar los notebooks del Fraud Detection Handbook
# Basado en Python 3.8 para compatibilidad con las versiones específicas de las dependencias

FROM python:3.8-slim

# Metadatos
LABEL maintainer="TFM Fraud Detection Handbook"
LABEL description="Entorno Docker para ejecutar notebooks unificados del Fraud Detection Handbook"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install nbconvert papermill jupyter-client ipykernel

# Copiar todo el proyecto
COPY . .

# Exponer puerto para Jupyter (opcional, para desarrollo)
EXPOSE 8888

# Comando por defecto (puede ser sobrescrito)
CMD ["python", "execute_notebooks.py"]
