# Entorno Docker para Fraud Detection Handbook

Este documento describe cómo ejecutar los notebooks unificados del Fraud Detection Handbook usando Docker.

## Requisitos Previos

- Docker instalado y funcionando
- Docker Compose instalado (incluido en Docker Desktop)

## Uso Rápido

### 1. Construir la imagen Docker

```bash
docker-compose build
```

### 2. Ejecutar los notebooks

```bash
docker-compose up
```

Esto ejecutará automáticamente los notebooks unificados:
- `Chapter_6_ImbalancedLearning/Chapter_6_Unified.ipynb`
- `Chapter_7_DeepLearning/Chapter_7_Unified.ipynb`

### 3. Ver resultados

Los resultados se guardan en la carpeta `execution_results/`:
- Notebooks ejecutados (con outputs)
- Reportes en formato JSON y TXT

## Desarrollo Interactivo

Para trabajar interactivamente con Jupyter Lab:

1. Editar `docker-compose.yml` y descomentar las líneas de `ports` y `command`
2. Ejecutar:
```bash
docker-compose up
```
3. Acceder a Jupyter Lab en: http://localhost:8888

## Estructura del Proyecto

```
.
├── Dockerfile                 # Imagen Docker con todas las dependencias
├── docker-compose.yml         # Configuración del entorno
├── execute_notebooks.py       # Script de ejecución automática
├── requirements.txt           # Dependencias de Python
└── execution_results/         # Resultados de ejecución (generado)
```

## Notas

- Los notebooks se ejecutan con un timeout de 10 minutos por celda
- Los errores se capturan y documentan en los reportes
- Los notebooks ejecutados se guardan con timestamp para mantener historial
