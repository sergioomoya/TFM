# Guía de Contribución y Desarrollo

¡Bienvenido al equipo de desarrollo del TFM! Este documento establece las reglas y flujos de trabajo obligatorios para garantizar la calidad, reproducibilidad y orden del proyecto.

## 1. Principios Fundamentales

### 🐳 Docker First (Entorno Aislado)
- **PROHIBIDO** instalar dependencias o ejecutar código directamente en tu máquina local (host).
- Todo debe ejecutarse a través de `docker compose`.
- Si necesitas una nueva librería, añádela a `requirements.txt` y reconstruye la imagen:
  ```bash
  docker compose build
  ```

### 🌳 Git Flow (Control de Versiones)
- **Rama `main`:** Solo contiene código estable y probado. **Nunca** hagas commit directo aquí.
- **Ramas de Feature/Fix:** Crea una rama para cada tarea:
  - `feat/nombre-funcionalidad` (ej: `feat/experimento-b`)
  - `fix/descripcion-error` (ej: `fix/ruta-datos`)
- **Commits:** Usa [Conventional Commits](https://www.conventionalcommits.org/) (ej: `feat: añadir cálculo de CP@100`, `docs: actualizar readme`).

### 🧪 Reproducibilidad
- Usa siempre semillas aleatorias fijas (`random_state=42`) en los experimentos.
- No subas archivos de datos pesados (`.csv`, `.pkl`) ni notebooks ejecutados con salidas masivas al repositorio. El `.gitignore` ya está configurado para esto.

## 2. Flujo de Trabajo Diario (Sprints)

1.  **Inicio de Tarea:**
    - Asegúrate de estar en `main` y actualizado: `git checkout main && git pull`.
    - Crea tu rama: `git checkout -b feat/mi-nueva-tarea`.

2.  **Desarrollo:**
    - Levanta el entorno: `docker compose up jupyter`.
    - Trabaja en los notebooks o scripts.
    - Verifica que tu código cumple con PEP8 (el linter te avisará).

3.  **Finalización:**
    - Limpia las salidas de los notebooks si son irrelevantes para el commit, o asegúrate de que aportan valor (como gráficas de resultados).
    - Ejecuta los tests o validaciones pertinentes.
    - Haz Push y crea un Pull Request (o merge local si trabajas solo).

## 3. Estructura de Documentación
- **`/sprints`:** Documentación viva del proyecto. Al iniciar un sprint, crea `Sprint_XX_Plan.md`. Al cerrar, `Sprint_XX_Retro.md`.
- **`README.md`:** Punto de entrada para cualquier usuario nuevo. Mantenlo actualizado.

## 4. Gestión de Errores
Si encuentras un error, no lo parches a ciegas.
1.  **Reproduce** el error en el entorno Docker.
2.  **Instrumenta** (añade logs/prints) para entender la causa.
3.  **Corrige** y verifica.
