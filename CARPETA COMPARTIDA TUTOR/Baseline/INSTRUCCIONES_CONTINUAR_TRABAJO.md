# 📋 Instrucciones para Continuar el Trabajo (Post-Crash)

## Situación Actual
- Cursor crashea por OOM al trabajar con Chapter 7
- Notebook tiene errores pendientes por corregir
- La solución es trabajar fuera de Cursor o con versión estable

---

## 🔧 PASO 1: Preparar Entorno Estable (5 min)

### Opción A: Usar Cursor 0.46.0 (Recomendado - Más estable)
1. Descarga: https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/aff57e1d9a74ed627fb5bd393e347079514436a7/win32/x64/user-setup/CursorUserSetup-x64-0.46.0.exe
2. Desinstala Cursor actual
3. Instala 0.46.0
4. Desactiva actualizaciones automáticas en Settings

### Opción B: Trabajar sin Cursor (100% estable)
- Usa VS Code + terminal PowerShell para ejecutar
- O ejecuta directamente desde PowerShell sin IDE

---

## 📝 PASO 2: Corregir Errores Pendientes en Chapter 7

### Error 1: `np.Inf` → `np.inf` (3 ocurrencias)
**Archivo:** `Chapter_7_DeepLearning/Chapter_7_Unified.ipynb`

Busca y reemplaza:
```python
self.best_score = np.Inf
```
Por:
```python
self.best_score = np.inf
```

Líneas aproximadas: 1740, 2851, 3468

### Error 2: Comentar `curl` que sobrescribe funciones (5 ocurrencias)
**Peligro:** El `curl` descarga la versión antigua con `np.Inf`

Busca celdas que contengan:
```python
!curl -O https://raw.githubusercontent.com/.../shared_functions.py
```

Cámbialas a:
```python
# !curl -O https://raw.githubusercontent.com/.../shared_functions.py  # Usar versión local
```

### Error 3: Corregir `ma_window` (2 ocurrencias)
**Archivo:** `Chapter_7_DeepLearning/Chapter_7_Unified.ipynb`

Busca celdas con:
```python
ma_window = 10
plt.plot(np.arange(len(epochs_train_losses)-ma_window + 1)+1, ...)
```

Cámbialas a:
```python
ma_window = min(10, len(epochs_train_losses), len(epochs_test_losses)) or 1
conv_train = np.convolve(epochs_train_losses, np.ones(ma_window)/ma_window, mode='valid')
conv_test = np.convolve(epochs_test_losses, np.ones(ma_window)/ma_window, mode='valid')
x = np.arange(len(conv_train))+1
plt.plot(x, conv_train)
plt.plot(x, conv_test)
```

---

## 🚀 PASO 3: Ejecutar Chapter 7 (30-60 min)

### Pre-ejecución
```powershell
cd "C:<Ruta>aseline"
wsl --shutdown  # Libera RAM de WSL
docker compose down --remove-orphans  # Limpia contenedores
```

### Ejecución (ELIGE UNA)

**Opción 1 - Doble clic (más fácil):**
```
ejecutar_ch7_sin_cursor.bat
```

**Opción 2 - PowerShell:**
```powershell
.\run_ch7_externo.ps1
```

**Opción 3 - Docker directo:**
```powershell
docker compose run --rm ch7-gpu
```

### Monitoreo
Revisa progreso en tiempo real:
```
execution_progress.txt
```

---

## 🐛 PASO 4: Si Hay Errores Durante Ejecución

### Error: `np.Inf not found`
**Solución:** Aplicar PASO 2 - Error 1

### Error: `ma_window` dimension mismatch
**Solución:** Aplicar PASO 2 - Error 3

### Error: `curl` sobrescribe funciones
**Solución:** Aplicar PASO 2 - Error 2

### Error: Out of Memory (OOM) en Docker
**Solución:**
1. Edita `docker-compose.yml`
2. Añade límite de memoria:
```yaml
ch7-gpu:
  deploy:
    resources:
      limits:
        memory: 8G
```

---

## ✅ Checklist Pre-Ejecución

- [ ] Cursor 0.46.0 instalado (o VS Code, o ningún IDE)
- [ ] `np.Inf` corregido a `np.inf` en notebook
- [ ] `curl` comentado en notebook
- [ ] `ma_window` corregido en notebook
- [ ] Docker Desktop ejecutándose
- [ ] GPU disponible (opcional, funciona con CPU)
- [ ] WSL reiniciado (`wsl --shutdown`)

---

## 📊 Tiempo Estimado

| Tarea | Tiempo |
|-------|--------|
| Preparar entorno | 5 min |
| Corregir errores | 10 min |
| Ejecutar Chapter 7 | 30-60 min |
| **Total** | **45-75 min** |

---

## 🆘 Emergencia: Si Todo Falla

1. **Instala Cursor 0.46.0** (enlace arriba)
2. **Abre solo el notebook** sin ejecutar
3. **Corrige manualmente** los 3 errores conocidos
4. **Cierra Cursor**
5. **Ejecuta desde PowerShell:** `docker compose run --rm ch7-gpu`
6. **No abras Cursor** hasta que termine

---

## 🎯 Resultado Esperado

Al finalizar correctamente:
- Archivo: `execution_results/unified_report_*.json`
- Archivo: `execution_results/unified_report_*.txt`
- Notebook ejecutado: 484/484 celdas

---

**Documento creado:** 2025-02-22
**Versión:** 1.0 - Post-OM Crisis
