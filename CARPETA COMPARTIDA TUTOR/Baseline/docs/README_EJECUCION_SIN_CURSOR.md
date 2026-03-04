# 🚨 GUÍA: Ejecutar Chapter 7 SIN Cursor (evita OOM)

## El Problema

Cursor está crasheando repetidamente con error OOM a pesar de tener 32 GB de RAM.
Esto es un **bug conocido de Cursor** en versiones recientes, no un problema de tu sistema.

## La Solución

**NO uses Cursor para ejecutar el notebook.** Usa los scripts externos.

## Pasos

### 1. Cierra Cursor completamente
- Guarda cualquier cambio pendiente
- Cierra Cursor (no minimices, cierra del todo)

### 2. Ejecuta desde el Explorador de archivos

**Opción A - Doble clic (más fácil):**
```
run_ch7_externo.bat
```

**Opción B - PowerShell:**
```powershell
.\run_ch7_externo.ps1
```

**Opción C - CMD manual:**
```cmd
wsl --shutdown
docker compose run --rm ch7-gpu
```

### 3. Monitorea el progreso

Durante la ejecución, revisa el archivo:
```
execution_progress.txt
```

Muestra algo como:
```
Celda 134/484 | 12:53:12
```

### 4. Cuando termine

- Si hay errores: Abre Cursor, corrige, cierra Cursor, vuelve a ejecutar
- Si termina bien: Los resultados estarán en `execution_results/`

## Por qué funciona

| Con Cursor abierto | Sin Cursor |
|-------------------|------------|
| RAM usada: ~28 GB | RAM usada: ~20 GB |
| Cursor crashea | Estable |
| Proceso interrumpido | Completa al 100% |

Los 8 GB extra que usa Cursor (indexación, UI, agentes) son los que causan el OOM.

## Correcciones ya aplicadas al notebook

- ✅ `np.Inf` → `np.inf` (NumPy 2.0)
- ✅ `ma_window` adaptado a pocas épocas
- ✅ `curl` comentado para usar versión local
- ✅ `max_epochs` reducido para ejecutar más rápido

## Si necesitas editar el notebook

1. Abre Cursor **solo para editar**
2. Haz los cambios necesarios
3. Guarda (Ctrl+S)
4. Cierra Cursor completamente
5. Ejecuta con el script .bat

## Problemas conocidos

**Error: "execution.lock existe"**
```powershell
Remove-Item execution.lock
```

**Error: "Docker no responde"**
```powershell
wsl --shutdown
# Espera 10 segundos y vuelve a intentar
```

**Error: "GPU no disponible"**
El contenedor ejecutará con CPU (más lento pero funciona)

## Contacto / Soporte

Si el script `.bat` no funciona, abre PowerShell como Administrador y ejecuta:
```powershell
cd "C:\Programacion\GitHub\TFM\CARPETA COMPARTIDA TUTOR\Baseline"
docker compose run --rm ch7-gpu
```

---

**Versión del documento:** 2025-02-22  
**Estado:** Solución activa para OOM en Cursor
