# Cómo conseguir Cursor stable con Composer 1.5

## 🚨 SOLUCIÓN RÁPIDA: Ejecuta desde fuera de Cursor

La mejor forma de evitar el crash por OOM:

1. **Cierra Cursor completamente**
2. Ejecuta desde PowerShell/CMD sin abrir Cursor:
   ```powershell
   .\run_ch7_externo.ps1
   ```

## 📥 Opción A: Descargar Cursor 0.46.0 (más estable)

**⚠️ Importante:** Composer 1.5 requiere versiones recientes pero 0.46.0 es más estable.

**Enlace directo Windows x64:**
```
https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/aff57e1d9a74ed627fb5bd393e347079514436a7/win32/x64/user-setup/CursorUserSetup-x64-0.46.0.exe
```

**Pasos:**
1. Desinstala Cursor actual desde "Aplicaciones y características"
2. Descarga y ejecuta el instalador antiguo
3. En `Settings` > `Extensions` > `Features`, asegura que Composer esté activado
4. **No actualices** cuando te pida actualización

## 📥 Opción B: Cursor 1.7.43/1.7.44 (recientes pero más inestables)

**Cursor 1.7.44 Windows x64 (reciente):**
```
https://downloads.cursor.com/production/9d178a4a5589981b62546448bb32920a8219a5de/win32/x64/system-setup/CursorSetup-x64-1.7.44.exe
```

**Cursor 1.7.43 es más estable que 1.7.44 pero no se encuentra el enlace directo** - usa Chocolatery o 1.7.44 más arriba.

## 🔍 Cómo buscar otras versiones

1. Repo de descargas antiguas: https://github.com/oslook/cursor-ai-downloads
2. Alternativa: https://shtse8.github.io/cursor-ai-downloads/
3. Historial oficial: https://cursor.com/en/changelog

## ⚙️ Prevenir actualizaciones automáticas

1. Añade a `settings.json` en Cursor:
   ```json
   "update.mode": "none"
   ```

## 🛠️ Trabaja fuera de Cursor (RECOMENDADO)

El crash de OOM probablemente es por el notebook gigante que Cursor intenta indexar:

1. **Crea `.cursorignore`** (ya lo tenemos):
   ```
   Chapter_7_DeepLearning/Chapter_7_Unified.ipynb
   ```

2. **Ejecuta fuera de Cursor** (ya existen scripts):
   ```powershell
   wsl --shutdown
   docker compose run --rm ch7-gpu
   ```

3. **Solo abre Cursor cuando tengas que EDITAR** código, no durante ejecución larga

## 📊 Tu caso

- **RAM total:** 32 GB
- **RAM usada pico:** 20 GB
- **Conclusión:** Es un problema específico de Cursor, no falta de RAM del sistema

## ✔️ Conclusión

**Paso 1:** Ejecuta el cuaderno FUERA de Cursor usando los scripts externos
**Paso 2:** Sólo abre Cursor para edición puntual
**Paso 3:** Si necesitas usarlo largo, considera Cursor 0.46.0 más estable

Con esto evitarás los seguros por OOM y podrás ejecutar el cuaderno completo sin que Cursor crashee.