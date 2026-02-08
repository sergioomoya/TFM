# Instrucciones para Traducción con LLM

## Estado Actual

Se ha iniciado la traducción de los cuadernos unificados usando un LLM para traducciones más naturales y contextuales. Se han traducido algunas celdas clave como demostración del enfoque.

## Scripts Disponibles

1. **`translate_with_openai.py`**: Script que usa la API de OpenAI para traducir. Requiere:
   - Instalar: `pip install openai`
   - Configurar variable de entorno: `OPENAI_API_KEY`

2. **`translate_direct_llm.py`**: Script base para traducción directa con LLM (estructura preparada).

## Proceso Recomendado

Para completar la traducción de todos los cuadernos usando un LLM:

### Opción 1: Usar OpenAI API (Recomendado)

1. Configurar API key:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="tu-api-key-aqui"
   ```

2. Ejecutar el script:
   ```bash
   python translate_with_openai.py
   ```

### Opción 2: Procesamiento Manual con LLM

Las celdas markdown y comentarios pueden procesarse manualmente usando un LLM (como este asistente) para traducir cada celda de manera contextual.

## Notas

- Las traducciones automáticas anteriores (con diccionarios) han sido parcialmente corregidas
- El contenido markdown extenso requiere traducción contextual para mantener el significado técnico
- Se preservan automáticamente: código, enlaces, nombres de variables/funciones

## Próximos Pasos

1. Completar traducción de todas las celdas markdown en los 5 cuadernos unificados
2. Traducir comentarios en código de manera coherente
3. Revisar traducciones para consistencia terminológica
