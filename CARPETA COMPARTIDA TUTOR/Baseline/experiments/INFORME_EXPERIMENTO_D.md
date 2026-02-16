# Informe del Experimento D: Interpretabilidad (XAI)

**Estado:** Implementado y Ejecutado
**Ubicación:** `experiments/experiment_d_interpretability.ipynb`

## 1. Introducción
Los modelos de "Caja Negra" como XGBoost o Redes Neuronales ofrecen alto rendimiento pero baja explicabilidad. En el sector financiero, es obligatorio justificar por qué se bloquea una transacción (regulaciones como GDPR). Este experimento aplica técnicas de Inteligencia Artificial Explicable (XAI) para abrir la caja negra.

## 2. Metodología

Se utiliza el mejor modelo obtenido en el Experimento A/B (XGBoost) y se aplican dos técnicas:

### 2.1. Feature Importance (Global)
Métrica intrínseca de los árboles de decisión que mide cuántas veces se usa una característica para dividir nodos y cuánto reduce la impureza (Gini/Entropía).
- **Limitación:** Puede estar sesgada hacia variables de alta cardinalidad y no indica la dirección del efecto (positivo/negativo).

### 2.2. SHAP (SHapley Additive exPlanations)
Método agnóstico del modelo basado en la teoría de juegos.
- **Valores SHAP:** Asignan a cada característica un valor de contribución para *cada* predicción individual.
- **Análisis Local:** Explica por qué la transacción #12345 fue marcada como fraude.
- **Análisis Global:** Agregando los valores SHAP absolutos se obtiene la importancia global real.

## 3. Resultados

### 3.1. Top Características (Feature Importance)
Las variables más influyentes consistentemente son:
1.  `CUSTOMER_ID_NB_TX_7DAY_WINDOW`: Frecuencia transaccional reciente.
2.  `CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW`: Monto promedio reciente.
3.  `TERMINAL_ID_RISK_7DAY_WINDOW`: Riesgo histórico del terminal.

### 3.2. Insights de SHAP
- **Direccionalidad:** SHAP revela que valores altos de `NB_TX_1DAY_WINDOW` (muchas transacciones hoy) aumentan drásticamente la probabilidad de fraude (valor SHAP positivo alto).
- **Interacciones:** Se observa que `TX_AMOUNT` alto solo es riesgo si se combina con un terminal de riesgo o un comportamiento anómalo del cliente. Por sí solo, un monto alto no siempre es fraude.

## 4. Conclusiones
- La explicabilidad valida que el modelo está aprendiendo patrones de negocio lógicos y no ruido.
- Las variables agregadas (RFM) son el motor predictivo del modelo.
- SHAP permite generar "Códigos de Razón" para los analistas de fraude, facilitando la investigación manual de las alertas generadas.
