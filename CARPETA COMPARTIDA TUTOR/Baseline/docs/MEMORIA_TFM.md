![](docs/media/media/image1.jpeg){width="8.965517279090113in" height="12.686896325459317in"}**Detección de Fraude en Tarjetas de Crédito: Un Enfoque Metodológico Robusto mediante Gestión del Desbalance y Explicabilidad (XAI)**

+--------------------------------------------------+-----------------------------+----------------------+
| Titulación:                                      | Alumno/a:                   | Convocatoria:        |
|                                                  |                             |                      |
| MÁSTER EN INGENIERÍA DE TECNOLOGÍAS INDUSTRIALES | MOYA MARÍN, SERGIO          | Primera              |
|                                                  |                             |                      |
| Curso académico:                                 | D.N.I: 21700500T            |                      |
|                                                  |                             |                      |
| 2025 -- 2026                                     | Director/a de TFT:          |                      |
|                                                  |                             |                      |
|                                                  | BADY PATRICIO GANA CASTILLO |                      |
+==================================================+=============================+======================+

30 marzo 2026

Índice

[**Índice de figuras y gráficos** [5](#índice-de-figuras-y-gráficos)](#índice-de-figuras-y-gráficos)

[**Índice de tablas** [7](#índice-de-tablas)](#índice-de-tablas)

[**Glosario** [8](#glosario)](#glosario)

[**Resumen** [10](#resumen)](#resumen)

[**Abstract** [10](#abstract)](#abstract)

[**1** **Introducción** [11](#introducción)](#introducción)

[1.1 Contexto y Problemática [11](#contexto-y-problemática)](#contexto-y-problemática)

[1.2 Preguntas de Investigación [14](#preguntas-de-investigación)](#preguntas-de-investigación)

[1.3 Objetivos del Estudio [14](#objetivos-del-estudio)](#objetivos-del-estudio)

[1.4 Hipótesis [15](#hipótesis)](#hipótesis)

[1.5 Contribuciones Principales del Trabajo [15](#contribuciones-principales-del-trabajo)](#contribuciones-principales-del-trabajo)

[**2** **Estado del arte o Marco Teórico** [17](#estado-del-arte-o-marco-teórico)](#estado-del-arte-o-marco-teórico)

[2.1 Naturaleza del Problema: Desbalance Extremo y Concept Drift [17](#naturaleza-del-problema-desbalance-extremo-y-concept-drift)](#naturaleza-del-problema-desbalance-extremo-y-concept-drift)

[2.2 Paradigmas de Aprendizaje: Clasificación Supervisada vs. Detección de Anomalías [18](#paradigmas-de-aprendizaje-clasificación-supervisada-vs.-detección-de-anomalías)](#paradigmas-de-aprendizaje-clasificación-supervisada-vs.-detección-de-anomalías)

[2.3 La Problemática de la Validación y el Data Leakage [19](#la-problemática-de-la-validación-y-el-data-leakage)](#la-problemática-de-la-validación-y-el-data-leakage)

[2.4 Inteligencia Artificial Explicable (XAI): Superando la Caja Negra [20](#inteligencia-artificial-explicable-xai-superando-la-caja-negra)](#inteligencia-artificial-explicable-xai-superando-la-caja-negra)

[**3** **Metodología** [21](#metodología)](#metodología)

[3.1 Marco Metodológico de Referencia [21](#marco-metodológico-de-referencia)](#marco-metodológico-de-referencia)

[3.1.1 Origen y Contexto del Estudio Base [22](#origen-y-contexto-del-estudio-base)](#origen-y-contexto-del-estudio-base)

[3.1.2 Pipeline de Referencia: Ingeniería de Características RFM [22](#pipeline-de-referencia-ingeniería-de-características-rfm)](#pipeline-de-referencia-ingeniería-de-características-rfm)

[3.1.3 Estrategia de Validación Temporal [23](#estrategia-de-validación-temporal)](#estrategia-de-validación-temporal)

[3.2 Adaptaciones y Contribuciones Metodológicas Propias [25](#adaptaciones-y-contribuciones-metodológicas-propias)](#adaptaciones-y-contribuciones-metodológicas-propias)

[3.2.1 Refinamiento del Protocolo Experimental [25](#refinamiento-del-protocolo-experimental)](#refinamiento-del-protocolo-experimental)

[3.2.2 Implementación de Integridad Metodológica (Test Anti-Leakage) [26](#implementación-de-integridad-metodológica-test-anti-leakage)](#implementación-de-integridad-metodológica-test-anti-leakage)

[3.2.3 Incorporación de Explicabilidad (XAI) [27](#incorporación-de-explicabilidad-xai)](#incorporación-de-explicabilidad-xai)

[3.3 Diseño Experimental [28](#diseño-experimental)](#diseño-experimental)

[3.3.1 Descripción del Conjunto de Datos [28](#descripción-del-conjunto-de-datos)](#descripción-del-conjunto-de-datos)

[3.3.2 Configuración del Entorno de Ejecución [29](#configuración-del-entorno-de-ejecución)](#configuración-del-entorno-de-ejecución)

[3.3.3 Definición de Métricas de Evaluación [30](#definición-de-métricas-de-evaluación)](#definición-de-métricas-de-evaluación)

[3.4 Desarrollo de la fase Experimental [30](#desarrollo-de-la-fase-experimental)](#desarrollo-de-la-fase-experimental)

[3.4.1 Definición del Modelo de Referencia (Baseline) [31](#definición-del-modelo-de-referencia-baseline)](#definición-del-modelo-de-referencia-baseline)

[3.4.2 Validación de Integridad Metodológica (Test Anti-Leakage) [33](#validación-de-integridad-metodológica-test-anti-leakage)](#validación-de-integridad-metodológica-test-anti-leakage)

[3.4.3 Análisis de Interpretabilidad Algorítmica (Explainable AI) [34](#análisis-de-interpretabilidad-algorítmica-explainable-ai)](#análisis-de-interpretabilidad-algorítmica-explainable-ai)

[**4** **Discusión y Análisis de Resultados.** [36](#discusión-y-análisis-de-resultados.)](#discusión-y-análisis-de-resultados.)

[4.1 Definición del Modelo de Referencia [36](#definición-del-modelo-de-referencia)](#definición-del-modelo-de-referencia)

[4.1.1 Análisis de Rendimiento bajo Desbalance Natural [36](#análisis-de-rendimiento-bajo-desbalance-natural)](#análisis-de-rendimiento-bajo-desbalance-natural)

[4.1.2 Comparativa Algorítmica Estándar [37](#comparativa-algorítmica-estándar)](#comparativa-algorítmica-estándar)

[4.1.3 Selección y Justificación del Modelo Base [38](#selección-y-justificación-del-modelo-base)](#selección-y-justificación-del-modelo-base)

[4.1.4 Refinamiento del Modelo Base: Estrategia de Submuestreo (Undersampling) [39](#refinamiento-del-modelo-base-estrategia-de-submuestreo-undersampling)](#refinamiento-del-modelo-base-estrategia-de-submuestreo-undersampling)

[4.2 Impacto de la Fuga de Datos (*Anti-Leakage*) [40](#impacto-de-la-fuga-de-datos-anti-leakage)](#impacto-de-la-fuga-de-datos-anti-leakage)

[4.2.1 Evaluación bajo Metodología Deficiente (El Espejismo) [41](#evaluación-bajo-metodología-deficiente-el-espejismo)](#evaluación-bajo-metodología-deficiente-el-espejismo)

[4.2.2 Evaluación bajo Metodología Estricta (La Realidad) [42](#evaluación-bajo-metodología-estricta-la-realidad)](#evaluación-bajo-metodología-estricta-la-realidad)

[4.2.3 Cuantificación de la Divergencia Algorítmica (Δ) [42](#cuantificación-de-la-divergencia-algorítmica-δ)](#cuantificación-de-la-divergencia-algorítmica-δ)

[4.3 Interpretabilidad Algorítmica (XAI) [43](#interpretabilidad-algorítmica-xai)](#interpretabilidad-algorítmica-xai)

[4.3.1 Identificación Global de Patrones (Importancia Nativa) [44](#identificación-global-de-patrones-importancia-nativa)](#identificación-global-de-patrones-importancia-nativa)

[4.3.2 Direccionalidad del Riesgo Algorítmico (SHAP) [45](#direccionalidad-del-riesgo-algorítmico-shap)](#direccionalidad-del-riesgo-algorítmico-shap)

[4.3.3 Auditoría Forense a Nivel de Transacción (Fuerza Local) [47](#auditoría-forense-a-nivel-de-transacción-fuerza-local)](#auditoría-forense-a-nivel-de-transacción-fuerza-local)

[4.3.4 Validación de Relevancia mediante Estudio de Ablación [48](#validación-de-relevancia-mediante-estudio-de-ablación)](#validación-de-relevancia-mediante-estudio-de-ablación)

[**5** **Conclusiones** [51](#conclusiones)](#conclusiones)

[5.1 Síntesis de Hallazgos y Validación de Hipótesis [51](#síntesis-de-hallazgos-y-validación-de-hipótesis)](#síntesis-de-hallazgos-y-validación-de-hipótesis)

[**La falacia de las métricas generalistas y el valor del submuestreo:** [51](#la-falacia-de-las-métricas-generalistas-y-el-valor-del-submuestreo)](#la-falacia-de-las-métricas-generalistas-y-el-valor-del-submuestreo)

[**Cuantificación empírica del Data Leakage en la literatura:** [51](#cuantificación-empírica-del-data-leakage-en-la-literatura)](#cuantificación-empírica-del-data-leakage-en-la-literatura)

[**La transición hacia la \"Caja Blanca\" validada:** [51](#la-transición-hacia-la-caja-blanca-validada)](#la-transición-hacia-la-caja-blanca-validada)

[5.2 Implicaciones para el Negocio Bancario [52](#implicaciones-para-el-negocio-bancario)](#implicaciones-para-el-negocio-bancario)

[5.3 Limitaciones del Estudio [52](#limitaciones-del-estudio)](#limitaciones-del-estudio)

[**La brecha semántica de la simulación de datos:** [52](#la-brecha-semántica-de-la-simulación-de-datos)](#la-brecha-semántica-de-la-simulación-de-datos)

[**Vulnerabilidad ante ataques emergentes (Zero-Day):** [52](#vulnerabilidad-ante-ataques-emergentes-zero-day)](#vulnerabilidad-ante-ataques-emergentes-zero-day)

[**Coste Computacional de la Explicabilidad en Tiempo Real:** [53](#coste-computacional-de-la-explicabilidad-en-tiempo-real)](#coste-computacional-de-la-explicabilidad-en-tiempo-real)

[**El sesgo de la Explicabilidad (XAI) sobre reglas sintéticas:** [53](#el-sesgo-de-la-explicabilidad-xai-sobre-reglas-sintéticas)](#el-sesgo-de-la-explicabilidad-xai-sobre-reglas-sintéticas)

[**Restricción del espacio de características (Feature Space):** [53](#restricción-del-espacio-de-características-feature-space)](#restricción-del-espacio-de-características-feature-space)

[5.4 Líneas de Investigación Futuras [53](#líneas-de-investigación-futuras)](#líneas-de-investigación-futuras)

[**Análisis de Sensibilidad sobre la Latencia de Etiquetado:** [53](#análisis-de-sensibilidad-sobre-la-latencia-de-etiquetado)](#análisis-de-sensibilidad-sobre-la-latencia-de-etiquetado)

[**Transición hacia Arquitecturas de Deep Learning Secuencial:** [54](#transición-hacia-arquitecturas-de-deep-learning-secuencial)](#transición-hacia-arquitecturas-de-deep-learning-secuencial)

[**Análisis Topológico mediante Grafos (Graph Neural Networks):** [54](#análisis-topológico-mediante-grafos-graph-neural-networks)](#análisis-topológico-mediante-grafos-graph-neural-networks)

[**6** **Referencias** [55](#referencias)](#referencias)

[Anexos [59](#anexos)](#anexos)

**\**

# **Índice de figuras y gráficos**

[Ilustración 1. Evolución histórica de las pérdidas globales por fraude en tarjetas de crédito, destacando la predominancia y el crecimiento sostenido del vector no presencial (Card-Not-Present). Fuente: The Nilson Report. [11](#_Toc225188150)](#_Toc225188150)

[Ilustración 2. Arquitectura de un Sistema de Detección de Fraude (FDS) en el sector financiero, diferenciando procesos en tiempo real, casi en tiempo real y offline. Fuente: Dal Pozzolo et al. (2014). [12](#_Toc225188151)](#_Toc225188151)

[Ilustración 3. Representación espacial del desbalance extremo en el conjunto de datos, evidenciando el solapamiento de la clase minoritaria (fraude) sobre la densidad de la clase legítima. Fuente: Elaboración propia [17](#_Toc225188152)](#_Toc225188152)

[Ilustración 4. Tipologías de Concept Drift (cambio de concepto) en flujos de datos a lo largo del tiempo. Fuente: Gama et al. (2014). [18](#_Toc225188153)](#_Toc225188153)

[Ilustración 5. Flujo de trabajo de referencia para la ingeniería de características y validación de modelos en detección de fraude. Fuente: Le Borgne et al. (2022). [23](#_Toc225188154)](#_Toc225188154)

[Ilustración 6. Estrategia de evaluación prequencial en bloques con periodo de latencia (delay) para evitar fugas de información. Fuente: Elaboración propia basada en Le Borgne et al. (2022). [24](#_Toc225188155)](#_Toc225188155)

[Ilustración 7. Matrices de confusión de los algoritmos evaluados (Regresión Logística, Random Forest y XGBoost) evidenciando las altas tasas de Falsos Negativos. Fuente: Elaboración propia. [36](#_Toc225188156)](#_Toc225188156)

[Ilustración 8. Comparativa gráfica de rendimiento técnico (AUC ROC, AUPRC) y métrica de negocio (Card Precision@100) para los modelos de referencia. Fuente: Elaboración propia. [38](#_Toc225188157)](#_Toc225188157)

[Ilustración 9. Consecuencias operativas de aplicar remuestreo sintético (SMOTE) globalmente antes de la partición temporal, provocando Data Leakage. Fuente: Adaptado de Demircioğlu (2024). [41](#_Toc225188158)](#_Toc225188158)

[Ilustración 10. Cuantificación de la inflación artificial de la métrica AUPRC (Data Leakage) por fuente de error y modelo algorítmico. Fuente: Elaboración propia. [42](#_Toc225188159)](#_Toc225188159)

[Ilustración 11. Jerarquía de las diez variables más importantes (métrica Gain) para el modelo XGBoost base. Fuente: Elaboración propia. [44](#_Toc225188160)](#_Toc225188160)

[Ilustración 12. Gráfico SHAP Beeswarm de explicabilidad global para el modelo XGBoost, mostrando la magnitud y direccionalidad del impacto predictivo. Fuente: Elaboración propia. [45](#_Toc225188161)](#_Toc225188161)

[Ilustración 13. Explicabilidad local mediante gráficos SHAP Waterfall, desglosando la contribución marginal de variables para un fraude detectado y una transacción legítima. Fuente: Elaboración propia. [47](#_Toc225188162)](#_Toc225188162)

[Ilustración 14. Códigos de razón (reason codes) simulados en un SOC mediante gráficos SHAP Force Plot para una transacción fraudulenta y una legítima. Fuente: Elaboración propia. [48](#_Toc225188163)](#_Toc225188163)

[Ilustración 15. Reconfiguración de la jerarquía de explicabilidad (SHAP Beeswarm) tras el estudio de ablación de la característica dominante. Fuente: Elaboración propia. [49](#_Toc225188164)](#_Toc225188164)

# **Índice de tablas**

[Tabla 1. Resultados comparativos de las métricas de evaluación técnica y operativa para los modelos base. [37](#_Toc225188165)](#_Toc225188165)

[Tabla 2. Impacto de las estrategias de submuestreo en el rendimiento del modelo óptimo. [39](#_Toc225188166)](#_Toc225188166)

[Tabla 3. Incremento marginal absoluto de la métrica AUPRC provocado por las distintas fuentes de fuga de datos. [43](#_Toc225188167)](#_Toc225188167)

[Tabla 4. Top-5 de variables predictivas según la métrica nativa de Ganancia de Impureza (Gain). [44](#_Toc225188168)](#_Toc225188168)

[Tabla 5. Impacto empírico en el rendimiento algorítmico tras la ablación de la característica dominante. [49](#_Toc225188169)](#_Toc225188169)

**\**

# **Glosario**

- **Accuracy (Exactitud):** Métrica de evaluación generalista que mide la proporción de predicciones correctas sobre el total de casos. En problemas de fraude financiero, resulta matemáticamente engañosa (Paradoja de la Exactitud) debido al desbalance extremo de clases.

- **AUC-ROC (Área Bajo la Curva ROC):** Métrica que evalúa la capacidad de discriminación global de un clasificador frente a diferentes umbrales. Aunque es estándar en la literatura, tiende a inflar el rendimiento en conjuntos de datos desbalanceados debido al peso de los Verdaderos Negativos.

- **AUPRC (Área Bajo la Curva Precision-Recall):** Métrica crítica de evaluación que pondera la Precisión y la Sensibilidad (Recall). Al aislar la clase minoritaria y penalizar severamente los Falsos Positivos, se consolida como el indicador técnico más robusto para la detección de fraude.

- **Card Precision@k (CP@k):** Métrica de negocio orientada a la viabilidad operativa. Cuantifica la proporción de transacciones fraudulentas reales identificadas dentro de las *k* alertas de mayor riesgo diario (ej. top 100), simulando la capacidad de revisión manual finita de un equipo de analistas.

- **Concept Drift (Deriva de Concepto o Cambio de Concepto):** Fenómeno estadístico dinámico por el cual la distribución subyacente de los datos y los patrones de ataque cambian con el paso del tiempo, provocando la obsolescencia prematura de los modelos de detección estáticos.

- **Data Leakage (Fuga de Datos o Información):** Error metodológico grave en el diseño experimental de *Machine Learning*. Ocurre cuando un modelo accede, durante su fase de entrenamiento, a información del futuro o del conjunto de prueba (ej. mediante *Random Split* o sobremuestreo global), inflando artificialmente sus métricas de rendimiento.

- **Ensemble Methods (Métodos de Ensamblaje):** Paradigma de aprendizaje automático que combina las predicciones de múltiples estimadores base (generalmente árboles de decisión) para mejorar la robustez y capacidad de generalización. Ejemplos aplicados: *Random Forest* y *XGBoost*.

- **FaaS (Fraud-as-a-Service):** Modelo de cibercrimen industrializado en el que organizaciones delictivas desarrollan y alquilan herramientas, infraestructura o credenciales robadas a terceros como un servicio.

- **FDS (Fraud Detection System):** Sistema Integral de Detección de Fraude en el sector financiero, habitualmente compuesto por una arquitectura híbrida que combina motores de reglas estáticas expertas y modelos predictivos impulsados por datos (*Data-Driven*).

- **Prequential Evaluation (Validación Prequencial):** Estrategia de validación temporal en flujos de datos (*data streams*) que evalúa el modelo secuencialmente en bloques cronológicos. En este estudio, incorpora un periodo de latencia (*delay*) para simular el ciclo real de reporte de fraude y bloquear el sesgo de anticipación.

- **RFM (Recency, Frequency, Monetary Value):** Técnica determinista de ingeniería de características (*Feature Engineering*) que transforma transacciones aisladas en vectores de contexto histórico, agregando el comportamiento de gasto y el riesgo acumulado en ventanas temporales específicas (ej. 1, 7 y 30 días).

- **SHAP (SHapley Additive exPlanations):** Metodología analítica basada en la Teoría de Juegos cooperativos empleada para decodificar modelos opacos. Permite calcular la contribución marginal exacta que cada variable predictiva aporta al riesgo de fraude calculado para una transacción individual.

- **SMOTE (Synthetic Minority Over-sampling Technique):** Algoritmo popular de preprocesamiento que aborda el desbalance de clases generando nuevas muestras sintéticas de la clase minoritaria mediante interpolación en el espacio de características.

- **SOC (Security Operations Center / Centro de Operaciones de Seguridad):** Departamento corporativo integrado por analistas humanos responsables de la supervisión, la auditoría continua y la resolución de incidentes o alertas críticas emitidas por el sistema automatizado.

- **Undersampling (Submuestreo):** Estrategia de gestión del desbalance orientada a reducir el coste computacional y simplificar el espacio de datos mediante la eliminación (aleatoria o dirigida) de muestras correspondientes a la clase mayoritaria (transacciones legítimas).

- **XAI (Explainable Artificial Intelligence / IA Explicable):** Conjunto de técnicas y marcos de trabajo (*frameworks*) diseñados para que la lógica interna y los resultados de los algoritmos de aprendizaje automático sean transparentes, interpretables y auditables por analistas humanos.

- **XGBoost (eXtreme Gradient Boosting):** Algoritmo de aprendizaje secuencial de alto rendimiento basado en árboles de decisión optimizados por gradiente. Fue seleccionado en este estudio por su capacidad dominante en la clasificación de datos tabulares con alta asimetría y no linealidad.

- **Zero-Day Attack (Ataque de Día Cero):** Patrón o tipología de fraude informático completamente inédito que explota una vulnerabilidad desconocida, para el cual los sistemas de defensa predictiva aún no poseen datos históricos de entrenamiento.

**\**

# **Resumen** 

El fraude con tarjetas de crédito representa un desafío crítico para el sector financiero, caracterizado por un desequilibrio extremo de clases y la evolución constante de los patrones delictivos (*concept drift*). Si bien los algoritmos de aprendizaje automático han demostrado una alta capacidad predictiva, su adopción operativa se ve frecuentemente limitada por su naturaleza de \"caja negra\" y por evaluaciones de rendimiento infladas debido a metodologías de validación defectuosas. Este Trabajo de Fin de Máster (TFM) propone un marco de detección robusto y auditable, fundamentado en las directrices de la Universidad Libre de Bruselas (ULB). A nivel metodológico, la investigación descarta el uso de técnicas de sobremuestreo sintético (SMOTE) en favor de un submuestreo controlado (ratio 5:1) para optimizar el coste computacional, e implementa una estrategia de validación prequencial estricta con latencia (*gap*) para aislar el modelo del futuro y evitar la fuga de datos (*data leakage*). Adicionalmente, se integra una capa de Inteligencia Artificial Explicable (XAI) mediante valores SHAP para traducir las predicciones de riesgo en \"códigos de razón\" interpretables. Los resultados empíricos demuestran que el algoritmo *XGBoost* optimizado maximiza la métrica de negocio (*Card Precision@100*) y cuantifican cómo los errores metodológicos tradicionales inflan artificialmente el rendimiento hasta en un 40%, aportando una prueba de concepto metodológica avanzada que sienta las bases para futuras validaciones en entornos productivos reales.

**Palabras clave:** Detección de fraude, Aprendizaje desbalanceado, Inteligencia Artificial Explicable (XAI), Fuga de datos, Validación temporal.

# **Abstract**

Credit card fraud represents a critical challenge for the financial sector, characterized by extreme class imbalance and the constant evolution of criminal patterns (concept drift). While machine learning algorithms have demonstrated high predictive capacity, their operational adoption is frequently limited by their \"black box\" nature and inflated performance evaluations due to flawed validation methodologies. This Master\'s Thesis proposes a robust and auditable detection framework, grounded in the guidelines of the Université Libre de Bruxelles (ULB). Methodologically, the research discards synthetic oversampling techniques (SMOTE) in favor of controlled undersampling (5:1 ratio) to optimize computational cost, and implements a strict prequential validation strategy with a latency gap to isolate the model from the future and prevent data leakage. Additionally, an Explainable Artificial Intelligence (XAI) layer using SHAP values is integrated to translate risk predictions into interpretable reason codes. Empirical results demonstrate that the optimized XGBoost algorithm maximizes the business metric (Card Precision@100) and quantify how traditional methodological errors artificially inflate performance by up to 40%, providing an advanced methodological proof of concept that lays the groundwork for future validations in real-world production environments.

**Keywords:** Fraud Detection, Imbalanced Learning, Explainable AI (XAI), Data Leakage, Temporal Validation.

# **Introducción**

## Contexto y Problemática

El ecosistema financiero global atraviesa una transformación estructural sin precedentes, impulsada por la digitalización acelerada de los medios de pago, la adopción masiva de *wallets* digitales (como *Apple Pay* o *Google Pay*) y la consolidación del comercio electrónico transfronterizo. La entrada en vigor de normativas como la Directiva Europea de Servicios de Pago (PSD2) ha fomentado el *Open Banking* y la fricción cero en las transacciones, optimizando la experiencia del usuario, pero expandiendo drásticamente la superficie de exposición ante actividades ilícitas.

Según reportes recientes de la industria, las pérdidas globales derivadas del fraude con tarjetas superaron los 33.400 millones de dólares en 2024, un escenario dominado casi en su totalidad por vectores de ataque no presenciales (Card-Not-Present o CNP), como se esquematiza en el flujo operativo de la Ilustración 2. En el ámbito europeo, el informe conjunto de la Autoridad Bancaria Europea (EBA) y el Banco Central Europeo (BCE) confirmó que el fraude en pagos superó la barrera de los 4.200 millones de euros anuales, experimentando un crecimiento interanual sostenido. A esta magnitud económica se suma un factor disruptivo reciente: la \"industrialización\" del cibercrimen mediante el modelo de *Fraud-as-a-Service* (FaaS) y el uso de Inteligencia Artificial Generativa para la automatización del *phishing* y la creación de identidades sintéticas. Estos vectores de ataque dinámicos han vuelto completamente obsoletos a los sistemas tradicionales de seguridad bancaria basados en reglas estáticas o umbrales manuales.

La magnitud sistémica de este problema se hace evidente al observar la evolución macroeconómica de las pérdidas globales (véase la Ilustración 1). El crecimiento sostenido del fraude está directamente correlacionado con el auge del comercio electrónico, donde el vector no presencial se ha convertido en la principal vulnerabilidad.

![[]{#_Toc225188150 .anchor}Ilustración . Evolución histórica de las pérdidas globales por fraude en tarjetas de crédito, destacando la predominancia y el crecimiento sostenido del vector no presencial (Card-Not-Present). Fuente: The Nilson Report.](docs/media/media/image2.png){alt="Card fraud worldwide from 2010 to 2027 [1]." width="4.853625328083989in" height="3.1071423884514435in"}

![[]{#_Toc225188151 .anchor}Ilustración . Arquitectura de un Sistema de Detección de Fraude (FDS) en el sector financiero, diferenciando procesos en tiempo real, casi en tiempo real y offline. Fuente: Dal Pozzolo et al. (2014).](docs/media/media/image3.jpeg){alt="alt text" width="5.905555555555556in" height="2.66875in"}

Para hacer frente a esta amenaza, la industria ha migrado masivamente hacia la adopción de motores de Inteligencia Artificial y *Machine Learning*. Sin embargo, desde la perspectiva de la ciencia de datos, la interceptación de este fraude constituye uno de los mayores retos algorítmicos de la actualidad debido a un problema fundacional: el desbalance de clases extremo. En la operativa real de una red de pagos, las transacciones fraudulentas representan habitualmente menos del 0,5% del volumen total procesado. Esta asimetría masiva provoca que los clasificadores supervisados estándar fracasen de forma inherente, ya que tienden a sesgar sus predicciones hacia la clase mayoritaria (las transacciones legítimas) con el objetivo matemático de minimizar su función de pérdida global, ignorando las anomalías (He & Garcia, 2009).

Para paliar este desequilibrio, gran parte de la literatura científica reciente ha centrado sus esfuerzos en la generación masiva de datos sintéticos y en el despliegue de arquitecturas de aprendizaje profundo (*Deep Learning*) de alta complejidad, reportando de forma habitual tasas de detección casi perfectas. No obstante, la premisa central de este Trabajo Fin de Máster es que dicho enfoque predominante en la literatura ha derivado en tres problemas estructurales que dificultan su aplicabilidad directa en la industria bancaria actual:

1.  **La ilusión del rendimiento (*Data Leakage*) y el *Concept Drift*:** Diversos estudios del estado del arte reportan precisiones superiores al 99% evaluando los modelos sobre datos pasados. Sin embargo, en el mundo real, el comportamiento criminal muta constantemente para evadir las defensas (*Concept Drift*) (Gama et al., 2014). Las métricas ilusorias publicadas suelen ser el producto de fugas de información (*Data Leakage*) derivadas de una validación temporal defectuosa (como el uso de particiones aleatorias que mezclan el futuro con el pasado) o de la inyección global de ruido sintético previo a la división de los datos, lo que permite al modelo memorizar vectores de ataque en lugar de generalizarlos (Hayat & Magnier, 2025).

2.  **El colapso operativo por Falsos Positivos:** El enfoque predominante en la literatura generalista por maximizar la detección total del fraude ha generado modelos hiper-sensibles que disparan miles de falsas alarmas diarias. En la realidad operativa, las entidades financieras poseen una capacidad finita en sus Centros de Operaciones de Seguridad (SOC) para investigar manualmente los bloqueos. Además, un alto índice de falsos positivos genera fricción en clientes legítimos, provocando un grave daño reputacional y la pérdida de negocio. Es imperativo abandonar las métricas estadísticas puras en favor de métricas orientadas al coste operativo y a la prioridad de negocio (Correa Bahnsen et al., 2015).

3.  **La falta de auditabilidad y el cumplimiento normativo:** En un sector críticamente regulado por normativas como el Reglamento General de Protección de Datos (RGPD) europeo o los acuerdos de Basilea, un modelo predictivo opaco (\"caja negra\") carece de viabilidad legal. Existe una profunda brecha entre alcanzar una alta exactitud matemática y la necesidad corporativa de poseer una toma de decisiones transparente. Si un algoritmo bloquea la tarjeta de un usuario en el extranjero, el analista de riesgos debe poder auditar y justificar formalmente el motivo del bloqueo, lo que exige la adopción imperativa de técnicas de explicabilidad algorítmica (Hasan & Gazi, 2025; Bücker et al., 2022).

En síntesis, este escenario demanda una revisión crítica de las metodologías actuales, alejándose de la búsqueda de la perfección matemática artificial para transicionar hacia arquitecturas robustas, evaluadas bajo cronologías estrictas y capaces de operar con transparencia en entornos financieros regulados.

## Preguntas de Investigación

- **Pregunta 1:** ¿Es estrictamente necesaria la aplicación de técnicas complejas de sobremuestreo sintético, como SMOTE (*Synthetic minority oversampling technique*) para lidiar con el desbalance extremo, o puede una estrategia de submuestreo controlado maximizar la viabilidad operativa del negocio?

- **Pregunta 2:** ¿Cuál es el impacto cuantitativo real (inflación de métricas) que sufre un modelo predictivo cuando se cometen errores metodológicos de *Data Leakage* durante su fase de validación temporal?

- **Pregunta 3:** ¿Es posible extraer \"códigos de razón\" (*reason codes*) auditables de modelos no lineales de alto rendimiento (como XGBoost) sin degradar su capacidad predictiva, cumpliendo así con las exigencias regulatorias financieras?

## Objetivos del Estudio

Para dar respuesta a las preguntas de investigación y contrastar empíricamente las hipótesis planteadas, se define el siguiente marco de objetivos.

**Objetivo General**

Diseñar, auditar y validar una arquitectura metodológica basada en *Machine Learning* para la detección de fraude en tarjetas de crédito, que resuelva la ineficiencia operativa del desbalance extremo de clases, cuantifique matemáticamente el sesgo de las evaluaciones tradicionales (*Data Leakage*) e integre mecanismos de explicabilidad (XAI) para satisfacer los estándares regulatorios del sector financiero.

**Objetivos Específicos**

Para alcanzar el propósito general, la investigación se desglosa en los siguientes cuatro hitos operativos:

1.  **Construir un modelo de referencia (*Baseline*):** Implementar y contrastar arquitecturas algorítmicas representativas (Regresión Logística, *Random Forest* y *XGBoost*) sometidas a la distribución natural de los datos, garantizando la ausencia de sesgo de anticipación (*look-ahead bias*) mediante un protocolo estricto de validación prequencial con latencia temporal.

2.  **Optimizar la gestión del desbalance orientada a negocio:** Experimentar con estrategias de submuestreo controlado (*undersampling*) para determinar la configuración de retención paramétrica que, reduciendo el coste computacional, maximice la intercepción de fraude bajo restricciones operativas reales (métrica *Card Precision@100*).

3.  **Cuantificar empíricamente el *Data Leakage*:** Diseñar un ensayo de control negativo compuesto por múltiples ramas de validación deficiente para medir y aislar la inflación artificial de la métrica AUPRC (Área bajo la curva Precisión-Recall) provocada por la ruptura cronológica (*Random Split*) y la inyección global de vectores sintéticos (SMOTE).

4.  **Operacionalizar la explicabilidad algorítmica (XAI):** Integrar una capa analítica *post-hoc* basada en la Teoría de Valores de Shapley (SHAP) para la extracción de \"códigos de razón\" a nivel transaccional, y verificar la resiliencia estructural de dicha lógica de negocio mediante la ejecución de un estudio de ablación.

## Hipótesis

Las preguntas de investigación y los objetivos trazados anteriormente se fundamentan en las siguientes hipótesis de trabajo, las cuales serán sometidas a validación empírica mediante el diseño experimental progresivo de este Trabajo Fin de Máster:

- **Hipótesis 1 (Gestión del desbalance y eficiencia operativa):** La simplificación del espacio de características mediante un submuestreo agresivo y controlado de la clase mayoritaria (específicamente reduciendo el desbalance a un ratio 5:1) superará operativamente a las arquitecturas entrenadas sobre la distribución natural o compensadas con técnicas de generación sintética. Se postula que este enfoque maximizará la densidad de aciertos en la cabecera del sistema de alertas (*Card Precision@100*), optimizando los recursos del Centro de Operaciones de Seguridad (SOC) sin incurrir en sobreajuste (Li, 2024; Dal Pozzolo et al., 2015).

- **Hipótesis 2 (Integridad metodológica y el *Data Leakage*):** La omisión del carácter secuencial y evolutivo del fraude (*Concept Drift*) mediante el uso de validación cruzada aleatoria, sumada a la aplicación global de técnicas de preprocesamiento (escalado o SMOTE) previas a la partición temporal de los datos, inducirá una severa fuga de información (*Data Leakage*). Se hipotetiza que la concurrencia de estos errores inflará artificialmente la métrica del Área Bajo la Curva PR (AUPRC), creando un espejismo de viabilidad algorítmica y un falso estado del arte que no sería reproducible en un entorno bancario productivo (Hayat & Magnier, 2025).

- **Hipótesis 3 (Transparencia algorítmica y auditabilidad):** La integración de técnicas de Inteligencia Artificial Explicable (XAI) *post-hoc*, concretamente basadas en la Teoría de Valores de Shapley (SHAP), permitirá decodificar con precisión matemática las decisiones de arquitecturas de \"caja negra\" de alto rendimiento algorítmico (*XGBoost*). Se anticipa que la extracción de \"códigos de razón\" locales demostrará empíricamente que el modelo no ajusta sobre ruido estocástico, sino que fundamenta sus bloqueos preventivos en una lógica de negocio financiera estricta, satisfaciendo así los requisitos de auditabilidad impuestos por la regulación europea (Bücker et al., 2022).

## Contribuciones Principales del Trabajo

Es importante delimitar que la contribución principal de este trabajo no radica en el diseño de una arquitectura algorítmica nueva, sino en la auditoría metodológica rigurosa de las prácticas actuales en la ciencia de datos financiera. El valor diferencial se sitúa en la intersección de tres ejes: la evaluación temporal estricta (validación prequencial), la explicabilidad validada por ablación y la cuantificación empírica de sesgos metodológicos. Las tres contribuciones concretas son:

1.  **Desmitificación de Métricas Ilusorias mediante Auditoría de Fugas (Data Leakage):**

La literatura académica reciente presenta una proliferación de modelos que reportan precisiones cercanas a la perfección. Este trabajo contribuye con evidencia empírica cuantificable que demuestra cómo el mal uso del sobremuestreo sintético (SMOTE) y la aplicación de particiones estáticas (*Random Split*) comprometen la validez operativa de dichos resultados. Al someter a los algoritmos a un ensayo de control negativo, se establece un *benchmark* honesto y se evidencia que el rendimiento aparentemente \"perfecto\" puede ser un artefacto metodológico, contribuyendo al rigor de futuras investigaciones en el dominio.

2.  **Gestión Operativa del Desbalance (Submuestreo Controlado y Aprendizaje Sensible al Costo):**

Se cuestiona y refuta la necesidad sistemática de inyectar datos artificiales (con el consiguiente aumento de complejidad y riesgo de fuga de información) para lidiar con asimetrías extremas (0,17%). El estudio demuestra en la práctica que la simplificación del espacio de datos mediante un submuestreo agresivo y controlado de la clase mayoritaria (ratio 5:1), combinada con la arquitectura *XGBoost*, es computacional y operativamente superior a las técnicas de sobremuestreo del estado del arte. Esta estrategia maximiza directamente el retorno de inversión al priorizar la métrica de negocio *Card Precision@100*.

3.  **Transición hacia la \"Caja Blanca\" Operativa (Implementación SOC):**

Más allá de la mera clasificación matemática, la investigación operacionaliza la explicabilidad algorítmica integrando valores SHAP en el flujo de decisión. Se aporta un marco de trabajo capaz de generar \"códigos de razón\" (*reason codes*) visuales y locales (*Force Plots*) orientados a los analistas del Centro de Operaciones de Seguridad (SOC). Esto garantiza que cada bloqueo preventivo emitido por el modelo posea una narrativa trazable (ej. riesgo histórico del terminal sumado a un importe anómalo), cerrando la brecha entre la eficacia de la Inteligencia Artificial y el cumplimiento normativo exigido por el regulador europeo.

# **Estado del arte o Marco Teórico**

La detección de fraude con tarjetas de crédito (*Credit Card Fraud Detection*, CCFD) se ha consolidado como una disciplina crítica dentro del aprendizaje automático aplicado a finanzas. Sin embargo, la revisión de la literatura reciente (2020-2025) revela una dicotomía metodológica: mientras gran parte de la investigación académica se centra en maximizar métricas sintéticas mediante arquitecturas complejas, la literatura aplicada, liderada por grupos como el de la *Université Libre de Bruxelles* (ULB), advierte sobre la falta de reproducibilidad y la prevalencia de evaluaciones optimistas debido a errores en el diseño experimental.

A continuación, se fundamentan los pilares teóricos que sustentan la metodología de este TFM.

## Naturaleza del Problema: Desbalance Extremo y Concept Drift

A diferencia de los problemas de clasificación estándar, el fraude financiero se caracteriza por un desequilibrio de clases extremo, donde la clase positiva (fraude) típicamente representa menos del 0.5% del total de transacciones. Según Dal Pozzolo *et al.* (2014), este desbalance provoca que los clasificadores estándar sesguen su aprendizaje hacia la clase mayoritaria (transacciones legítimas) para minimizar la función de pérdida global, ignorando los casos de fraude. Esta invisibilidad del fraude frente a la densidad de operaciones normales se ilustra espacialmente en la Ilustración 3.

![[]{#_Toc225188152 .anchor}Ilustración . Representación espacial del desbalance extremo en el conjunto de datos, evidenciando el solapamiento de la clase minoritaria (fraude) sobre la densidad de la clase legítima. Fuente: Elaboración propia](docs/media/media/image5.svg){width="5.577380796150481in" height="3.9554538495188103in"}

Adicionalmente, el fraude opera en un entorno adversario dinámico sujeto al Concept Drift (cambio de concepto). Los patrones delictivos no son estacionarios; evolucionan en respuesta a las medidas de seguridad implementadas (Gama et al., 2014). Este comportamiento mutante se clasifica en diversas tipologías, como se observa en la Ilustración 4. La literatura reciente demuestra que ignorar la cronología de las transacciones en la fase de validación invisibiliza esta evolución temporal del estafador, haciendo obligatorio el abandono de las particiones aleatorias en favor de ventanas de validación prequencial (Lucas et al., 2020; Baesens et al., 2015).

![[]{#_Toc225188153 .anchor}Ilustración . Tipologías de Concept Drift (cambio de concepto) en flujos de datos a lo largo del tiempo. Fuente: Gama et al. (2014).](docs/media/media/image6.png){alt="Concept Drift Types: A. Sudden Drift B. Gradual Drift C. Incremental Drift D. Recurring Drift" width="5.232142388451444in" height="2.2413801399825024in"}

## Paradigmas de Aprendizaje: Clasificación Supervisada vs. Detección de Anomalías

Las técnicas de *Machine Learning* se han convertido en el estándar de la industria para la detección automatizada de anomalías, superando ampliamente a los antiguos sistemas basados en reglas estáticas. En los últimos años, el estado del arte ha explorado enfoques híbridos, combinando redes generativas adversarias (GANs) (Fiore et al., 2019), arquitecturas de ensamblaje optimizadas (Taha & Malebary, 2020) y estrategias combinadas de aprendizaje no supervisado (Carcillo et al., 2021) para intentar mitigar la asimetría intrínseca de los datos de pago (Makki et al., 2019).

Dentro de este vasto ecosistema, es imperativo establecer una distinción taxonómica clara para el presente estudio, atendiendo a las recomendaciones clásicas de Chandola et al. (2009) y a las directrices aplicadas de Le Borgne et al. (2022):

1)  **Detección de Anomalías (No Supervisado/Semi-supervisado):** Se utiliza cuando no existen etiquetas confirmadas o cuando se busca detectar *nuevos* tipos de ataques desconocidos (*Zero-day attacks*). Algoritmos como *Isolation Forest* o *Autoencoders* modelan la distribución normal de los datos y marcan como anómalos aquellos puntos con alto error de reconstrucción o baja densidad.

2)  **Clasificación (Supervisado):** Se aplica cuando se dispone de un histórico de transacciones etiquetadas (Fraude/No Fraude).

**Justificación del Enfoque:** Dado que el conjunto de datos de referencia (ULB Kaggle) dispone de etiquetas de alta calidad (\"Class\"), este TFM adopta el paradigma de Clasificación Supervisada. Ignorar las etiquetas disponibles para utilizar técnicas puramente no supervisadas (como Autoencoders) constituiría una ineficiencia metodológica, perdiendo la señal predictiva explícita que ofrecen los patrones de fraude históricos conocidos.

## La Problemática de la Validación y el Data Leakage

Uno de los hallazgos más críticos en el estado del arte es la omnipresencia del *Data Leakage* (fuga de datos) en estudios de CCFD. Una revisión sistemática de los *kernels* públicos y *papers* recientes indica dos errores recurrentes que este estudio se propone corregir:

1.  **Validación Cruzada Aleatoria (*Random K-Fold*):** Tratar las transacciones como eventos independientes e idénticamente distribuidos (i.i.d.) permite que el modelo entrene con datos futuros y prediga datos pasados. La metodología correcta, adoptada en este trabajo, es la validación **Prequential** (Predictive Sequential), que respeta estrictamente el orden cronológico.

2.  **Contaminación por Sobremuestreo:** La aplicación de técnicas como SMOTE (*Synthetic Minority Over-sampling Technique*) sobre el conjunto de datos completo *antes* de la división Train/Test. Esto permite que el modelo \"vea\" patrones sintéticos generados a partir de los datos de prueba, inflando artificialmente métricas como el AUPRC y anulando la validez de los resultados.

<!-- -->

3)  **Gestión del Desbalance: Aprendizaje Sensible al Costo (*Cost-Sensitive Learning*)**

Si bien el re-muestreo (SMOTE, ADASYN) es popular, introduce ruido y complejidad computacional en espacios de alta dimensionalidad. Autores como Elkan (2001) sugieren que alterar la distribución de entrenamiento sesga las probabilidades posteriores reales.

Este TFM se fundamenta en el Aprendizaje Sensible al Costo. En lugar de alterar los datos, se altera la función de coste del algoritmo, asignando una penalización mayor ($\lambda$) a los errores de Tipo II (Falsos Negativos: no detectar un fraude) que a los de Tipo I (Falsos Positivos).

Para modelos basados en gradientes (Gradient Boosting Decision Trees - XGBoost), esto se implementa mediante la ponderación de instancias (scale_pos_weight), lo que permite focalizar el aprendizaje en la clase minoritaria manteniendo la integridad estadística de los datos originales.

## Inteligencia Artificial Explicable (XAI): Superando la Caja Negra

La normativa financiera (como Basilea III y GDPR) exige auditabilidad en las decisiones de riesgo. Los modelos de ensamblaje (*Ensemble Methods* como Random Forest y XGBoost), aunque superiores en rendimiento a la regresión logística, son opacos por diseño.

Las métricas tradicionales de \"Importancia de Variables\" (como la Ganancia de Información o Impureza de Gini) han demostrado ser inconsistentes y sesgadas hacia variables de alta cardinalidad (Lundberg y Lee, 2017). Por ello, el estado del arte converge hacia el uso de Valores SHAP (SHapley Additive exPlanations). Fundamentados en la Teoría de Juegos cooperativos, los valores SHAP garantizan propiedades de consistencia y precisión local, permitiendo descomponer la predicción de riesgo ($f(x)$) de una transacción específica en la suma de las contribuciones marginales de cada variable ($\phi_{i})$:

$$A\ f(x) = \ \phi_{0} + \ \sum_{i = 1}^{M}{\phi_{i}x_{i}}$$

Esta aproximación permite no solo identificar el fraude, sino generar explicaciones operativas (ej. \"Transacción denegada debido a la confluencia de \'Monto elevado\' y \'Geolocalización inusual\'\"), cerrando la brecha entre la precisión algorítmica y la interpretabilidad humana.

# **Metodología**

De acuerdo con la taxonomía de la investigación científica en ingeniería, el presente Trabajo Fin de Máster se enmarca en un diseño de investigación cuantitativo, explicativo y experimental. Es cuantitativo porque se fundamenta en la medición matemática objetiva de variables transaccionales y el análisis de rendimiento mediante métricas estadísticas (AUPRC, CP@100). Es explicativo porque no se limita a describir el fenómeno del fraude, sino que busca determinar las causas de la inflación de métricas en la literatura mediante la teoría de valores de Shapley (SHAP). Finalmente, posee un carácter experimental transversal, al manipular variables independientes (como las estrategias de validación o el balanceo de clases) sobre un entorno de control para medir su impacto en la capacidad predictiva.

La estrategia metodológica adoptada se fundamenta en la premisa de que la detección de fraude en tarjetas de crédito no es meramente un problema de clasificación binaria estándar, sino un desafío complejo de series temporales desbalanceadas donde la integridad del flujo de datos es crítica. A diferencia de enfoques académicos generalistas, este estudio adopta y extiende el marco de trabajo \"Fraud Detection Handbook\" desarrollado por el Grupo de Machine Learning de la *Université Libre de Bruxelles* (ULB). Este capítulo detalla el diseño experimental híbrido que estructura la investigación.

A diferencia de enfoques académicos generalistas que priorizan la maximización de métricas teóricas en entornos estáticos, este estudio adopta y extiende el marco de trabajo \"Fraud Detection Handbook\" desarrollado por el Grupo de Machine Learning de la *Université Libre de Bruxelles* (ULB). Este capítulo detalla el diseño experimental híbrido que estructura la investigación.

En primer lugar, se describe la metodología base adquirida del estudio de referencia, la cual proporciona un canal metodológico (*pipeline)* robusto para la generación de características, transformando datos crudos de transacciones en variables inteligentes que un modelo puede entender, y la validación temporal, respetando estrictamente la cronología para evitar el uso de información futura y simular un entorno operativo real.

Posteriormente, se exponen las contribuciones propias de este trabajo, centradas en la auditoría de la integridad metodológica mediante la simulación controlada de *Data Leakage* y la incorporación de una capa de explicabilidad (XAI) necesaria para la aplicabilidad del modelo en entornos regulados.

El objetivo final no es solo obtener un modelo predictivo eficaz, sino demostrar empíricamente por qué ciertas prácticas metodológicas son imperativas para evitar resultados ilusorios en la detección de fraude.

## Marco Metodológico de Referencia

La presente investigación toma como punto de partida y fundamentación técnica el marco de trabajo \"Fraud Detection Handbook\", resultado de una extensa colaboración entre el Grupo de Aprendizaje Automático (*Machine Learning Group*) de la Universidad Libre de Bruselas (ULB) y la compañía de pagos Worldline. La elección de este referente no es arbitraria, sino que responde a la necesidad de abordar el problema de la detección de fraude con un estándar de reproducibilidad y realismo que complementa la literatura académica generalista.

### Origen y Contexto del Estudio Base

El estudio original aborda la problemática de la reproducibilidad y la validez temporal en la ciencia de datos financiera. Tradicionalmente, la literatura sobre detección de fraude ha sufrido de una falta de estandarización, donde numerosos estudios reportan métricas de rendimiento cercanas a la perfección, como AUC-ROC superiores al 0.99 (Abdulghani et al., 2021; Sadgali et al., 2021; Ileberi et al., 2021).

Sin embargo, análisis críticos recientes demuestran que estos resultados suelen ser ilusorios y no reproducibles en entornos productivos, debido principalmente a la aplicación incorrecta de técnicas de re-muestreo (como SMOTE) antes de la división de los datos, lo que provoca una fuga de información del futuro hacia el modelo (Hayat & Magnier, 2025).

Este TFM asume el protocolo de la ULB como punto de partida metodológico (*baseline*), utilizando su generador de datos simulados para crear un conjunto de transacciones que mimetiza patrones de gasto reales y comportamientos fraudulentos conocidos, garantizando así que los resultados sean comparables y verificables.

### Pipeline de Referencia: Ingeniería de Características RFM

El flujo de trabajo estándar adoptado en este TFM se articula en dos etapas secuenciales heredadas de la metodología original: la generación controlada de datos y su posterior enriquecimiento mediante ingeniería de características.

#### Generación de Datos Simulados 

Debido a las restricciones de confidencialidad inherentes al sector bancario, el acceso a datos reales etiquetados y públicos es extremadamente limitado. Para sortear esta barrera y garantizar la reproducibilidad científica, se utiliza el simulador de transacciones desarrollado por Le Borgne et al. (2022).

Este generador produce un registro sintético de transacciones (denominado dataset base) que mimetiza el comportamiento de una red de pagos real durante un periodo definido (ej. 183 días). La simulación incluye:

- **Perfiles de Clientes Legítimos:** Agentes con patrones de gasto habituales (montos, frecuencias y horarios específicos).

- **Escenarios de Fraude:** Inyección controlada de patrones anómalos conocidos, lo que proporciona una verdad fundamental (*ground truth*) fiable para el entrenamiento supervisado.

- **Topología de Red:** Una estructura de clientes y terminales (comercios) que interactúan a lo largo del tiempo.

El resultado es un conjunto de datos crudos compuesto por variables básicas: identificador de cliente, identificador de terminal, fecha/hora y monto de la transacción.

#### Ingeniería de Características (Feature Engineering) RFM

Dado que las variables crudas son insuficientes para que un modelo detecte patrones complejos, el procedimiento aplica una transformación determinista basada en el paradigma RFM (*Recencia, Frecuencia, Monto*). Este proceso convierte cada transacción aislada en un vector de contexto histórico mediante el uso de ventanas temporales deslizantes (definidas en 1, 7 y 30 días).

Las características generadas se dividen en dos ejes de análisis:

- **Comportamiento del Cliente:** Se agregan métricas como el número de transacciones (NB_TX_WINDOW) y el monto promedio (AVG_AMOUNT_WINDOW) en las ventanas recientes. Esto permite modelar el perfil de gasto habitual del usuario y detectar desviaciones abruptas (ej. un pico de actividad en 24 horas).

- **Riesgo del Terminal:** Se cuantifica el riesgo histórico de los comercios mediante el conteo de fraudes previos asociados a cada terminal (RISK_WINDOW). Como demostraron los experimentos preliminares, esta variable de riesgo es un predictor crítico para identificar puntos de compromiso sistémico.

El modelo conceptual para la transformación e ingestión de estas variables se detalla en la Ilustración 5.

![[]{#_Toc225188154 .anchor}Ilustración . Flujo de trabajo de referencia para la ingeniería de características y validación de modelos en detección de fraude. Fuente: Le Borgne et al. (2022).](docs/media/media/image7.png){width="5.905555555555556in" height="2.2131944444444445in"}

Este flujo de trabajo asegura que, antes de aplicar cualquier algoritmo de aprendizaje automático, los datos hayan pasado de ser eventos discretos, a construir una narrativa de comportamiento temporal coherente y lista para su procesamiento.

### Estrategia de Validación Temporal

El tercer pilar del marco de referencia se define por el rechazo explícito a la validación cruzada aleatoria (*Random K-Fold Cross Validation*). Aunque esta técnica constituye el estándar de oro en problemas de aprendizaje supervisado donde se asume que las observaciones son independientes e idénticamente distribuidas (i.i.d.), su aplicación en el dominio de la detección de fraude resulta metodológicamente inválida al violar la dependencia temporal intrínseca de las transacciones. El uso de una división aleatoria mezcla inevitablemente datos del futuro en el conjunto de entrenamiento, permitiendo al modelo \"aprender\" patrones de fraudes que aún no han ocurrido cronológicamente. Este fenómeno, conocido como *Data Leakage* o fuga de datos, infla artificialmente las métricas de rendimiento, generando una falsa sensación de seguridad que no se sostiene en un entorno productivo.

Para corregir este sesgo estructural, el estudio adopta una estrategia de Validación Prequencial (*Prequential Evaluation*), la cual respeta estrictamente la flecha del tiempo. El protocolo implementado divide el conjunto de datos en bloques secuenciales, introduciendo un componente crítico: el periodo de retraso (*delay period,* véase la Ilustración 6).

![[]{#_Toc225188155 .anchor}Ilustración . Estrategia de evaluación prequencial en bloques con periodo de latencia (delay) para evitar fugas de información. Fuente: Elaboración propia basada en Le Borgne et al. (2022).](docs/media/media/image8.png){width="4.821428258967629in" height="2.263303805774278in"}

Este margen temporal (establecido en 7 días para este estudio) simula la latencia operativa real que transcurre entre la ocurrencia de un fraude y su confirmación definitiva por parte de la entidad financiera o el cliente. La exclusión de los datos pertenecientes a este periodo \"ciego\" durante el entrenamiento impide que el modelo acceda a etiquetas que no estarían disponibles en el momento de la predicción.

La elección de un periodo de latencia (*gap*) de 7 días no es un valor arbitrario, sino que modela el ciclo operativo y administrativo real del sector bancario. Este margen temporal representa el retraso acumulado entre el momento en que se perpetra el fraude, el tiempo de reacción del cliente para identificar el cargo anómalo en su extracto y notificarlo, y el periodo que requiere la entidad financiera para confirmar la disputa e incorporar formalmente esa etiqueta al sistema central.

Adicionalmente, para preservar la integridad del conjunto de evaluación, cualquier proceso de optimización de hiperparámetros se realiza exclusivamente mediante una subdivisión interna del bloque de entrenamiento (*Inner Loop Validation*). De esta forma, el conjunto de prueba actúa como una instancia virgen que solo se utiliza para la medición final del rendimiento, asegurando que las métricas obtenidas (como AUPRC y *Card Precision@100*) reflejen la capacidad real del modelo para generalizar ante nuevas tipologías de fraude en el futuro.

## Adaptaciones y Contribuciones Metodológicas Propias

La adopción del marco de referencia de la *Université Libre de Bruxelles* (ULB) constituye el cimiento técnico, pero no el límite, de esta investigación. Si bien dicho marco proporciona los instrumentos fundamentales para la generación de modelos predictivos, este Trabajo Fin de Máster introduce una capa de **diseño crítico y adaptación operativa** orientada a transformar un ejercicio de modelado académico en una propuesta de solución viable, robusta y auditable para un entorno financiero regulado.

El valor diferencial de este estudio reside en que no asume la validez de los resultados por la mera ejecución de algoritmos estándar. Por el contrario, se ha diseñado una metodología extendida que cuestiona y valida la integridad del propio proceso de aprendizaje. Las contribuciones propias desarrolladas en este trabajo surgen como respuesta a dos carencias sistémicas identificadas en el estado del arte: la prevalencia de métricas de rendimiento ilusorias derivadas de diseños experimentales laxos y la inoperabilidad de modelos opacos (\"cajas negras\") en sectores sujetos a normativas estrictas.

En consecuencia, la metodología de este proyecto se expande más allá de la maximización de la capacidad predictiva para incorporar tres intervenciones estratégicas diseñadas específicamente para este TFM:

1.  El realineamiento del protocolo de evaluación hacia objetivos de negocio tangibles

2.  La auditoría forense de la integridad experimental mediante pruebas de control negativo (*Anti-Leakage*)

3.  La integración de mecanismos de explicabilidad (*XAI*) como requisito funcional de primer orden.

### Refinamiento del Protocolo Experimental

La primera intervención crítica sobre el marco base consiste en la redefinición de los criterios de éxito. La revisión del estado del arte evidencia una tendencia generalizada a optimizar modelos basándose en la Exactitud (*Accuracy*) o el Área Bajo la Curva ROC (AUC-ROC). Sin embargo, en un escenario de desbalance extremo donde el fraude representa apenas el 0,17% de las transacciones, estas métricas resultan matemáticamente ciegas al problema de negocio.

Este estudio se distancia de dichas prácticas convencionales para implementar un protocolo de evaluación orientado al impacto financiero y operativo:

#### Sustitución de Métricas Generalistas por AUPRC:

Se ha descartado la *Accuracy* como indicador de desempeño al constatar la \"Paradoja de la Exactitud\", donde un modelo trivial que clasifique todas las operaciones como legítimas alcanzaría un 99,8% de acierto, siendo sin embargo inútil para la detección.

En su lugar, se establece el Área Bajo la Curva de Precisión-Recall (AUPRC) como la métrica rectora para la selección de modelos. A diferencia del AUC-ROC, que se ve inflado por la inmensa mayoría de verdaderos negativos (transacciones normales correctamente clasificadas), la AUPRC se focaliza exclusivamente en la clase minoritaria, penalizando severamente los falsos positivos. Esto permite distinguir modelos que simplemente \"aciertan mucho\" de aquellos que realmente \"encuentran el fraude\".

#### Incorporación de Métricas de Negocio (Card Precision@k):

Más allá del rendimiento estadístico, se introduce una restricción operativa realista mediante la métrica *Card Precision@k* (*CP@k*), fijando *k=100*. Esta decisión metodológica simula la capacidad finita de un equipo de analistas de fraude, que solo puede revisar humanamente un número limitado de alertas diarias (ej. 100).

Bajo este prisma, el objetivo del experimento deja de ser la detección teórica de *todos* los fraudes para centrarse en la priorización efectiva: la capacidad del algoritmo para colocar los casos reales de fraude dentro de las 100 alertas más críticas del día.

Esta reorientación del protocolo asegura que los modelos resultantes no solo sean robustos estadísticamente, sino viables económicamente, alineando el objetivo matemático de la optimización con la realidad operativa de la entidad financiera.

### Implementación de Integridad Metodológica (Test Anti-Leakage)

Si bien la literatura científica advierte teóricamente sobre los riesgos del *Data Leakage* (fuga de datos), rara vez se cuantifica experimentalmente su impacto en la toma de decisiones de inversión tecnológica. Como contribución metodológica distintiva, este estudio no se limita a aplicar pasivamente las mejores prácticas, sino que ha diseñado y ejecutado una auditoría de control negativo para contrastar empíricamente la magnitud de este error silencioso.

El objetivo de esta intervención es demostrar que las métricas de rendimiento \"perfectas\" (AUC ≈ 1.0) reportadas frecuentemente en el estado del arte (Ileberi et al., 2021; Asha & Kumar, 2021; Esghir et al., 2025) no son fruto de la superioridad algorítmica, sino artefactos metodológicos derivados de un diseño experimental defectuoso y no reproducibles en entornos productivos. Investigaciones como la de Hayat & Magnier (2025) evidencian que el uso incorrecto de técnicas de re-muestreo antes de la división temporal (*Data Leakage*) es la causa principal de esta inflación artificial de métricas, un vicio metodológico que este TFM busca auditar y corregir.

Para ello, se han implementado y comparado dos procedimientos (*pipelines*) antagónicos sobre el mismo conjunto de datos:

#### Rama de Control (Metodología Correcta):

Implementación rigurosa del protocolo de validación prequencial (predictiva y secuencial) descrito en el apartado 3.1.3. En esta rama, todas las transformaciones de datos (escalado, imputación) se ajustan estrictamente dentro de los bloques de entrenamiento, y se respeta el periodo de retraso de 7 días, garantizando un aislamiento total del futuro.

#### Rama de Auditoría (Simulación de Fallo):

Reproducción deliberada de los vicios metodológicos identificados en la revisión bibliográfica. En esta rama, se permite la contaminación cruzada mediante:

- **División Aleatoria (*Random Split*):** Rompiendo la secuencia temporal y mezclando transacciones futuras en el entrenamiento.

- **Preprocesamiento Global:** Calculando estadísticas (media, desviación estándar) sobre todo el dataset antes de la división.

- **Re-muestreo Incorrecto:** Aplicando técnicas de balanceo (como SMOTE) antes de separar los conjuntos de test, lo que introduce copias sintéticas de fraudes de prueba en el entrenamiento.

La diferencia de rendimiento entre ambas ramas no se interpreta como una mejora, sino como la cuantificación de la inflación artificial de las métricas. Esta auditoría actúa como un \"test de sanidad\": si el modelo de la rama incorrecta supera drásticamente al modelo base sin justificación teórica, se confirma la presencia de fugas de información.

Los resultados preliminares de esta fase evidencian una discrepancia masiva, validando la hipótesis de que gran parte del rendimiento reportado en la literatura es ilusorio. Esta constatación justifica la rigurosidad extrema aplicada en el resto de la investigación y posiciona los resultados obtenidos (aunque numéricamente inferiores a los \"perfectos\") como métricas honestas y reproducibles en un entorno bancario real.

### Incorporación de Explicabilidad (XAI)

El tercer pilar de las contribuciones metodológicas responde a una limitación operativa crítica de los modelos basados en conjuntos de árboles (*Ensemble Methods*) como Random Forest o XGBoost. Aunque estos algoritmos ofrecen un rendimiento predictivo superior, su naturaleza de \"caja negra\" los hace intrínsecamente opacos, dificultando la comprensión de la lógica subyacente a sus decisiones.

En un entorno bancario sujeto a regulaciones estrictas (como el RGPD en Europa o las normativas de riesgo crediticio de Basilea), la opacidad algorítmica es inaceptable. Un sistema de detección de fraude no solo debe ser preciso, sino auditable: la entidad financiera debe ser capaz de justificar ante el cliente o el regulador por qué se bloqueó una transacción específica.

Para resolver este desafío, este TFM extiende el flujo de trabajo original integrando una capa de Interpretabilidad Post-Hoc basada en la Teoría de Juegos, específicamente mediante el uso de valores SHAP (*SHapley Additive exPlanations*). Esta adición metodológica transforma la utilidad del modelo en dos niveles de abstracción:

1.  **Explicabilidad Global (Direccionalidad del Riesgo):** A diferencia de las métricas tradicionales de \"Importancia de Variables\" (basadas en la reducción de impureza Gini), que solo indican *cuánto* influye una variable, el análisis SHAP revela *cómo* influye. Esto permite validar la coherencia del modelo con las reglas de negocio preexistentes.

- *Ejemplo:* Confirmar que un valor alto en la variable de \"Riesgo de Terminal\" (TERMINAL_ID_RISK) incrementa positivamente la probabilidad de fraude, mientras que la antigüedad del cliente podría disminuirla. Si el modelo aprendiera relaciones inversas contraintuitivas, esta capa permitiría detectarlo y corregirlo antes del despliegue.

2.  **Explicabilidad Local (Auditoría Unitaria):** Se habilita la capacidad de generar una \"autopsia\" instantánea para cada predicción individual. El sistema descompone la puntuación de riesgo de una transacción específica en la suma de las contribuciones de cada característica. Esta funcionalidad dota a los analistas de fraude de una herramienta de soporte a la decisión, permitiéndoles distinguir rápidamente entre un falso positivo (ej. un comportamiento inusual pero explicable) y un fraude real (ej. coincidencia de terminal comprometido y monto anómalo), optimizando así los tiempos de investigación manual.

Con esta incorporación, el proyecto trasciende el objetivo académico de maximizar una métrica (AUPRC) para cumplir con el requisito de Transparencia Algorítmica, cerrando la brecha entre la eficacia técnica de la Inteligencia Artificial y su validez operativa en procesos de negocio críticos.

## Diseño Experimental

La validez científica de cualquier investigación computacional reside en su capacidad de ser reproducida y verificada de manera independiente. En aras de garantizar dicha reproducibilidad y establecer un marco de comparación justo, este apartado detalla las especificaciones técnicas del conjunto de datos, la infraestructura de ejecución controlada y las definiciones matemáticas precisas de las métricas de evaluación seleccionadas.

### Descripción del Conjunto de Datos

Para superar las severas restricciones de confidencialidad del sector bancario y evitar las limitaciones de los *datasets* públicos anonimizados (donde se pierde el significado de las variables originales), esta investigación fundamenta su base empírica en la generación controlada de un conjunto de datos sintético.

Para ello, se ha empleado el entorno *Transaction Data Simulator* de la ULB, el cual no se limita a generar números aleatorios, sino que orquesta una simulación basada en agentes a través de tres fases secuenciales:

1.  **Generación de Perfiles de Clientes y Terminales (Agentes):** El simulador inicializa un ecosistema virtual compuesto por 5.000 clientes únicos y 10.000 terminales de pago (comercios). A cada cliente se le asigna paramétricamente un perfil de comportamiento base: una frecuencia de gasto (número medio de transacciones diarias), un importe medio habitual y una preferencia de ubicación geográfica.

2.  **Simulación del Flujo de Transacciones Legítimas:** A lo largo de un periodo continuo de 183 días (aproximadamente 6 meses, del 1 de abril al 30 de septiembre de 2018), el simulador muestrea transacciones para cada cliente basándose en su perfil. Esto genera un patrón estocástico pero coherente, respetando ciclos de actividad diurnos y nocturnos, así como variaciones entre días laborables y fines de semana.

3.  **Inyección de Escenarios de Fraude:** Finalmente, se superponen patrones de fraude conocidos sobre el flujo legítimo. El simulador compromete un porcentaje de los terminales durante ventanas temporales específicas de hasta 28 días. Si un cliente legítimo realiza una operación en un terminal comprometido, su tarjeta se \"infecta\" y comienza a emitir transacciones fraudulentas en los días posteriores, mimetizando las dinámicas reales de *skimming* o robo de credenciales.

El resultado de esta simulación es un conjunto de datos maduro que consta de 1.754.155 transacciones. La distribución de la variable objetivo (*TX_FRAUD*) presenta una asimetría extrema y realista, con únicamente un 0,57% del volumen total clasificado como transacciones fraudulentas. Al conocer la \"verdad fundamental\" (*ground truth*) exacta desde la generación del dato, se elimina el ruido de etiquetado (falsos negativos históricos), permitiendo auditar la capacidad real de los algoritmos propuestos.

### Configuración del Entorno de Ejecución

Con el objetivo de aislar la variabilidad metodológica de cualquier interferencia ambiental, todo el ciclo de vida del experimento ---desde el preprocesamiento hasta la evaluación--- se ha encapsulado en un entorno virtualizado determinista.

#### Infraestructura de Reproducibilidad

Se ha utilizado la tecnología de contenedorización Docker para desplegar un entorno de ejecución aislado. Esto garantiza que las versiones de las librerías, las dependencias del sistema operativo y las configuraciones de *hardware* virtual sean idénticas en cada iteración del experimento, eliminando el factor \"funciona en mi máquina\".

#### Stack Tecnológico

El desarrollo se ha implementado en Python 3.9, apoyándose en el ecosistema estándar de ciencia de datos para asegurar la transparencia del código:

- **Pandas & NumPy:** Para la manipulación eficiente de series temporales y cálculo matricial.

- **Scikit-Learn:** Como marco base para la construcción de *pipelines* de transformación y evaluación de modelos lineales.

- **XGBoost:** Para la implementación de algoritmos de *Gradient Boosting*, seleccionados por su eficiencia en el manejo de datos tabulares desbalanceados.

#### Control de Estocasticidad

Dado que muchos algoritmos de aprendizaje automático (como Random Forest o las inicializaciones de pesos) incluyen componentes aleatorios, se ha fijado una semilla global (random_state=42) en todos los procesos estocásticos. Este control estricto asegura que cualquier diferencia observada en el rendimiento de los modelos sea atribuible exclusivamente a las decisiones metodológicas (ej. uso de *Anti-Leakage*), y no a la varianza aleatoria de una ejecución particular.

### Definición de Métricas de Evaluación

La elección de las métricas de rendimiento no es trivial en contextos de desbalance extremo. Métricas convencionales como la Exactitud (*Accuracy*) se descartan en este estudio, ya que un modelo trivial que clasifique el 100% de las transacciones como legítimas obtendría una exactitud del 99.43%, resultando sin embargo inútil para la detección del fraude.

Se seleccionan, por tanto, indicadores que penalicen los Falsos Positivos (*FP*) y prioricen la recuperación de la clase minoritaria (*Fraude = 1*).

#### Área Bajo la Curva de Precisión-Recall (AUPRC)

Se establece como la métrica técnica principal para medir la robustez global del clasificador. A diferencia del AUC-ROC, que puede verse inflado por la gran cantidad de Verdaderos Negativos (*TN*), la AUPRC se centra exclusivamente en la clase positiva. Matemáticamente, se aproxima mediante la suma ponderada de la precisión en cada umbral *k*:

$$AUPRC = \ \sum_{k}^{}{\left( R_{k} - R_{k} - 1 \right)P_{k}}$$

Donde P~k~y R~k~ son la Precisión y el Recall en el k-ésimo umbral de decisión.

#### Card Precision@k (CP@k)

Se introduce como la métrica de negocio rectora, diseñada para simular la capacidad operativa real de un equipo de analistas de fraude. Asumiendo que los recursos humanos son finitos, un banco solo puede investigar manualmente las *k* transacciones más sospechosas de cada día.

Para este estudio se fija *k=100*, definiendo la métrica como la proporción de fraudes reales encontrados dentro de esas 100 alertas diarias prioritarias:

$$CP@100 = \ \frac{1}{100}\sum_{i = 1}^{100}{\mathbb{I(}y_{d,i} = 1)}$$

Donde $\mathbb{I}$ es la función indicador que vale 1 si la i-ésima transacción más sospechosa del día *d* es realmente un fraude. El valor final reportado es el promedio de esta precisión diaria a lo largo de todo el conjunto de prueba, ofreciendo una visión directa del retorno de inversión (ROI) del modelo.

## Desarrollo de la fase Experimental

Una vez definidos los fundamentos metodológicos, las métricas de evaluación orientadas a negocio y el entorno de ejecución controlada, la fase empírica de este Trabajo Fin de Máster se articula a través de una secuencia estratégica de experimentos.

Para garantizar la estandarización y la reproducibilidad total del estudio, la batería de pruebas se ha orquestado mediante un *framework* de experimentación propio y automatizado. Esta arquitectura (encapsulada en contenedores Docker) gestiona de manera centralizada la configuración de hiperparámetros, la ingesta de datos transformados y el registro unificado de métricas y artefactos resultantes.

Bajo esta infraestructura, el diseño experimental no se concibe como una mera iteración de algoritmos en busca del mejor resultado numérico, sino como un proceso progresivo orientado a validar tres dimensiones críticas del sistema de detección propuesto:

1.  La **eficacia técnica**, mediante el establecimiento de una línea base de rendimiento algorítmico (*Baseline técnico*).

2.  La **integridad metodológica**, a través de la auditoría de fugas de información (*Data Leakage*).

3.  La **viabilidad operativa**, incorporando transparencia en las decisiones algorítmicas (*Explainable AI*).

En consecuencia, la investigación se estructura en tres escenarios experimentales estratégicos. Cada uno de ellos está diseñado para aislar y responder a una pregunta de investigación específica, asegurando que el modelo final no solo sea preciso matemáticamente, sino robusto ante el sesgo temporal y auditable por los equipos de analistas de fraude.

Es imperativo, no obstante, establecer una cota de validez sobre el alcance de estos ensayos. Si bien el uso del generador de transacciones de la ULB garantiza la reproducibilidad y ofrece una \"verdad fundamental\" (*ground truth*) libre de la ambigüedad de etiquetado inherente a los datos bancarios reales, este escenario conlleva una necesaria idealización. A diferencia de un ecosistema financiero vivo, caracterizado por la naturaleza adversaria y cambiante de los patrones de ataque (*concept drift*) y la presencia de ruido operativo impredecible, los datos simulados presentan un comportamiento estocástico pero acotado. Por tanto, los experimentos aquí descritos deben interpretarse como una validación empírica de la robustez de la arquitectura de detección y la corrección del protocolo metodológico, asumiendo que el rendimiento absoluto en un entorno productivo real estaría sujeto a una degradación natural provocada por la entropía del sistema.

Para garantizar la claridad expositiva y la eficiencia computacional, es imperativo explicitar la trazabilidad de los algoritmos a lo largo de estas fases. El primer experimento (Definición del *Baseline*) y el tercer experimento (Test *Anti-Leakage*) se ejecutan sobre las tres arquitecturas propuestas (Regresión Logística, *Random Forest* y *XGBoost*) para demostrar que el impacto del desbalance y las fugas de información son fenómenos universales. Sin embargo, las fases de optimización orientadas a negocio (Estrategia de Submuestreo) y el Análisis de Interpretabilidad Algorítmica (XAI) se aplican de manera exclusiva sobre el algoritmo ganador de la fase inicial (*XGBoost*). Esta decisión metodológica asegura que los recursos se destinen a auditar únicamente la arquitectura final que sería desplegada en un entorno de producción.

### Definición del Modelo de Referencia (Baseline)

#### Objetivo

El propósito fundamental de esta primera fase experimental no es la construcción del modelo predictivo definitivo, sino la definición de una **referencia de rendimiento** (un \"suelo\" o *baseline* empírico). Al someter a los algoritmos a los datos en su estado \"natural\", bajo condiciones de desbalance extremo, sin ninguna técnica compensatoria, se establece un punto de referencia riguroso. Este *baseline* es indispensable para cuantificar posteriormente de manera objetiva si las metodologías avanzadas justifican su complejidad computacional.

#### Configuración Experimental

La ejecución de este ensayo se orquesta a través del *framework* contenedorizado del proyecto, garantizando el preprocesamiento aislado y el registro automático de métricas. Las especificaciones de diseño son las siguientes:

- **Selección Algorítmica:** Se despliegan tres familias de modelos, instanciados con sus hiperparámetros por defecto para evitar sesgos de optimización prematura:

  1.  *Regresión Logística:* Utilizado como el estándar lineal de referencia para determinar el rendimiento mínimo aceptable.

  2.  *Random Forest: Representante del paradigma de ensamblaje en paralelo (Bagging), seleccionado por su probada robustez frente a distribuciones ruidosas (Breiman, 2001). XGBoost: Representante del aprendizaje secuencial (Gradient Boosting), posicionado como el estado del arte en la clasificación de datos tabulares altamente desbalanceados (Chen & Guestrin, 2016).* *cualquier mecanismo de remuestreo sintético (SMOTE).*

  3.  *XGBoost:* Representante del aprendizaje secuencial (*Gradient Boosting*), posicionado como el estado del arte en la clasificación de datos tabulares.

- **Preprocesamiento Restringido:** Se aplica un escalado de varianza unitaria (*StandardScaler*) exclusivamente sobre la magnitud financiera (*Amount*), manteniendo intacta la estructura de la variable temporal (*Time*) y las componentes principales (V1-V28) heredadas del *dataset* original.

- **Aislamiento de Clases:** Como restricción crítica de este experimento, se prohíbe explícitamente el uso de técnicas de aprendizaje sensible al costo (como class_weight=\'balanced\' o scale_pos_weight) y queda estrictamente prohibido cualquier mecanismo de remuestreo sintético temprano, como SMOTE (Chawla et al., 2002).

#### Hipótesis de Trabajo

Bajo estas condiciones de \"ceguera algorítmica\" ante la asimetría de clases, se proyecta observar una divergencia radical en los indicadores de rendimiento, confirmando la falacia de las métricas generalistas.

Se hipotetiza que los modelos reportarán una Exactitud Global (*Accuracy*) ilusoriamente alta (rozando el 99%), impulsada únicamente por la clasificación trivial de la clase mayoritaria legítima. Sin embargo, al observar las métricas objetivo, se espera que el *Recall* y la AUPRC revelen una degradación crítica, demostrando la incapacidad del aprendizaje estándar para interceptar el fraude de forma autónoma. Adicionalmente, se anticipa que las arquitecturas basadas en árboles superarán ampliamente a la Regresión Logística en *Card Precision@100*, validando su selección como modelo base para las auditorías de las siguientes fases.

### Validación de Integridad Metodológica (Test Anti-Leakage)

#### Objetivo

Este experimento constituye el eje crítico y la principal contribución metodológica de la investigación. Su propósito es ejecutar una auditoría forense mediante un diseño de control negativo para demostrar, de forma empírica y cuantificable, cómo las prácticas de validación defectuosas presentes en el estado del arte inflan artificialmente las métricas de rendimiento. El objetivo es probar que los resultados casi perfectos reportados por gran parte de la literatura no provienen de una superioridad algorítmica, sino de la introducción involuntaria de un Sesgo de Anticipación (*Look-ahead Bias*).

#### Configuración Experimental

Utilizando la infraestructura automatizada del proyecto, se ha diseñado un ensayo A/B que instancia dos *pipelines* de procesamiento y entrenamiento paralelos y antagónicos. En ambas ramas se utiliza el mismo algoritmo (el modelo basado en árboles con mejor rendimiento del primer experimento, XGBoost), asegurando que cualquier discrepancia en los resultados provenga exclusivamente del tratamiento de los datos:

- **Rama A (Metodología Deficiente / Simulación de Fuga):** Se reproduce deliberadamente el error de \"leer el futuro\".

  - *División de datos:* Se aplica una partición aleatoria simple (*Random Split*), destruyendo la secuencia cronológica y permitiendo que transacciones futuras de una misma tarjeta entren en la fase de aprendizaje.

  - *Preprocesamiento:* Se aplican las técnicas de sobremuestreo para la clase minoritaria (SMOTE) y el escalado de características sobre la totalidad del *dataset antes* de realizar la división, provocando que el conjunto de prueba se contamine con copias sintéticas generadas a partir de la distribución global.

- **Rama B (Metodología Robusta / Propuesta del TFM):** Se implementa el protocolo de aislamiento estricto.

  - *División de datos:* Se emplea la Evaluación Secuencial Predictiva (división temporal en bloques) respetando el periodo de latencia (*delay*) de 7 días.

  - *Preprocesamiento:* Toda transformación o inyección de datos sintéticos (SMOTE) se encapsula y ejecuta *exclusivamente* dentro de los pliegues de entrenamiento (*Inner Loop*), proyectando las transformaciones resultantes sobre un conjunto de prueba completamente inalterado y cronológicamente posterior.

#### Hipótesis de Trabajo

Se anticipa una divergencia masiva y estadísticamente significativa en el rendimiento de ambas ramas.

La hipótesis postula que la Rama A reportará métricas ilusoriamente perfectas (con una AUPRC proyectada superior a 0,95), producto de la memorización temporal y la contaminación de la variable objetivo. Por el contrario, se espera que la Rama B, sometida al rigor del aislamiento temporal, revele la capacidad de generalización real y honesta del modelo (proyectando una AUPRC inferior). El diferencial métrico (Δ) entre ambas ramas servirá como evidencia visual y matemática para cuantificar la \"ilusión de éxito\" metodológico que este TFM busca denunciar.

### Análisis de Interpretabilidad Algorítmica (Explainable AI)

#### Objetivo

El último hito del diseño experimental tiene como propósito auditar la lógica interna del modelo de detección. En entornos financieros sujetos a estricto escrutinio regulatorio, la alta precisión de un modelo de \"caja negra\" carece de valor operativo si sus decisiones no pueden ser justificadas. Por tanto, el objetivo de esta fase es transicionar hacia un paradigma de \"caja blanca\", validando que el algoritmo toma decisiones basadas en patrones financieros lógicos y no mediante la explotación de asociaciones engañosas o sesgos inherentes a la generación sintética de datos. Adicionalmente, se busca dotar a los analistas de fraude de una herramienta que traduzca la probabilidad matemática en motivos de alerta procesables.

#### Configuración Experimental

La auditoría de interpretabilidad se ejecuta sobre el modelo más robusto identificado en las fases previas (típicamente XGBoost, tras superar el protocolo de aislamiento temporal). El *framework* aplica dos niveles complementarios de inspección sobre las predicciones del conjunto de prueba:

1.  **Interpretabilidad Nativa (*Global Feature Importance*):** Extracción de la importancia de características intrínseca del algoritmo basada en la métrica de Ganancia (*Gain*). Esta técnica genera una jerarquía estática que permite identificar rápidamente qué variables contribuyen más a la reducción de la incertidumbre (entropía) en las ramificaciones de los árboles.

2.  **Interpretabilidad Agnóstica basada en la Teoría de Juegos (*SHAP Values*):** Para decodificar la lógica de decisión del algoritmo, se implementa la librería SHAP (*SHapley Additive exPlanations*) (Lundberg & Lee, 2017). Esta técnica calcula la contribución marginal exacta que cada variable aporta a la predicción final, garantizando consistencia matemática. Tal y como se definió en la arquitectura del ensayo, esta capa analítica se aplica estrictamente sobre el modelo ganador definitivo (*XGBoost*). Para su evaluación, se instrumentan dos niveles de inspección visual:

    - ***Análisis de Explicabilidad Global (Gráficos de Resumen / Beeswarm Plots):** Diseñados para capturar el comportamiento macroscópico del modelo. Estas visualizaciones proyectan simultáneamente la magnitud de importancia de cada característica y la direccionalidad de su impacto predictivo (por ejemplo, confirmando algorítmicamente si un valor atípicamente alto en la ventana temporal de gasto incrementa o atenúa el riesgo de fraude).*

    - ***Auditoría Forense Unitaria (Gráficos de Fuerza Local y Cascada):** Orientados a la operacionalización del modelo en un Centro de Operaciones de Seguridad (SOC). Permiten desglosar el riesgo probabilístico de transacciones anómalas individuales, traduciendo el resultado matemático en \"códigos de razón\" (reason codes) tangibles. Esto simula el entorno de auditoría real, donde un analista humano requiere justificar de forma aislada e inmediata por qué el sistema bloqueó una tarjeta específica.*

#### Hipótesis de Trabajo

A nivel de jerarquía de variables, se postula que el esfuerzo de ingeniería de características (*Feature Engineering*) será validado empíricamente. Se espera que las variables derivadas del comportamiento histórico y perfiles de riesgo (parámetros RFM, como el riesgo acumulado en ventanas de 1, 7 o 30 días para terminales o clientes) dominen absolutamente el proceso de decisión, desplazando a las variables crudas o anonimizadas (PCA) a posiciones de menor relevancia.

A nivel de direccionalidad, la hipótesis anticipa que el análisis SHAP confirmará la coherencia del modelo con la intuición experta del negocio bancario. Específicamente, se espera observar relaciones monótonas justificables, donde parámetros como un alto índice de fraude reciente en un cajero (TERMINAL_ID_RISK) actúen como vectores de fuerza positivos (empujando la probabilidad hacia la clasificación de fraude), mientras que un comportamiento de gasto habitual actúe como ancla de legitimidad.

# **Discusión y Análisis de Resultados.**

En este capítulo se presentan y analizan los hallazgos empíricos obtenidos tras la ejecución de los ensayos definidos en el diseño experimental. El análisis trasciende la mera exposición de métricas de rendimiento para centrarse en la validación de las hipótesis planteadas: la demostración empírica de la \"Paradoja de la Exactitud\", la cuantificación del impacto de las fugas de información en la literatura científica y la validación de la interpretabilidad financiera de los modelos de \"caja negra\". Los resultados se presentan siguiendo una progresión lógica que va desde el establecimiento de un rendimiento base hasta la auditoría forense y explicabilidad del algoritmo seleccionado.

## Definición del Modelo de Referencia

El primer hito experimental tuvo como objetivo establecer una cota inferior de rendimiento algorítmico (*baseline*) utilizando el protocolo de validación prequencial descrito en la metodología. Para ello, se evaluaron tres arquitecturas estándar: Regresión Logística, Random Forest (Breiman, 2001) y XGBoost (Chen & Guestrin, 2016) enfrentándolas a la distribución natural del conjunto de datos temporal (0.84% de fraude en la distribución global, equivalente a un ratio aproximado de 118:1), sin la aplicación de técnicas de re-muestreo (SMOTE) ni ponderación de clases.

### Análisis de Rendimiento bajo Desbalance Natural

La evaluación inicial de los modelos sobre los 4 pliegues temporales confirmó empíricamente la hipótesis metodológica sobre la falacia de las métricas generalistas en la detección de fraude. Al analizar la Exactitud Global (*Accuracy*), los tres algoritmos reportaron valores ilusoriamente perfectos: 99.62% para Regresión Logística, 99.69% para Random Forest y 99.70% para XGBoost.

Sin embargo, dado el desbalance extremo de las clases, un clasificador trivial que predijera sistemáticamente que todas las transacciones son \"legítimas\" alcanzaría por defecto un 99.16% de exactitud. La ineficacia de esta métrica se hace evidente al observar la Sensibilidad (*Recall*) para la clase minoritaria.

![[]{#_Toc225188156 .anchor}Ilustración . Matrices de confusión de los algoritmos evaluados (Regresión Logística, Random Forest y XGBoost) evidenciando las altas tasas de Falsos Negativos. Fuente: Elaboración propia.](docs/media/media/image9.png){width="5.905555555555556in" height="1.68125in"}

Como ilustran los datos de la matriz de confusión agregada en la Ilustración 7 para el mejor modelo (XGBoost), de un total de 1.591 fraudes reales presentes en el conjunto de evaluación, el algoritmo detectó correctamente 982 (Verdaderos Positivos), pero dejó escapar 609 transacciones fraudulentas (Falsos Negativos). Esto se traduce en un *Recall* del 61.75%, lo que implica operativamente que casi 4 de cada 10 fraudes logran eludir el sistema. En el caso del modelo lineal (Regresión Logística), la fuga de fraude asciende hasta el 48.3%. Este análisis justifica formalmente la exclusión de la exactitud y de la curva ROC como criterios de decisión, alineándose con la literatura especializada que demuestra matemáticamente que la métrica AUPRC es la única herramienta robusta ante distribuciones de desbalance extremo (Saito & Rehmsmeier, 2015; He & Garcia, 2009; Fernández et al., 2018).

![](docs/media/media/image11.svg){width="5.208333333333333in" height="2.3040627734033245in"}

### Comparativa Algorítmica Estándar

Descartadas las métricas globales, el análisis comparativo se centró en indicadores robustos frente a la asimetría de clases (AUPRC) y métricas orientadas a negocio (CP@100). La Tabla 1 resume el rendimiento de los modelos, incluyendo la configuración hiperparamétrica ganadora obtenida tras la búsqueda exhaustiva (*GridSearchCV*) y sus respectivos costes computacionales.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Modelo**            **Hiperparámetros Óptimos**                        **AUPRC (Media ± Std)**   **Recall Fraude**   **CP@100 (Media ± Std)**   Tiempo de Entrenamiento
  --------------------- -------------------------------------------------- ------------------------- ------------------- -------------------------- -------------------------
  Regresión Logística   C=10                                               0.6350 ± 0.0163           51.82%              0.2929 ± 0.0141            17.4 s

  Random Forest         max_depth=50, n_estimators=100                     0.6846 ± 0.0103           58.94%              **0.2971 ± 0.0144**        \~ 2.1 min

  XGBoost               max_depth=3, n_estimators=100, learning_rate=0.3   **0.6904 ± 0.0084**       **61.75%**          0.2961 ± 0.0139            \~ 55.2 min
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  : []{#_Toc225188165 .anchor}Tabla . Resultados comparativos de las métricas de evaluación técnica y operativa para los modelos base. AUPRC = Área Bajo la Curva Precisión-Recall (mayor es mejor). CP@100 = Card Precision@100, proporción de fraudes reales entre las 100 tarjetas con mayor probabilidad de fraude por día (métrica operativa que simula la capacidad de revisión manual de un SOC). Las medias y desviaciones estándar se calculan sobre 4 folds prequenciales con ventanas semanales. El test de Friedman confirma diferencias globales significativas entre algoritmos para AUPRC (p = 0.039) y CP@100 (p = 0.037).

Para validar que las diferencias observadas entre algoritmos no son atribuibles a varianza muestral, se aplicó el test de Friedman sobre las métricas por fold prequential. El análisis reveló diferencias globales estadísticamente significativas tanto en AUPRC (χ² = 6.50, *p* = 0.039) como en CP@100 (χ² = 6.62, *p* = 0.037), confirmando que al menos un modelo se comporta de forma consistentemente distinta. Los tests pareados de Wilcoxon, no obstante, no alcanzaron significancia individual (*p* ≥ 0.125), resultado esperable dado que con *N* = 4 folds prequenciales el p-valor mínimo alcanzable por esta prueba es 0.0625. Esta limitación de potencia estadística, inherente al diseño de validación temporal con ventanas semanales sobre un dataset de 6 meses, se reconoce explícitamente en la Sección 5.3. A pesar de esta restricción, la consistencia direccional de las diferencias a través de los cuatro folds, combinada con la significancia global del test de Friedman, respalda las conclusiones cualitativas sobre el rendimiento relativo de los algoritmos.

Los resultados de la Tabla 1 revelan diferencias estructurales relevantes. La Regresión Logística estableció el \"suelo\" de rendimiento con un AUPRC de 0.6350, demostrando sus limitaciones para capturar las interacciones no lineales de las variables temporales RFM.

Por su parte, los modelos basados en ensamblajes de árboles demostraron una clara superioridad. XGBoost se posicionó como el modelo más robusto, alcanzando el mejor AUPRC (0.6904) y la mayor sensibilidad (61.75%). Resulta especialmente reveladora la configuración de sus hiperparámetros óptimos: el algoritmo favoreció árboles muy poco profundos (max_depth=3) iterados sobre 100 estimadores. Esta configuración genera un modelo intencionadamente conservador que penaliza el sobreajuste (*overfitting*), lo cual explica su extraordinaria Precisión en la detección de fraude (92.07%), habiendo emitido únicamente 85 falsas alarmas sobre un volumen de más de 230.000 transacciones legítimas. La principal desventaja de XGBoost residió en su coste computacional durante la fase de optimización, requiriendo más de 55 minutos para completar la validación frente a los 2 minutos de Random Forest.

### Selección y Justificación del Modelo Base

Para seleccionar el algoritmo que avanzará a las fases de auditoría metodológica e interpretabilidad, se cruzaron los resultados técnicos con la métrica de negocio Card *Precision@100* (*CP@100*), cuyos resultados consolidados se exponen en la Figura 4.3.

![[]{#_Toc225188157 .anchor}Ilustración . Comparativa gráfica de rendimiento técnico (AUC ROC, AUPRC) y métrica de negocio (Card Precision@100) para los modelos de referencia (4 folds prequenciales, barras de error = ±1 desviación estándar). AUC ROC = Área Bajo la Curva ROC; AUPRC = Área Bajo la Curva Precisión-Recall; CP@100 = proporción de fraudes reales entre las 100 tarjetas más sospechosas por día. Fuente: Elaboración propia.](docs/media/media/image12.png){alt="Gráfico, Gráfico de barras El contenido generado por IA puede ser incorrecto." width="5.905555555555556in" height="1.8458333333333334in"}

En términos puramente operativos, *Random Forest* y *XGBoost* presentaron un desempeño funcionalmente idéntico. Random Forest obtuvo una media de *CP@100* de 0.2971, superando marginalmente a *XGBoost* (0.2961). Esto significa que, de las 100 tarjetas diarias bloqueadas manualmente por los analistas, ambos modelos aseguran la intercepción de casi 30 fraudes reales.

A pesar de esta levísima ventaja de *Random Forest* en el *Top-100*, se selecciona de manera definitiva XGBoost como el modelo de referencia para el resto de la investigación. Esta decisión se fundamenta en tres factores determinantes:

1.  **Superioridad Global (AUPRC):** *XGBoost* demuestra un mejor comportamiento ponderado en todo el abanico de umbrales de probabilidad, no solo en la cabecera operativa.

2.  **Sensibilidad al Fraude (Recall):** *XGBoost* recupera casi 3 puntos porcentuales más de fraude total que *Random Forest* (61.75% frente a 58.94%).

3.  **Estabilidad Temporal:** La desviación estándar mostrada por *XGBoost* a través de los diferentes pliegues prequenciales (0.0084 en *AUPRC*) es la más baja de los tres contendientes, garantizando una menor vulnerabilidad a la deriva conceptual (*concept drift*) a lo largo del tiempo.

Establecida esta línea base y seleccionado XGBoost como algoritmo principal, en el siguiente apartado se someterá a este modelo a un análisis de integridad metodológica para evaluar el impacto de las fugas de información.

### Refinamiento del Modelo Base: Estrategia de Submuestreo (Undersampling)

A pesar de seleccionar XGBoost como el modelo de referencia más robusto, el entrenamiento sobre la distribución natural (con un ratio de desbalance aproximado de 118 transacciones legítimas por cada fraude) conlleva ineficiencias computacionales y un sesgo inherente hacia la clase mayoritaria. Para evaluar si la simplificación del espacio de datos mejora la sensibilidad del algoritmo sin recurrir a técnicas de generación sintética complejas, se implementó una variante de submuestreo controlado sobre el conjunto de entrenamiento.

Manteniendo intacto el conjunto de prueba para garantizar una evaluación honesta, se limitó la presencia de la clase mayoritaria en los pliegues de entrenamiento. La selección de los escenarios de submuestreo (10:1, 5:1 y 1:1) responde a un diseño experimental deliberado que cubre los tres regímenes operativos relevantes: submuestreo moderado (10:1, una reducción de ~12× sobre la distribución original), submuestreo agresivo (5:1, el punto habitual de compromiso en la literatura de *imbalanced learning*) y balanceo estricto (1:1, la opción más extrema). Un *Grid Search* paramétrico exhaustivo sobre el ratio de submuestreo no se consideró necesario dado que el objetivo del experimento no era encontrar la frontera de Pareto óptima, sino demostrar cualitativamente el efecto de la reducción de asimetría sobre las métricas de negocio frente al recurso al sobremuestreo sintético.

  --------------------------------------------------------------------------------
  **Escenario de Submuestreo**   **Mejor Modelo**      **AUPRC**   **CP@100**
  ------------------------------ --------------------- ----------- ---------------
  Original (\~118:1)             Random Forest         0.685       0.256

  Ratio 10:1                     XGBoost               0.659       0.279

  Ratio 5:1                      XGBoost               0.651       0.293
  --------------------------------------------------------------------------------

  : []{#_Toc225188166 .anchor}Tabla . Impacto de las estrategias de submuestreo en el rendimiento del modelo óptimo. El ratio indica la proporción de transacciones legítimas retenidas por cada fraude en el conjunto de entrenamiento. El conjunto de test se mantiene intacto con la distribución original (~118:1). AUPRC y CP@100 se reportan como media sobre folds prequenciales.

Los resultados de esta estrategia revelan dinámicas fundamentales sobre el aprendizaje en entornos de anomalías. En primer lugar, se observa que la eliminación extrema de datos de la clase mayoritaria (escenario 1:1) resulta contraproducente. Forzar una distribución equilibrada destruyó la capacidad predictiva de todos los algoritmos, colapsando el AUPRC de XGBoost a 0.497 y su precisión operativa a un inaceptable 0.113. Esto confirma que, en la detección de fraude, el modelo requiere una representación masiva de la \"normalidad\" para poder identificar eficazmente las desviaciones.

Por el contrario, las reducciones moderadas demostraron ser altamente beneficiosas. El escenario con un ratio 5:1 empleando la arquitectura XGBoost emergió como la configuración óptima para las necesidades del negocio. Si bien el ratio 10:1 logró un Área Bajo la Curva PR ligeramente superior (0.659 frente a 0.651), el entrenamiento a 5:1 maximizó la métrica de viabilidad operativa (*Card Precision@100*), alcanzando el valor más alto de todo el ensayo preliminar: 0.293.

Esta decisión subraya un principio esencial en este estudio: en un entorno de producción donde la capacidad de auditoría manual (SOC) está estrictamente limitada a las 100 principales alertas diarias, la maximización del CP@100 justifica sobradamente una degradación marginal en el AUPRC global. Adicionalmente, esta compresión del volumen de entrada aceleró de manera drástica los tiempos de optimización, consolidando a XGBoost (5:1) como una solución ágil y operativamente superior.

Aunque el marco teórico inicial contemplaba el aprendizaje sensible al costo mediante la ponderación algorítmica (scale_pos_weight en XGBoost), la fase empírica demostró que su aplicación sobre la distribución natural (\~118:1) mantenía un coste computacional prohibitivo en la fase de optimización. El submuestreo manual y controlado (ratio 5:1) resultó operativamente superior: simplificó drásticamente el espacio de características, filtró el ruido inherente a la clase mayoritaria y aceleró los tiempos de entrenamiento, logrando maximizar la métrica de negocio (CP@100) sin la sobrecarga de procesar la totalidad de las transacciones legítimas.

## Impacto de la Fuga de Datos (*Anti-Leakage*)

El siguiente experimento constituye el núcleo crítico de esta investigación. Su objetivo es ejecutar una auditoría metodológica para demostrar de forma empírica cómo las prácticas de validación defectuosas inflan artificialmente las métricas de rendimiento. Para ello, se diseñó un ensayo de control con cinco ramas experimentales aislando las principales fuentes de fuga de información (*Data Leakage*): la partición aleatoria del tiempo y la aplicación global de técnicas de escalado y sobremuestreo sintético como SMOTE (Chawla et al., 2002).

Para garantizar la consistencia del patrón, la auditoría se replicó sobre los tres modelos base (Regresión Logística, Random Forest y XGBoost).

### Evaluación bajo Metodología Deficiente (El Espejismo)

La rama metodológica denominada *Leak_todas* simuló el peor escenario posible de rigor analítico, reproduciendo deliberadamente los tres errores de diseño concurrentes: partición aleatoria de datos (*random split*), escalado global de variables y aplicación de *SMOTE* (*k_neighbors*=5, *sampling_strategy*=\'auto\') sobre la totalidad del *dataset* antes de la división en conjuntos de entrenamiento y prueba. Las consecuencias visuales y geométricas de esta contaminación se ilustran en la Figura 4.4.

![[]{#_Toc225188158 .anchor}Ilustración . Consecuencias operativas de aplicar remuestreo sintético (SMOTE) globalmente antes de la partición temporal, provocando Data Leakage. Fuente: Adaptado de Demircioğlu (2024).](docs/media/media/image13.jpeg){alt="Figure 1" width="4.8870931758530185in" height="6.9300863954505685in"}

Bajo estas condiciones de contaminación extrema, los algoritmos reportaron métricas ilusoriamente perfectas. La Regresión Logística alcanzó un AUPRC de 0.9287, mientras que los modelos de conjunto lograron memorizar el patrón por completo: Random Forest registró un AUPRC de 0.9999 y XGBoost un 0.9995. Adicionalmente, los valores de AUC ROC rozaron el 1.0 en los modelos de árboles.

Estos resultados, frecuentemente presentados en publicaciones recientes como supuestos \"avances del estado del arte\", no reflejan ninguna superioridad algorítmica. Son, por el contrario, el síntoma evidente de que el modelo ha evaluado copias sintéticas exactas que ya había internalizado durante su fase de entrenamiento, beneficiándose además del sesgo de anticipación (*look-ahead bias*) al usar eventos futuros para predecir el pasado.

### Evaluación bajo Metodología Estricta (La Realidad)

En contraposición, la rama Correcta implementó el protocolo de aislamiento estricto exigido para entornos financieros. La partición se realizó respetando la cronología temporal de las transacciones (validación prequencial), y cualquier transformación de los datos se encapsuló exclusivamente dentro del pliegue de entrenamiento (*Train*), evaluándose sobre un conjunto de prueba completamente inalterado.

Privados de la fuga de información, el rendimiento real y honesto de los algoritmos sufrió una penalización drástica. La Regresión Logística descendió a un AUPRC de 0.5830, Random Forest se situó en 0.6115 y XGBoost alcanzó un 0.6163. Esta degradación refleja la verdadera capacidad de generalización de los modelos ante vectores de fraude inéditos, subrayando la extrema dificultad del problema una vez eliminado el espejismo metodológico.

### Cuantificación de la Divergencia Algorítmica (Δ)

Para identificar qué defecto metodológico genera mayor contaminación, se aislaron las tres fuentes de fuga de datos de forma independiente. La Tabla 4.3 y la Ilustración 10 muestran el desglose del incremento marginal (inflación) en la métrica AUPRC respecto al rendimiento base honesto.

![[]{#_Toc225188159 .anchor}Ilustración . Cuantificación de la inflación artificial de la métrica AUPRC (Data Leakage) por fuente de error y modelo algorítmico. Fuente: Elaboración propia.](docs/media/media/image14.png){width="5.905555555555556in" height="1.9541666666666666in"}

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Modelo                AUPRC Honesto (Correcta)   Fuga 1: Leak_split (Solo Partición Aleatoria)   Fuga 2: Leak_scaler (Solo Escalado Global)   Fuga 3: Leak_smote (Solo SMOTE Global)   Fuga Total (Leak_todas)
  --------------------- -------------------------- ----------------------------------------------- -------------------------------------------- ---------------------------------------- -------------------------
  Regresión Logística   0.5830                     \+ 0.032 (0.615)                                \- 0.001 (0.582)                             \+ 0.007 (0.590)                         **+ 0.346** (0.929)

  Random Forest         0.6115                     \+ 0.066 (0.677)                                \- 0.004 (0.607)                             \+ 0.292 (0.904)                         **+ 0.389** (1.000)

  XGBoost               0.6163                     \+ 0.075 (0.691)                                \- 0.001 (0.615)                             \+ 0.130 (0.746)                         **+ 0.383** (0.999)
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  : []{#_Toc225188167 .anchor}Tabla . Incremento marginal absoluto de la métrica AUPRC provocado por las distintas fuentes de fuga de datos. Cada columna representa una configuración de pipeline: *Random Split* = validación aleatoria sin respetar orden cronológico; *SMOTE global* = sobremuestreo sintético aplicado antes de la partición temporal. Los valores positivos indican inflación artificial respecto al baseline prequencial correcto.

El desglose empírico arroja tres conclusiones fundamentales que validan la hipótesis central de este ensayo:

1.  **Irrelevancia del escalado global:** La aplicación del *StandardScaler* sobre la totalidad del *dataset* (Leak_scaler) no produjo inflación; de hecho, generó un impacto microscópicamente negativo (entre -0.001 y -0.004 en el AUPRC). Esto descarta el preprocesamiento de varianza como una fuente de *leakage* crítica en este conjunto de datos.

2.  **El peligro latente de las particiones estáticas:** Simplemente ignorar la cronología mediante un *split* aleatorio (Leak_split) infló el rendimiento entre un 3% y un 7.5% de forma injustificada, al permitir que el modelo \"aprendiera del futuro\" y reconociera firmas de fraude repetitivas asociadas a terminales o tarjetas comprometidas.

3.  **La contaminación masiva por SMOTE:** La inyección de datos sintéticos sobre el conjunto global de transacciones antes de su división demostró ser catastrófica para la validez del estudio. Su impacto afectó de manera desigual según la arquitectura: mientras que el modelo lineal experimentó una ligera inflación (+0.007), los algoritmos no lineales sufrieron un severo sobreajuste a los datos replicados, incrementando su AUPRC en +0.130 (XGBoost) y un masivo +0.292 (Random Forest).

La acumulación sinérgica de estos errores metodológicos eleva el AUPRC casi un 40% (0.38 Δ), creando la ilusión matemática del éxito. Este hallazgo constituye la principal evidencia del Trabajo Fin de Máster para sostener que cualquier propuesta en la literatura que reporte valores de Precisión y Recall próximos al 99% sobre este simulador sin detallar explícitamente sus mecanismos de aislamiento temporal, debe ser sometida a estricto escrutinio o descartada por invalidez metodológica.

## Interpretabilidad Algorítmica (XAI)

Tras establecer la arquitectura óptima y someterla a la auditoría de integridad metodológica, este último experimento aborda el último requisito crítico para la viabilidad operativa del sistema: la explicabilidad. En el sector financiero, fuertemente regulado por normativas como el RGPD, los modelos de \"caja negra\" de alto rendimiento carecen de utilidad si sus decisiones no pueden ser justificadas ante una auditoría.

Para este análisis, se utilizó el modelo XGBoost de referencia entrenado bajo la división temporal estricta (que arrojó un AUPRC honesto de 0.6389 y un CP@100 de 0.2729), con el objetivo de garantizar que la extracción de reglas lógicas represente al modelo que realmente sería desplegado en producción.

### Identificación Global de Patrones (Importancia Nativa)

El primer nivel de inspección consistió en extraer la Importancia de Características (*Feature Importance*) intrínseca del algoritmo XGBoost, basada en la métrica de Ganancia de Impureza (*Gain*). Esta métrica cuantifica la mejora relativa en la precisión de las ramas del árbol que aporta una determinada variable. La jerarquía de importancia nativa extraída del modelo se detalla en la Tabla 4 y visualmente en la Ilustración 11.

  ----------------------------------------------------------------------------------------------------------
  Ranking   Variable Original                     Ganancia de Impureza (Gain)   Frecuencia de uso (Weight)
  --------- ------------------------------------- ----------------------------- ----------------------------
  1         TERMINAL_ID_RISK_7DAY_WINDOW          **0.388**                     76

  2         TX_AMOUNT                             **0.129**                     330

  3         TERMINAL_ID_RISK_30DAY_WINDOW         **0.101**                     157

  4         CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW   0.056                         324

  5         TERMINAL_ID_RISK_1DAY_WINDOW          0.041                         21
  ----------------------------------------------------------------------------------------------------------

![[]{#_Toc225188168 .anchor}Tabla . Top-5 de variables predictivas según la métrica nativa de Ganancia de Impureza (Gain).](docs/media/media/image15.png){alt="Tabla El contenido generado por IA puede ser incorrecto." width="4.638930446194226in" height="2.72586176727909in"}

[]{#_Toc225188160 .anchor}Ilustración . Jerarquía de las diez variables más importantes (métrica Gain) para el modelo XGBoost base. Fuente: Elaboración propia.

El análisis revela un dominio absoluto de los perfiles de riesgo del terminal comercial. La variable TERMINAL_ID_RISK_7DAY_WINDOW se posiciona como el vector discriminativo principal, concentrando por sí sola el 38.8% de la capacidad de decisión del modelo. Si se agrupan las tres ventanas temporales de riesgo asociadas al terminal (1, 7 y 30 días), estas representan aproximadamente el 53% de la ganancia total del algoritmo. En segundo lugar, la magnitud de la transacción (TX_AMOUNT) aporta un 12.9% de ganancia. Estos resultados validan la lógica de negocio subyacente: el modelo no está memorizando ruido estocástico, sino que ha aprendido que el fraude tiende a concentrarse en datáfonos o pasarelas de pago previamente comprometidas y con importes superiores a la media.

### Direccionalidad del Riesgo Algorítmico (SHAP)

Aunque la métrica nativa resulta útil, es estática y no proporciona información sobre la direccionalidad del impacto. Para superarlo, se aplicó la Teoría de Juegos mediante la librería SHAP (*SHapley Additive exPlanations*), evaluando una muestra de 1.000 transacciones del conjunto de prueba.

El análisis del impacto medio absoluto (mean \|SHAP\|) reveló una discrepancia notable frente a la métrica intrínseca del árbol. Mientras que la ganancia estructural (*Gain*) prioriza los nodos superiores asociados al riesgo del terminal, el impacto marginal exacto por predicción individual (SHAP) está dominado por el comportamiento del usuario. Variables como CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW y TX_AMOUNT registraron los mayores impactos marginales (1.29 y 0.79, respectivamente), indicando que las desviaciones abruptas del patrón de gasto habitual del titular son el detonante final que inclina la balanza probabilística de cada transacción.

![[]{#_Toc225188161 .anchor}Ilustración . Gráfico SHAP Beeswarm de explicabilidad global para el modelo XGBoost de referencia (1000 muestras aleatorias de test). Cada punto representa una transacción; la posición en el eje horizontal indica la magnitud del valor SHAP (contribución marginal a la predicción de fraude), y el color codifica el valor de la característica (azul = bajo, rojo = alto). Las variables se ordenan de arriba a abajo según su importancia media absoluta (mean |SHAP|). CP@100 = Card Precision@100. Fuente: Elaboración propia.](docs/media/media/image16.png){width="4.574099956255468in" height="2.857199256342957in"}

El gráfico de enjambre (*Beeswarm plot*, Ilustración 12) confirma visualmente las hipótesis direccionales: valores altos en las ventanas de riesgo del terminal (puntos rojos) generan vectores de fuerza fuertemente positivos (empujan hacia la clase Fraude), mientras que valores bajos (puntos azules) actúan como anclas de legitimidad. Análogamente, montos inusualmente elevados rompen la normalidad comportamental y disparan la probabilidad de riesgo.

### Auditoría Forense a Nivel de Transacción (Fuerza Local)

Finalmente, se evaluó la utilidad operativa de la explicabilidad simulando el flujo de trabajo de un centro de operaciones de seguridad (SOC). En la práctica, un analista antifraude requiere \"códigos de razón\" (*reason codes*) para justificar el bloqueo preventivo de una tarjeta específica, tal y como se desglosa en la Ilustración 13.

![[]{#_Toc225188162 .anchor}Ilustración . Explicabilidad local mediante gráficos SHAP Waterfall, desglosando la contribución marginal de variables para un fraude detectado y una transacción legítima. Fuente: Elaboración propia.](docs/media/media/image18.svg){width="5.095238407699037in" height="3.8981200787401575in"}

Para demostrar esta capacidad, se generaron explicaciones locales (*Force Plots*) enriquecidas con el contexto descriptivo de la transacción.

![Texto El contenido generado por IA puede ser incorrecto.](docs/media/media/image19.png){width="5.905555555555556in" height="1.6958333333333333in"} ![Escala de tiempo El contenido generado por IA puede ser incorrecto.](docs/media/media/image20.png){width="5.905555555555556in" height="1.9736111111111112in"}

[]{#_Toc225188163 .anchor}Ilustración . Códigos de razón (reason codes) simulados en un SOC mediante gráficos SHAP Force Plot para una transacción fraudulenta y una legítima. Fuente: Elaboración propia.

La observación de la Ilustración 14 permite reconstruir la narrativa matemática de una alarma. El algoritmo no emitió el bloqueo por puro azar, sino porque identificó un incremento simultáneo en factores de riesgo determinantes: un alto monto combinado con una coincidencia en un terminal catalogado como de alto riesgo en la ventana de 7 días. Por el contrario, en operaciones legítimas (Figura 4.7), un historial de gasto predecible y la ausencia de alertas recientes en el comercio actúan como fuerzas estabilizadoras.

Esta transición de un modelo predictivo opaco hacia un sistema de decisión trazable concluye el proceso de validación técnica, demostrando que la solución propuesta no solo es matemáticamente precisa y resistente a las fugas de información, sino que es auditable y directamente integrable en procesos humanos de toma de decisiones financieras

### Validación de Relevancia mediante Estudio de Ablación

Para dotar de rigor empírico a los hallazgos de interpretabilidad, se diseñó una última prueba de esfuerzo metodológica: un estudio de ablación. El objetivo de esta fase fue comprobar si el alto impacto marginal atribuido por SHAP a la variable dominante del comportamiento de usuario (CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW) se traducía verdaderamente en una dependencia predictiva crítica por parte del modelo.

El experimento consistió en extirpar deliberadamente dicha característica del conjunto de datos y someter al modelo XGBoost de referencia a un reentrenamiento completo bajo el mismo rigor de aislamiento temporal. Los resultados de este proceso se resumen en la Tabla 5.

  -------------------------------------------------------------------------
  Modelo Evaluado                      AUC ROC      AUPRC       CP@100
  ------------------------------------ ------------ ----------- -----------
  XGBoost (Conjunto Completo)          0.8618       0.6389      0.2729

  XGBoost (Ablación Top-Feature)       0.8498       0.5942      0.2714

  Diferencial (\$\\Delta\$)            \- 0.0121    \- 0.0447   \- 0.0014
  -------------------------------------------------------------------------

  : []{#_Toc225188169 .anchor}Tabla . Impacto empírico en el rendimiento algorítmico tras la ablación de la característica dominante (CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW, identificada como la de mayor mean |SHAP|). Δ = diferencia ablado − completo. La degradación del AUPRC fue consistente en los 4 folds prequenciales (Wilcoxon *p* = 0.125; el mínimo alcanzable con *N* = 4 es 0.0625).

La supresión de la variable confirmó la validez del análisis XAI previo. Como detalla la tabla, el rendimiento técnico del modelo sufrió una erosión inmediata, evidenciada por una caída del 4.47% absoluto en la métrica principal AUPRC (descendiendo de 0.6389 a 0.5942) y una ligera pérdida de capacidad de separación global reflejada en el AUC ROC (-0.0121). Para evaluar la significancia de esta degradación, se replicó el estudio de ablación sobre los 4 folds prequenciales. El test de Wilcoxon arrojó un *p*-valor de 0.125 para AUPRC y AUC ROC, lo que no permite rechazar la hipótesis nula al nivel convencional α = 0.05. No obstante, con *N* = 4 folds el p-valor mínimo alcanzable por Wilcoxon es 0.0625, lo que impide la confirmación estadística formal con este tamaño muestral. La degradación observada fue consistente en los cuatro folds (la AUPRC del modelo ablado fue inferior en todos ellos), lo que sugiere un efecto direccional robusto aunque no formalmente significativo.

Resulta igualmente revelador analizar la resiliencia de la arquitectura subyacente. Tras la ablación, un nuevo análisis de los valores absolutos medios de impacto redefinió la jerarquía de decisiones del algoritmo. La magnitud bruta de la transacción (TX_AMOUNT) ascendió a la primera posición como vector principal (impacto marginal de 1.027), seguida estrechamente por la ventana de gasto del cliente a corto plazo (CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW, con 0.966). Esto nos indica que al eliminar la característica más trascendente, otras características escalan su importancia de forma proporcional (TX_AMOUNT) y otras de forma exponencial (CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW).

![[]{#_Toc225188164 .anchor}Ilustración . Reconfiguración de la jerarquía de explicabilidad (SHAP Beeswarm) tras el estudio de ablación de la característica dominante. Fuente: Elaboración propia.](docs/media/media/image21.png){width="5.913888888888889in" height="2.5805555555555557in"}

El hecho de que la métrica de viabilidad operativa (CP@100) apenas sufriera una penalización marginal de -0.0014 se explica con mayor probabilidad por la alta multicolinealidad intrínseca entre las ventanas temporales construidas en la fase de ingeniería de características. El análisis de correlación de Pearson confirma que CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW y CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW presentan un coeficiente *r* = 0.957, mientras que la ventana a 1 día correlaciona con la de 7 días en *r* = 0.868. Esta redundancia informativa permite al modelo, al perder la variable dominante, apoyarse en variables proxy que contienen casi la misma señal empírica, manteniendo así la eficacia del sistema en las alertas de máxima prioridad sin necesidad de invocar una resiliencia intrínseca de la arquitectura *Gradient Boosting*.

Para garantizar que la variación en el rendimiento se debiera exclusivamente a la ausencia de la característica extirpada, se decidió mantener estáticos los hiperparámetros originales del modelo XGBoost base. Una reoptimización posterior a la ablación habría introducido una variable de confusión, impidiendo aislar el valor predictivo intrínseco de la variable analizada frente a la capacidad de adaptación de la arquitectura del modelo.

# **Conclusiones**

La presente investigación se articuló con el propósito de auditar, desde una perspectiva crítica y operativa, el ciclo de vida completo de los modelos de *Machine Learning* aplicados a la detección de fraude en tarjetas de crédito. A través de un diseño experimental progresivo, se ha trascendido la mera búsqueda de rendimiento algorítmico para abordar problemas estructurales del dominio, tales como la asimetría extrema de clases, la propagación de sesgos metodológicos (*Data Leakage*) y la opacidad en la toma de decisiones.

Los resultados empíricos obtenidos, evaluados sobre datos simulados en un entorno controlado, permiten extraer conclusiones que validan las hipótesis iniciales y aportan directrices metodológicas relevantes como prueba de concepto avanzada para futuras implementaciones en entornos financieros productivos.

## Síntesis de Hallazgos y Validación de Hipótesis

El análisis de los ensayos metodológicos arroja tres conclusiones técnicas fundamentales:

### **La falacia de las métricas generalistas y el valor del submuestreo:**

El experimento con el que se definió el modelo de referencia, demostró empíricamente que la Exactitud (*Accuracy*) carece de validez como indicador de éxito en entornos de fraude, al invisibilizar la tasa de falsos negativos bajo el peso de la clase mayoritaria. Adicionalmente, se constató que las estrategias de balanceo extremo (1:1) destruyen la semántica de la normalidad, degradando severamente la capacidad predictiva. Por el contrario, un submuestreo moderado (ratio 5:1) empleando la arquitectura *XGBoost* emergió como la solución óptima, incrementando drásticamente el volumen de fraude interceptado y reduciendo los tiempos de optimización, consolidándose como un modelo ágil y altamente sensible.

### **Cuantificación empírica del Data Leakage en la literatura:**

El experimento sobre el impacto de la fuga de datos constituye la aportación más crítica de este trabajo. Se ha cuantificado empíricamente cómo prácticas frecuentes en la investigación actual inflan artificialmente la métrica AUPRC hasta en un 40%. Este hallazgo cuestiona la validez operativa de aquellos modelos que reportan eficacias cercanas al 99% sin aplicar un protocolo estricto de validación prequencial, sugiriendo que parte de dicho rendimiento podría derivarse de la memorización de patrones temporalmente contaminados en lugar de la detección genuina de fraude inédito.

### **La transición hacia la \"Caja Blanca\" validada:**

El experimento de la interpretabilidad algorítmica confirmó que el alto rendimiento predictivo no tiene por qué sacrificar la transparencia. La integración de Teoría de Juegos (valores SHAP) permitió revelar que el modelo fundamenta sus decisiones en principios lógicos de negocio (riesgo histórico del terminal a corto plazo y alteraciones bruscas en el gasto del cliente), alejando cualquier sospecha de ajuste sobre ruido estocástico. Más relevante aún, el estudio de ablación demostró empíricamente la robustez técnica del análisis XAI: al extirpar la variable catalogada como dominante, el modelo sufrió una penalización medible en su rendimiento global (caída consistente del AUPRC en los cuatro folds prequenciales), mientras que la mínima degradación en CP@100 se atribuye a la alta multicolinealidad entre las ventanas temporales RFM (*r* = 0.957 entre las ventanas de 7 y 30 días de gasto medio), que actúan como variables proxy redundantes.

## Implicaciones para el Negocio Bancario

Desde una perspectiva operativa, las conclusiones de este Trabajo Fin de Máster subrayan la necesidad de alinear la métrica matemática con la capacidad humana.

La adopción de la métrica *Card Precision@100* (CP@100) como criterio de decisión ha evidenciado que el mejor modelo no es aquel que detecta teóricamente más anomalías, sino el que maximiza la densidad de fraudes reales en la cabecera del sistema de alertas. El modelo final desarrollado garantiza la intercepción de casi 30 tarjetas comprometidas diarias asumiendo una restricción estricta de 100 revisiones manuales por parte del Centro de Operaciones de Seguridad (SOC).

Paralelamente, la generación de \"códigos de razón\" (*reason codes*) a través de los gráficos de fuerza local (SHAP *Force Plots*) resuelve la barrera regulatoria. Dotar a los analistas humanos de una explicación visual, transacción por transacción, sobre qué vectores de riesgo detonaron un bloqueo preventivo, permite a la entidad bancaria justificar sus acciones ante los clientes, mitigar el daño reputacional de los falsos positivos y cumplir con los requisitos de explicabilidad algorítmica exigidos por la normativa europea.

## Limitaciones del Estudio

La validez de cualquier investigación empírica en ingeniería de datos está supeditada a las condiciones de contorno de su diseño experimental. Para interpretar correctamente el alcance de los resultados de este Trabajo Fin de Máster, es preceptivo reconocer las siguientes limitaciones estructurales:

### **La brecha semántica de la simulación de datos:**

Para sortear las restricciones de confidencialidad impuestas por la regulación bancaria y garantizar la trazabilidad absoluta de la \"verdad fundamental\" (*ground truth*), este estudio se ha fundamentado íntegramente en el *Transaction Data Simulator* de la ULB. Aunque esta herramienta mimetiza con alta fidelidad las distribuciones estocásticas de transacciones legítimas y superpone escenarios de ataque realistas (compromiso de terminales), no deja de ser un entorno determinista acotado. En el mundo real, el fraude financiero es un juego del gato y el ratón en constante evolución (*concept drift* dinámico), fuertemente influenciado por variables exógenas (ataques coordinados, filtraciones masivas de credenciales o ingeniería social) que el simulador no tiene capacidad de replicar.

### **Vulnerabilidad ante ataques emergentes (Zero-Day):**

La implementación de un periodo de bloqueo (*gap*) de 7 días en la validación prequencial es necesaria para simular el tiempo real de confirmación de fraude y evitar el *data leakage*. Sin embargo, esta latencia introduce un riesgo operativo: el modelo es temporalmente ciego a nuevos vectores de ataque que ocurran durante esa ventana, careciendo de la señal de entrenamiento necesaria para detectarlos hasta el siguiente ciclo.

### **Latencia de Inferencia XAI:**

Si bien la integración de SHAP satisface la exigencia regulatoria de transparencia, impone una sobrecarga computacional cuantificable. Las mediciones empíricas realizadas sobre el modelo XGBoost de referencia (100 árboles, max_depth=6, 15 características) arrojan una latencia de inferencia de ~0.33 ms por transacción individual para `predict_proba`, a la que el cómputo de TreeSHAP añade ~0.89 ms adicionales, totalizando ~1.2 ms por transacción. En modo batch (100 transacciones), la amortización reduce el coste combinado a ~0.07 ms/transacción. Estas cifras sitúan la inferencia pura dentro de los márgenes aceptables para un sistema de autorización de pagos en tiempo real (típicamente < 100 ms end-to-end). No obstante, en un entorno de producción de alta frecuencia (miles de transacciones por segundo), computar explicaciones SHAP para *cada* transacción podría convertirse en un cuello de botella. Una estrategia operativamente viable consistiría en generar los valores SHAP únicamente para las transacciones que superen el umbral de riesgo del modelo (*near-real-time*), reservando la explicabilidad completa para el subconjunto de alertas que requiera revisión por parte del SOC, en lugar de aplicarla indiscriminadamente a todo el volumen transaccional.

### **El sesgo de la Explicabilidad (XAI) sobre reglas sintéticas:**

Derivado del uso de datos simulados, la extracción de patrones lógicos mediante SHAP debe interpretarse como una validación de la *arquitectura de auditoría*, no como el descubrimiento de nueva psicología criminal. SHAP está haciendo ingeniería inversa sobre cómo los creadores del simulador programaron el fraude, no necesariamente sobre un estafador humano real.

### **Restricción del espacio de características (Feature Space):**

Los algoritmos evaluados han basado sus decisiones en un vector de características estrictamente transaccional (importes, temporalidad y métricas RFM). En un sistema de detección de fraude en producción contemporáneo, el motor de aprendizaje automático se enriquece con cientos de variables no transaccionales, tales como la telemetría del dispositivo (*device fingerprinting*), la geolocalización IP, biometría comportamental (cadencia de tecleo) o información del ecosistema 3D-Secure. La ausencia de estas dimensiones limita la capacidad predictiva máxima teórica alcanzable en el presente ensayo comparado con un entorno comercial en vivo.

### **Potencia estadística limitada de los tests de significancia:**

La estrategia de validación prequencial con ventanas semanales sobre un dataset de 6 meses permite un máximo de *N* = 4 folds independientes. Este tamaño muestral impone un límite inferior al p-valor del test de Wilcoxon de 0.0625 (una cola), lo que imposibilita alcanzar significancia al nivel convencional α = 0.05 en comparaciones pareadas. Si bien los tests omnibus (Friedman) sí detectan diferencias globales significativas entre los tres algoritmos, la confirmación estadística formal de superioridad par a par requeriría un mayor número de folds (ej. evaluación mensual sobre múltiples años de datos) o la disponibilidad de un dataset de mayor extensión temporal.

## Líneas de Investigación Futuras

La presente investigación establece un marco fundacional robusto para la evaluación justa y transparente de modelos de detección de fraude. Partiendo de las limitaciones identificadas y del conocimiento adquirido, se proponen las siguientes líneas de desarrollo para proyectar este trabajo hacia su siguiente fase de madurez tecnológica:

### **Análisis de Sensibilidad sobre la Latencia de Etiquetado:**

En el presente estudio, la validación prequencial asume un retraso de verificación (*feedback delay*) estático de 7 días, mimetizando el tiempo medio que transcurre en el mundo real hasta que un titular detecta un cargo ilícito y el banco confirma el fraude. Si bien este parámetro garantiza la ausencia de sesgo de anticipación (*look-ahead bias*), su optimización paramétrica queda fuera del alcance analítico del presente Trabajo Fin de Máster.

Como línea de investigación prioritaria, resultaría de alto valor operativo plantear un análisis de sensibilidad temporal. Este experimento consistiría en evaluar la degradación del Área Bajo la Curva PR (AUPRC) variando sistemáticamente la ventana de retraso (ej. 1, 7, 14 y 30 días). Dicho análisis permitiría a una entidad bancaria cuantificar exactamente cuánto rendimiento predictivo pierde por cada día de demora en sus procesos de reclamación de clientes, facilitando así un análisis de coste-beneficio sobre la rentabilidad de agilizar sus canales de atención al cliente.

### **Transición hacia Arquitecturas de Deep Learning Secuencial:**

Aunque la arquitectura *XGBoost* combinada con la ingeniería de características RFM ha demostrado ser altamente competente, este enfoque tabular obliga a resumir la historia del cliente en agregaciones estáticas (medias, conteos). El siguiente paso natural es explorar el paradigma del *Deep Learning*, modelando las transacciones como secuencias temporales puras. La implementación de Redes Neuronales Recurrentes (RNN, LSTM) o arquitecturas basadas en mecanismos de Atención (*Transformers*) permitiría al modelo ingerir el historial completo de un usuario y aprender dependencias a largo plazo de forma nativa, sin depender de la creación manual de ventanas temporales.

### **Análisis Topológico mediante Grafos (Graph Neural Networks):**

El fraude organizado rara vez actúa de forma aislada; opera a través de redes de terminales comprometidos y tarjetas clonadas que interactúan entre sí. Una prometedora línea de mejora consistiría en abandonar la visión individual de la transacción para adoptar un enfoque topológico. Utilizando *Graph Neural Networks* (GNN), se podría modelar a los clientes y terminales como nodos, y las transacciones como aristas, permitiendo que el sistema detecte \"anillos\" de fraude o estructuras criminales complejas (como el lavado de dinero o la triangulación de pagos) analizando la densidad y las anomalías en la estructura del grafo del ecosistema financiero.

# **Referencias** 

Abdulghani, A. Q., Uçan, O. N., & Alheeti, K. M. A. (2021). Credit card fraud detection using XGBoost algorithm. *2021 14th International Conference on Developments in eSystems Engineering (DeSE)*, 487-492. <https://doi.org/10.1109/DeSE54285.2021.9719584>

Agrahari, S., Srivastava, S., & Singh, A. K. (2023). Review on novelty detection in the non-stationary environment. *Knowledge and Information Systems*, *65*(3), 1549--1574. <https://doi.org/10.1007/s10115-023-02018-x>

Baesens, B., Van Vlasselaer, V., & Verbeke, W. (2015). *Fraud analytics using descriptive, predictive, and social network techniques: a guide to data science for fraud detection*. John Wiley & Sons.

Bracke, P., Datta, A., Jung, C., & Sen, S. (2019). Machine learning explainability in finance: an application to default risk analysis. *Bank of England Working Paper*, No. 816. [https://ssrn.com/abstract=3436449](https://www.google.com/search?q=https://ssrn.com/abstract%3D3436449)

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. <https://doi.org/10.1023/A:1010933404324>

Bücker, M., Szepannek, G., Rostamzadeh, A., & Rosen, H. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. *Journal of the Operational Research Society*, 73(1), 70-90. <https://doi.org/10.1080/01605682.2021.1922098>

Carcillo, F., Le Borgne, Y. A., Caelen, O., Khedr, A., & Bontempi, G. (2021). Combining unsupervised and supervised learning in credit card fraud detection. *Information Sciences*, 557, 317-331. [https://doi.org/10.1016/j.ins.2020.12.059](https://www.google.com/search?q=https://doi.org/10.1016/j.ins.2020.12.059)

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys (CSUR)*, 41(3), 1-58. <https://doi.org/10.1145/1541880.1541882>

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. <https://doi.org/10.1613/jair.953>

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. <https://doi.org/10.1145/2939672.2939785>

Correa Bahnsen, A., Aouada, D., Stojanovic, A., & Ottersten, B. (2016). Feature engineering strategies for credit card fraud detection. *Expert Systems with Applications*, 51, 134-142. <https://doi.org/10.1016/j.eswa.2015.12.030>

Correa Bahnsen, A., Stojanovic, A., Aouada, D., & Ottersten, B. (2015). Example-dependent cost-sensitive decision trees. *Expert Systems with Applications*, 42(19), 6609-6619. [https://doi.org/10.1016/j.eswa.2015.04.042](https://www.google.com/search?q=https://doi.org/10.1016/j.eswa.2015.04.042)

Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, 41(10), 4915-4928. <https://doi.org/10.1016/j.eswa.2014.02.026>

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *2015 IEEE Symposium Series on Computational Intelligence*, 159-166. <https://doi.org/10.1109/SSCI.2015.33>

Demircioğlu, A. (2024). Applying oversampling before cross-validation will lead to high bias in radiomics. *Scientific Reports*, *14*(1), 11563. <https://doi.org/10.1038/s41598-024-62585-z>

Esghir, M., Jilal, A., & Elomri, A. (2025). Credit Card Fraud Detection: Overcoming Data Leakage and Class Imbalance. *IEEE Transactions on Information Forensics and Security*, 20, 112-125. [https://doi.org/10.1109/TIFS.2025.3214567](https://www.google.com/search?q=https://doi.org/10.1109/TIFS.2025.3214567)

Fernández, A., Garcia, S., Herrera, F., & Chawla, N. V. (2018). SMOTE for learning from imbalanced data: progress and challenges, marking the 15-year anniversary. *Journal of artificial intelligence research*, 61, 863-905. <https://doi.org/10.1613/jair.1.11192>

Fiore, U., De Santis, A., Perla, F., Zanetti, P., & Palmieri, F. (2019). Using generative adversarial networks for improving classification effectiveness in credit card fraud detection. *Information Sciences*, 479, 448-455. <https://doi.org/10.1016/j.ins.2018.12.030>

Gama, J., Žliobaite, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM computing surveys (CSUR)*, 46(4), 1-37. <https://doi.org/10.1145/2523813>

Gómez, J. A., & Silva, M. (2024). Concept Drift and Temporal Validation in Financial Anomaly Detection. *IEEE Access*, 12, 45102-45115.

Hafidurrohman, M. (2026). Application of Random Forest and XGBoost for Credit Card Fraud Detection with Unbalanced Data. *Khatulistiwa Smart Journal of Artificial Intelligence*, 3(1). <https://journal.literasikhatulistiwa.org/index.php/kjarti/article/download/229/36>

Hasan, N. G. M. R., & Gazi, M. S. (2025). Explainable AI for credit card fraud detection: Bridging the gap between accuracy and interpretability. *World Journal of Advanced Research and Reviews*, 25(1). <https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-0492.pdf>

Hayat, K., & Magnier, B. (2025). Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies. *Mathematics*, 13(16), 2563. <https://doi.org/10.3390/math13162563>

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. <https://doi.org/10.1109/TKDE.2008.239>

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. <https://otexts.com/fpp3/>

Ileberi, E., Sun, Y., & Wang, Z. (2021). Performance Evaluation of Machine Learning Methods for Credit Card Fraud Detection Using SMOTE and AdaBoost. *IEEE Access*, 9, 165286-165294. <https://doi.org/10.1109/ACCESS.2021.3134330>

Iqbal, T., et al. (2025). Credit Card Fraud Detection Through Explainable Artificial Intelligence for Managerial Oversight. *Heca Sentra Analitika*. <https://heca-analitika.com/ijma/article/download/301/204/2406>

Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook. *Université Libre de Bruxelles*. <https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook>

Li, W. (2024). Credit Card Fraud Detection: A System Based on Imbalanced Learning and Ensemble Models. *Atlantis Press*. <https://www.atlantis-press.com/article/126022073.pdf>

Lucas, Y., Portier, P. E., Lapuyade-Lahorgue, L., Calabretto, S., Lulek, L., & Brunie, L. (2020). Towards automated machine learning for credit card fraud detection. *Knowledge-Based Systems*, 195, 105650. [https://doi.org/10.1016/j.knosys.2020.105650](https://www.google.com/search?q=https://doi.org/10.1016/j.knosys.2020.105650)

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Makki, S., Assaghir, Z., Taher, Y., Haque, R., Hacid, M. S., & Zeineddine, H. (2019). An experimental study with imbalanced classification approaches for credit card fraud detection. *IEEE Access*, 7, 93010-93022. <https://doi.org/10.1109/ACCESS.2019.2927266>

Molnar, C. (2020). *Interpretable machine learning: A guide for making black box models explainable* (2nd ed.). <https://christophm.github.io/interpretable-ml-book/>

Pragna, L. (2025). Credit Card Fraud Detection Using Machine Learning. *Stephen F. Austin State University*.

Sadgali, I., Sael, N., & Benabbou, F. (2021). Adaptive Model for Credit Card Fraud Detection. *International Journal of Interactive Mobile Technologies*, 14(3).

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PloS one*, 10(3), e0118432. <https://doi.org/10.1371/journal.pone.0118432>

Shanaa, M. (2025). Credit Card Fraud Detection using Explainable AI Methods. *Journal of Information Systems Engineering and Management*.

Taha, A. A., & Malebary, S. J. (2020). An intelligent approach to credit card fraud detection using an optimized light gradient boosting machine. *IEEE Access*, 8, 25579-25587. [https://doi.org/10.1109/ACCESS.2020.2971354](https://www.google.com/search?q=https://doi.org/10.1109/ACCESS.2020.2971354)

The Nilson Report. (2025). *Card Fraud Losses Worldwide --- 2024* (Issue 1298). Recuperado de <https://nilsonreport.com/articles/card-fraud-losses-worldwide-2024/>

## Anexos {#anexos .Apartado}
