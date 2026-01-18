(Deep_learning)=
# Introducción

Los modelos más utilizados para la detección de fraudes en la industria o en competiciones de aprendizaje automático {cite}`kaggle2019fraud` son algoritmos de boosting de gradiente como XGBoost {cite}`chen2016xgboost`, LightGBM {cite}`ke2017lightgbm`, CatBoost {cite}`prokhorenkova2017catboost`, y modelos basados en árboles como random forest {cite}`breiman2001random`. Con el preprocesamiento y la ingeniería de características adecuados, estos modelos proporcionan resultados muy convincentes en sistemas de detección de fraudes del mundo real.

Los algoritmos de redes neuronales se consideran menos a menudo en los benchmarks de fraude con datos estáticos, ya que son más difíciles de ajustar para alcanzar un rendimiento predictivo competitivo. Sin embargo, tienen muchas ventajas que los hacen esenciales en la caja de herramientas de un profesional de detección de fraudes.

## ¿Por qué utilizar una red neuronal para la detección de fraude?

No hay razón para asumir que una red neuronal feed-forward multicapa podría superar a random forest o XGBoost en conjuntos de datos estáticos, pero existen varios otros criterios importantes en el problema de detección de fraudes además del rendimiento de detección.

### Aprendizaje incremental

XGBoost y random forest son ambos conjuntos de árboles. Los árboles de decisión generalmente no son incrementales porque requieren el conjunto de datos completo para calcular las divisiones óptimas y construir su estructura. Modificar una división dado un nuevo conjunto de datos está lejos de ser trivial. En particular, como se construye jerárquicamente, actualizar una condición en una división alta de un árbol hace que la estructura de los subárboles sea inutilizable directamente. Vale la pena señalar que se han propuesto varias técnicas para actualizar árboles de manera incremental, como árboles Hoeffding {cite}`domingos2000mining`, árboles Mondrian {cite}`lakshminarayanan2014mondrian`, o conjuntos incrementales de árboles {cite}`sun2018concept`. Sin embargo, los algoritmos basados en árboles siguen utilizándose principalmente en un escenario de aprendizaje por lotes.

El aprendizaje incremental es útil para la detección de fraudes porque (1) es menos intensivo en recursos ya que los modelos se pueden actualizar a menudo en los últimos fragmentos de datos en lugar de tener que ser entrenados completamente en todo el conjunto de datos desde cero cada vez, y (2) descarta la necesidad de almacenar datos históricos durante mucho tiempo, evitando así problemas de regulación de datos.

Las redes neuronales tienen la ventaja de ser incrementales por naturaleza, ya que su entrenamiento es iterativo y por instancias.

### Aprendizaje de representación y entrenamiento de extremo a extremo

Muchos estudios han demostrado que, además de las características de transacción sin procesar, el uso de ingeniería de características experta (construcción de agregados relevantes basados en el historial de transacciones del titular de la tarjeta) mejora significativamente la tasa de detección de fraudes {cite}`bahnsen2016feature,dal2014learned`.

Sin embargo, este proceso tiene limitaciones, principalmente la de depender del costoso conocimiento experto humano. Ha habido intentos de reemplazar la agregación manual a través del aprendizaje automático de representaciones {cite}`fu2016credit,jurgovsky2018sequence,dastidar2020nag`. Estos métodos se basan principalmente en redes neuronales (Autoencoders, redes neuronales convolucionales, redes de memoria a corto y largo plazo).

Además, encima de estas representaciones aprendidas, usar una red neuronal feed-forward en lugar de XGBoost o random forests es más interesante, ya que permite entrenar todo el modelo (parte de representación + parte de clasificación) de un extremo al otro.

### Aprendizaje federado

El aprendizaje federado consiste en compartir y entrenar un modelo en múltiples dispositivos, manteniendo cada dispositivo sus datos localmente. La idea es compartir un modelo inicial entre los dispositivos, actualizarlo localmente y federar frecuentemente las actualizaciones de todos los dispositivos en un modelo global para todos. En general, la actualización global se calcula con métodos como el promedio federado {cite}`konevcny2016federated`, es decir, a través de un promedio ponderado de los pesos de cada modelo local.

A diferencia de los modelos basados en árboles, las redes neuronales con la misma arquitectura pueden tener sus pesos promediados, lo que las convierte en la primera opción cuando se trata de aprendizaje federado.

### Un modelo adicional para apilamiento (stacking)

Aunque las redes neuronales pueden alcanzar un rendimiento global cercano a XGBoost o random forests, esto no significa que estos diferentes modelos capturen los mismos patrones de fraude. En particular, los experimentos a menudo muestran que combinar un enfoque basado en árboles y una red neuronal en un conjunto de promedio simple puede conducir, gracias a la diversidad, a un mejor rendimiento general.

### Mensaje clave

Aparte del rendimiento de detección, las redes neuronales tienen varias ventajas para el problema de detección de fraude con tarjetas de crédito: se pueden apilar con otros modelos, se pueden entrenar incrementalmente, se pueden federar fácilmente, permiten el aprendizaje de representaciones y pueden aprender representaciones y clasificación juntas con entrenamiento de extremo a extremo.

## Contenido del capítulo

Este capítulo cubre técnicas para construir redes neuronales para el problema de detección de fraudes. La Sección 2 describe consideraciones generales para diseñar un primer modelo (red neuronal feed-forward totalmente conectada). Las siguientes secciones exploran técnicas de aprendizaje profundo más avanzadas para aprender representaciones útiles de los datos. Las secciones 3 y 4 describen respectivamente el uso de autoencoders y modelos secuenciales (Redes neuronales convolucionales, redes de memoria a corto y largo plazo, y mecanismo de atención). Finalmente, la sección 5 describe los resultados de todos los métodos en datos del mundo real, para comparar con los métodos por lotes del capítulo 5.
