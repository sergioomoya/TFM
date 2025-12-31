(Model_Validation_And_Selection)=
# Introducción

El propósito de un modelo de predicción es proporcionar predicciones precisas sobre datos *nuevos*, es decir, sobre datos que no se utilizan para entrenar el modelo. A esto se le llama la *capacidad de generalización* de un modelo de predicción. El rendimiento predictivo que se puede esperar de un modelo de predicción en datos nuevos se denomina *rendimiento de generalización* o *rendimiento de prueba*, y es lo que se debe maximizar.  

El rendimiento predictivo de un modelo de predicción en los datos de entrenamiento, llamado *rendimiento de entrenamiento*, es a menudo un mal indicador del rendimiento de generalización. Los modelos de predicción contienen hiperparámetros que les permiten ajustarse más o menos a los datos de entrenamiento. Elegir hiperparámetros que ajusten estrechamente un modelo a los datos de entrenamiento, por ejemplo aumentando la expresividad, generalmente resulta en una pérdida en términos de rendimiento de generalización, un fenómeno conocido como *sobreajuste* (overfitting). Esto se observó, por ejemplo, en la Sección {ref}`Baseline_FDS_Performances_Simulation` con árboles de decisión. Un árbol de decisión con profundidad ilimitada fue capaz de detectar todos los fraudes en el conjunto de entrenamiento. Sin embargo, tuvo tasas de detección de fraude muy pobres en el conjunto de prueba y una tasa de detección más baja que un árbol de decisión de profundidad dos. 

El enfoque estándar para evaluar la capacidad de generalización de un modelo de predicción es un proceso conocido como *validación*. La validación consiste en dividir los datos históricos en dos conjuntos. El primero se utiliza para el entrenamiento. El segundo, llamado *conjunto de validación*, se utiliza para evaluar (validar) la capacidad de generalización del modelo. El diagrama que resume la metodología de validación, como se presentó en el Capítulo 2 - {ref}`ML_For_CCFD`, se reproduce en la Fig. 1.    

![alt text](images/baseline_ML_workflow_subset2.png)
<p style="text-align: center;">
Fig. 1. Aprendizaje automático para CCFD: Metodología base. El proceso de validación (parte superior, cuadro verde) se basa en un conjunto de validación para estimar la capacidad de generalización de diferentes modelos de predicción.   
</p>

El proceso de validación permite estimar la capacidad de generalización de diferentes modelos de predicción gracias al conjunto de validación. El proceso se puede utilizar para comparar la capacidad de generalización con diferentes hiperparámetros del modelo, clases de modelos o técnicas de ingeniería de características. El modelo que proporciona la mejor capacidad de generalización estimada se selecciona finalmente y se despliega para producción. 

En la detección de fraude de tarjetas de crédito, el propósito de un modelo de predicción es más específicamente proporcionar predicciones precisas para transacciones que ocurrirán *en el futuro*. Debido a la naturaleza secuencial de los datos de transacciones, se debe tener especial cuidado al dividir los datos en conjuntos de entrenamiento y validación. En particular, las transacciones del conjunto de validación deben ocurrir *después* de las transacciones del conjunto de entrenamiento.  

Este capítulo explora las estrategias de validación que se pueden utilizar para problemas de detección de fraude. La [Sección 5.2](Validation_Strategies) cubre primero tres tipos de estrategias de validación conocidas como *hold-out*, *hold-out repetido* y *validación prequential*. La [Sección 5.3](Model_Selection) luego discute la selección de modelos y estrategias de optimización que se pueden utilizar para explorar de manera más eficiente el espacio de modelos competidores. 


 