(ML_For_CCFD)=
# Aprendizaje automático para la detección de fraude con tarjetas de crédito

La detección de fraude con tarjetas de crédito (CCFD) es como buscar agujas en un pajar. Requiere encontrar, entre millones de transacciones diarias, cuáles son fraudulentas. Debido a la cantidad cada vez mayor de datos, ahora es casi imposible para un especialista humano detectar patrones significativos a partir de los datos de transacciones. Por esta razón, el uso de técnicas de aprendizaje automático ahora está generalizado en el campo de la detección de fraude, donde se requiere la extracción de información de grandes conjuntos de datos {cite}`lucas2020credit,priscilla2019credit,carcillo2018beyond,dal2015adaptive`.

El aprendizaje automático (ML) es el estudio de algoritmos que mejoran automáticamente a través de la experiencia {cite}`bontempi2021statistical,friedman2001elements`. ML está estrechamente relacionado con los campos de Estadística, Reconocimiento de Patrones y Minería de Datos. Al mismo tiempo, surge como un subcampo de la informática y la inteligencia artificial y presta especial atención a la parte algorítmica del proceso de extracción de conocimiento. ML juega un papel clave en muchas disciplinas científicas y sus aplicaciones son parte de nuestra vida diaria. Se utiliza, por ejemplo, para filtrar correo no deseado, para la predicción meteorológica, en el diagnóstico médico, recomendación de productos, detección de rostros, detección de fraude, etc. {cite}`dal2015adaptive,bishop2006pattern`.

La capacidad de las técnicas de ML para abordar eficazmente los desafíos planteados por CCFD ha llevado a un gran y creciente cuerpo de investigación en la última década. Como se informa en la Fig. 1, miles de artículos relacionados con este tema se han publicado entre 2010 y 2020, con alrededor de 1500 artículos publicados solo en 2020. 

![alt text](images/ML_CCFD_GoogleScholar_2010_2020.png)
<p style="text-align: center;">
Fig. 1. Número de artículos publicados sobre el tema del aprendizaje automático y la detección de fraude con tarjetas de crédito entre 2010 y 2020. Fuente: Google Scholar.
</p>

Esta sección tiene como objetivo proporcionar una visión general de este cuerpo de investigación reciente, resumiendo los principales desafíos de investigación y los conceptos clave de aprendizaje automático que se pueden utilizar para abordarlos.   


## Encuestas recientes

Para obtener una imagen del estado actual de la investigación sobre ML para CCFD, buscamos en Google Scholar todas las revisiones y encuestas realizadas sobre este tema en los últimos cinco años. Usando la siguiente búsqueda booleana: `("machine learning" OR "data mining") AND "credit card" AND "fraud detection" AND (review OR survey)` y restringiendo el período de búsqueda de 2015 a 2021, identificamos diez revisiones/encuestas que informamos en la siguiente tabla. 


| Título | Fecha| Referencia |
|---|---|---|
|Una encuesta sobre técnicas de detección de fraude con tarjetas de crédito: <br> Perspectiva orientada a datos y técnicas | 2016 | {cite}`zojaji2016survey`|
|Una encuesta de técnicas de detección de fraude con tarjetas de crédito basadas en <br> aprendizaje automático y bio-inspiradas | 2017 | {cite}`adewumi2017survey`|
|Una encuesta sobre detección de fraude con tarjetas de crédito <br> utilizando aprendizaje automático | 2018 | {cite}`popat2018survey`|
|Una revisión del estado del arte de las técnicas de aprendizaje automático <br> para la investigación de detección de fraude | 2018 | {cite}`sinayobye2018state`|
|Detección de fraude con tarjetas de crédito: Estado del arte | 2018 | {cite}`sadgali2018detection`|
|Una encuesta sobre diferentes métodos de minería de datos y aprendizaje automático <br> para la detección de fraude con tarjetas de crédito | 2018 | {cite}`patil2018survey`|
|Una revisión sistemática de los enfoques de minería de datos <br> para la detección de fraude con tarjetas de crédito | 2018 | {cite}`mekterovic2018systematic`|
|Una encuesta completa sobre técnicas de aprendizaje automático <br> y enfoques de autenticación de usuarios <br>para la detección de fraude con tarjetas de crédito | 2019 | {cite}`yousefi2019comprehensive`|
|Detección de fraude con tarjetas de crédito: Una revisión sistemática | 2019 | {cite}`priscilla2019credit`|
|Detección de fraude con tarjetas de crédito utilizando aprendizaje automático: <br> Una encuesta | 2020 | {cite}`lucas2020credit`|

Un conjunto de diez encuestas en cinco años puede considerarse alto. El hecho de que se publicaran tantas encuestas en un período tan corto (en particular para las cinco encuestas publicadas en 2018) refleja la rápida evolución del tema de ML para CCFD y la necesidad que sintieron los equipos de investigadores independientes de sintetizar el estado de la investigación en este campo. 

Dado el objetivo común de estas encuestas, vale la pena señalar que se puede encontrar un alto grado de redundancia en términos de contenido. En particular, todas enfatizan un conjunto común de metodologías y desafíos, que presentamos en las siguientes dos secciones. Primero cubrimos la metodología de referencia, es decir, el flujo de trabajo común que se sigue típicamente en los artículos que tratan sobre el uso de técnicas de ML para abordar CCFD. Luego resumimos los desafíos que caracterizan este tema. 

(ML_For_CCFD_Baseline_Methodology)=
## Metodología de referencia - Aprendizaje supervisado

Una gran cantidad de técnicas de ML se pueden utilizar para abordar el problema de CCFD. Esto se refleja directamente en la gran cantidad de artículos publicados sobre el tema en la última década. A pesar de este gran volumen de trabajos de investigación, la mayoría de los enfoques propuestos siguen una metodología de ML de referencia común {cite}`patil2018survey,friedman2001elements,bishop2006pattern`, que resumimos en la Fig. 2.

![alt text](images/baseline_ML_workflow.png)
<p style="text-align: center;">
Fig. 2. ML para CCFD: Metodología de referencia seguida por la mayoría de los enfoques propuestos en las encuestas recientes sobre el tema.
</p>

En la detección de fraude con tarjetas de crédito, los datos generalmente consisten en datos de transacciones, recopilados, por ejemplo, por un procesador de pagos o un banco. Los datos de transacciones se pueden dividir en tres grupos {cite}`lucas2020credit,adewumi2017survey,VANVLASSELAER201538`

* Características relacionadas con la cuenta: Incluyen, por ejemplo, el número de cuenta, la fecha de apertura de la cuenta, el límite de la tarjeta, la fecha de vencimiento de la tarjeta, etc.
* Características relacionadas con la transacción: Incluyen, por ejemplo, el número de referencia de la transacción, el número de cuenta, el monto de la transacción, el número de terminal (es decir, POS), la hora de la transacción, etc. Desde el terminal, también se puede obtener una categoría adicional de información: características relacionadas con el comerciante, como su código de categoría (restaurante, supermercado, ...) o su ubicación.
* Características relacionadas con el cliente: Incluyen, por ejemplo, el número de cliente, el tipo de cliente (perfil bajo, perfil alto, ...), etc.

En su forma más simple, una transacción con tarjeta de pago consiste en cualquier cantidad pagada a un comerciante por un cliente en un momento determinado. Un conjunto de datos históricos de transacciones puede representarse como una tabla tal como se ilustra en la Fig. 3. Para la detección de fraude, generalmente se asume que la legitimidad de todas las transacciones es conocida (es decir, si la transacción fue genuina o fraudulenta). Esto se representa generalmente mediante una etiqueta binaria, con un valor de 0 para una transacción genuina y un valor de 1 para transacciones fraudulentas. 

![alt text](images/tx_table.png)
<p style="text-align: center;">
Fig. 3. Ejemplo de datos de transacciones representados como una tabla. Cada fila corresponde a una transacción de un cliente a un terminal. La última variable es la etiqueta, que indica si la transacción fue genuina (0) o fraudulenta (1).
</p>

Se pueden distinguir dos etapas en el diseño de un sistema de detección de fraude basado en ML. La primera etapa consiste en construir un modelo de predicción a partir de un conjunto de datos históricos etiquetados (Fig. 2, parte superior). Este proceso se llama *aprendizaje supervisado* ya que se conoce la etiqueta de las transacciones (genuina o fraudulenta). En la segunda etapa, el modelo de predicción obtenido del proceso de aprendizaje supervisado se utiliza para predecir la etiqueta de nuevas transacciones (Fig. 2, parte inferior). 

Formalmente, un modelo de predicción es una función paramétrica con parámetros $\theta$, también llamada *hipótesis*, que toma una entrada $x$ de un dominio de entrada $\mathcal{X}\subset \mathbb{R}^n$ y emite una predicción $\hat{y}=h(x,\theta)$ sobre un dominio de salida $\mathcal{Y} \subset \mathbb{R}$ {cite}`carcillo2018beyond,dal2015adaptive`:

$$
h(x,\theta): \mathcal{X} \rightarrow \mathcal{Y}
$$
  
El dominio de entrada $\mathcal{X}$ generalmente difiere del espacio de datos de transacciones sin procesar por dos razones. Primero, por razones matemáticas, la mayoría de los algoritmos de aprendizaje supervisado requieren que el dominio de entrada sea de valor real, es decir, $\mathcal{X} \subset \mathbb{R}^n$, lo que requiere transformar las características de la transacción que no son números reales (como marcas de tiempo, variables categóricas, etc...). En segundo lugar, generalmente es beneficioso enriquecer los datos de las transacciones con otras variables que pueden mejorar el rendimiento de detección del modelo de predicción. Este proceso se denomina *ingeniería de características* (también conocido como *transformación de características*, *extracción de características* o *preprocesamiento de datos*).

Para la detección de fraude, el dominio de salida $\mathcal{Y}$ suele ser la clase predicha para una entrada dada $x$, es decir, $\mathcal{Y}=\{0,1\}$. Dado que la clase de salida es binaria, estos modelos de predicción también se denominan *clasificadores binarios*. Alternativamente, la salida también puede expresarse como una probabilidad de fraude, con $\mathcal{Y}=[0,1]$, o más generalmente como una puntuación de riesgo, con $\mathcal{Y} = \mathbb{R}$, donde los valores más altos expresan mayores riesgos de fraude. 

El entrenamiento (o construcción) de un modelo de predicción $h(x,\theta)$ consiste en encontrar los parámetros $\theta$ que proporcionan el mejor rendimiento. El rendimiento de un modelo de predicción se evalúa utilizando una función de pérdida, que compara la etiqueta verdadera $y$ con la etiqueta predicha $\hat{y}=h(x,\theta)$ para una entrada $x$. En problemas de clasificación binaria, una función de pérdida común es la función de pérdida cero/uno $L_{0/1}$, que asigna una pérdida igual a uno en caso de predicción incorrecta y cero en caso contrario:

$$
\begin{align}
L_{0/1}: \mathcal{Y} \times \mathcal{Y} &\rightarrow& \{0,1\} \\
y,\hat{y} &= & 
\begin{cases}
    1,& \text{if } y \ne \hat{y}\\
    0,& \text{if } y=\hat{y}
\end{cases}
\end{align}
$$

```
La función de pérdida cero/uno es una función de pérdida estándar para problemas de clasificación binaria. Sin embargo, no es adecuada para problemas de detección de fraude con tarjetas de crédito, debido al desequilibrio de clases altas (muchas más transacciones genuinas que fraudulentas). Estimar el rendimiento de un sistema de detección de fraude es un problema no trivial, que se cubrirá en profundidad en el [Capítulo 4](Performance_Metrics).
```

Para obtener una estimación justa del rendimiento de un modelo de predicción, una práctica metodológica importante, conocida como *validación*, es evaluar el rendimiento de un modelo de predicción en datos que no se utilizaron para el entrenamiento. Esto se logra dividiendo el conjunto de datos, antes del entrenamiento, en un *conjunto de entrenamiento* y un *conjunto de validación*. El conjunto de entrenamiento se utiliza para el entrenamiento del modelo de predicción (es decir, para encontrar los parámetros $\theta$ que minimizan la pérdida en el conjunto de entrenamiento). Una vez que se han fijado los parámetros $\theta$, la pérdida se estima con el conjunto de validación, lo que proporciona una mejor estimación del rendimiento que se espera que tenga el modelo de predicción en transacciones futuras (y no vistas). 


```
Se debe tener especial cuidado en la práctica al dividir el conjunto de datos en conjuntos de entrenamiento y validación, debido a la naturaleza secuencial de las transacciones con tarjetas de crédito y el retraso en la notificación de fraudes. Estos problemas se abordarán en detalle en el [Capítulo 5](Model_Validation_And_Selection).    
```

El procedimiento de aprendizaje supervisado generalmente consiste en entrenar un conjunto de modelos de predicción y estimar su rendimiento utilizando el conjunto de validación. Al final del procedimiento, se selecciona el modelo que se asume que proporciona el mejor rendimiento (es decir, la pérdida más baja en el conjunto de validación) y se utiliza para proporcionar predicciones sobre nuevas transacciones (Ver Fig. 2).

Existe una amplia gama de métodos para diseñar y entrenar modelos de predicción. Esto explica en parte la gran literatura de investigación sobre ML para CCFD, donde los artículos generalmente se centran en uno o un par de métodos de predicción. La encuesta de Priscilla et al. en 2019 {cite}`priscilla2019credit` proporciona una buena descripción general de los métodos de aprendizaje automático que se han considerado para el problema de CCFD. Su encuesta cubrió cerca de cien trabajos de investigación, identificando para cada trabajo qué técnicas de ML se utilizaron, ver Fig. 4.  

![alt text](images/ReviewMLforCCFD_2019_Table.png)
<p style="text-align: center;">
Fig. 4. Frecuencia de uso de técnicas de ML en CCFD. Fuente: Priscilla et al., 2019 {cite}`priscilla2019credit`. Las referencias dadas en la tabla están en {cite}`priscilla2019credit`. 
</p>

La clasificación de las técnicas de aprendizaje en categorías de "alto nivel" no es un ejercicio sencillo, ya que a menudo existen conexiones metodológicas, algorítmicas o históricas entre ellas. Priscilla et al. optaron por dividir los enfoques en cuatro grupos: aprendizaje supervisado, aprendizaje no supervisado, aprendizaje conjunto y aprendizaje profundo. Se podría argumentar que el aprendizaje conjunto y el aprendizaje profundo son parte del aprendizaje supervisado, ya que requieren que se conozcan las etiquetas. Además, el aprendizaje profundo y las redes neuronales pueden considerarse parte de la misma categoría.  

Cubrir todas las técnicas de ML está fuera del alcance de este libro. Más bien, nuestro objetivo es proporcionar un marco de referencia y reproducible para CCFD. Decidimos, basándonos en nuestros trabajos de investigación, cubrir cinco tipos de métodos: regresión logística (LR), árboles de decisión (DT), bosques aleatorios (RF), Boosting y redes neuronales/aprendizaje profundo (NN/DL). LR y DT se eligieron debido a su simplicidad e interpretabilidad. RF y Boosting se eligieron ya que actualmente se consideran el estado del arte en términos de rendimiento. Los métodos NN/DL se eligieron ya que proporcionan direcciones de investigación prometedoras.  

(ML_For_CCFD_Challenges)=
## Descripción general de los desafíos

ML para CCFD es un problema notoriamente difícil. A continuación resumimos los desafíos comúnmente destacados en las revisiones sobre el tema {cite}`lucas2020credit,priscilla2019credit,mekterovic2018systematic,adewumi2017survey,zojaji2016survey`.

**Desequilibrio de clases**: Los datos de transacciones contienen muchas más transacciones legítimas que fraudulentas: el porcentaje de transacciones fraudulentas en un conjunto de datos del mundo real suele estar muy por debajo del 1%. Aprender de datos desequilibrados es una tarea difícil, ya que la mayoría de los algoritmos de aprendizaje no manejan bien grandes diferencias entre clases. Tratar con el desequilibrio de clases requiere el uso de estrategias de aprendizaje adicionales como muestreo o ponderación de pérdidas, un tema conocido como *aprendizaje desequilibrado*. 

**Deriva conceptual (Concept drift)**: Los patrones de transacciones y fraudes cambian con el tiempo. Por un lado, los hábitos de gasto de los usuarios de tarjetas de crédito son diferentes durante los días laborables, fines de semana, períodos de vacaciones y, en general, evolucionan con el tiempo. Por otro lado, los estafadores adoptan nuevas técnicas a medida que las antiguas se vuelven obsoletas. Estos cambios dependientes del tiempo en las distribuciones de transacciones y fraudes se denominan *deriva conceptual*. La deriva conceptual requiere el diseño de estrategias de aprendizaje que puedan hacer frente a los cambios temporales en las distribuciones estadísticas, un tema conocido como *aprendizaje en línea*. El problema de la deriva conceptual se acentúa en la práctica por los comentarios retrasados (Ver sección {ref}`Fraud_Detection_System`).

**Requisitos casi en tiempo real**: Los sistemas de detección de fraude deben ser capaces de detectar rápidamente transacciones fraudulentas. Dado el volumen potencialmente alto de datos de transacciones (millones de transacciones por día), pueden requerirse tiempos de clasificación tan bajos como decenas de milisegundos. Este desafío se relaciona estrechamente con la *paralelización* y *escalabilidad* de los sistemas de detección de fraude.

**Características categóricas**: Los datos transaccionales generalmente contienen numerosas características *categóricas*, como el ID de un cliente, un terminal, el tipo de tarjeta, etc. Las características categóricas no son bien manejadas por los algoritmos de aprendizaje automático y deben transformarse en características numéricas. Las estrategias comunes para transformar características categóricas incluyen agregación de características, transformación basada en gráficos o enfoques de aprendizaje profundo como incrustaciones de características (embeddings).

**Modelado secuencial**: Cada terminal y/o cliente genera un flujo de datos secuenciales con características únicas. Un desafío importante de la detección de fraude consiste en modelar estos flujos para caracterizar mejor sus comportamientos esperados y detectar cuándo ocurren comportamientos anormales. El modelado se puede realizar agregando características a lo largo del tiempo (por ejemplo, haciendo un seguimiento de la frecuencia media o los montos de transacción de un cliente) o confiando en modelos de predicción secuenciales (como modelos ocultos de Markov o redes neuronales recurrentes, por ejemplo). 

**Superposición de clases**: Los dos últimos desafíos pueden asociarse con el desafío más general de superposición entre las dos clases. Con solo información sin procesar sobre una transacción, distinguir entre una transacción fraudulenta o genuina es casi imposible. Este problema se aborda comúnmente utilizando técnicas de ingeniería de características, que agregan información contextual a la información de pago sin procesar.  

**Medidas de rendimiento**: Las medidas estándar para los sistemas de clasificación, como el error medio de clasificación o el AUC ROC, no son adecuadas para problemas de detección debido al problema de desequilibrio de clases y la compleja estructura de costos de la detección de fraude. Un sistema de detección de fraude debe ser capaz de maximizar la detección de transacciones fraudulentas mientras minimiza el número de fraudes predichos incorrectamente (falsos positivos). A menudo es necesario considerar múltiples medidas para evaluar el rendimiento general de un sistema de detección de fraude. A pesar de su papel central en el diseño de un sistema de detección de fraude, actualmente no hay consenso sobre qué conjunto de medidas de rendimiento debe utilizarse. 

**Falta de conjuntos de datos públicos**: Por razones obvias de confidencialidad, las transacciones de tarjetas de crédito del mundo real no se pueden compartir públicamente. Solo existe un conjunto de datos compartido públicamente, que nuestro equipo puso a disposición en Kaggle {cite}`Kaggle2016` en 2016. A pesar de sus limitaciones (solo dos días de datos y características ofuscadas), el conjunto de datos se ha utilizado ampliamente en la literatura de investigación y es uno de los más votados y descargados en Kaggle. La escasez de conjuntos de datos para la detección de fraude también es cierta con datos simulados: aún no hay simuladores o conjuntos de datos simulados de referencia disponibles. Como resultado, la mayoría de los trabajos de investigación no se pueden reproducir, lo que hace imposible la comparación de diferentes técnicas por parte de investigadores independientes.    




