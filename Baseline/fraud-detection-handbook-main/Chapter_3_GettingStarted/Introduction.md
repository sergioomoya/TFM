# Introducción


La mejor manera de hacerse una idea de los desafíos subyacentes al diseño de un sistema de detección de fraude con tarjetas de crédito (FDS) es diseñando uno. Este capítulo presenta una implementación de un FDS de referencia y cubre los pasos principales que deben considerarse. 

[Sección 3.2](Transaction_data_Simulator) describe primero un simulador simple para datos de transacciones de tarjetas de pago. Aunque simplista, el simulador proporciona un entorno lo suficientemente desafiante para abordar un problema típico de detección de fraude. En particular, el simulador permitirá generar conjuntos de datos que i) están muy desequilibrados (baja proporción de transacciones fraudulentas), ii) contienen variables numéricas y categóricas (con características categóricas que tienen un número muy alto de valores posibles), y iii) presentan escenarios de fraude dependientes del tiempo.

[Sección 3.3](Baseline_Feature_Transformation) y [sección 3.4](Baseline_FDS) abordan los dos pasos principales de un proceso de modelado predictivo estándar: transformación de características y modelado predictivo. Estas secciones proporcionarán algunas estrategias de referencia para realizar transformaciones de características significativas y construir un primer modelo predictivo cuya precisión servirá como referencia en el resto del libro. 

Finalmente, en la [sección 3.5](Baseline_FDS_RealWorldData), aplicamos esta metodología de referencia (transformación de características y modelado predictivo) a un conjunto de datos del mundo real de transacciones con tarjetas e ilustramos su capacidad para detectar eficazmente transacciones fraudulentas.  