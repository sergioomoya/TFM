(Summary_Performance_Metrics)=
# Resumen



Este capítulo ilustró que el diseño de un sistema de detección de fraude *de referencia* se puede lograr utilizando estrategias de preprocesamiento simples y clasificadores de aprendizaje automático estándar. En particular, logramos obtener rendimientos de detección de fraude que están muy por encima de los de un clasificador aleatorio. 

Sin embargo, el capítulo solo arañó la superficie de cómo abordar un problema de detección de fraude. Como veremos, se puede utilizar una gran cantidad de técnicas más avanzadas para mejorar el rendimiento. El rendimiento se puede abordar en términos de precisión de detección de fraude, pero también en términos de requisitos computacionales (memoria/tiempos de ejecución). Esto último es importante en la práctica durante el entrenamiento, ya que los sistemas de detección de fraude deben lidiar con grandes cantidades de datos (mucho más altos que los utilizados en este ejemplo de referencia) y también durante la inferencia para el procesamiento en tiempo real o casi en tiempo real. Generalmente se deben considerar cuidadosamente las compensaciones entre la precisión y los requisitos computacionales. 

Los capítulos avanzados cubrirán en detalle las posibles vías que se pueden explorar para mejorar el enfoque de referencia propuesto. 

Antes de eso, el enfoque de los siguientes dos capítulos abordará más específicamente la metodología experimental, es decir, qué medidas de rendimiento deben utilizarse y cómo se pueden estimar. Estos problemas son fundamentales para encontrar una forma objetiva de comparar los rendimientos de diferentes sistemas de detección de fraude e identificar el de mejor rendimiento. 









