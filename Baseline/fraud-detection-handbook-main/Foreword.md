# Prólogo


El fraude con tarjetas de pago es un desafío importante para los propietarios de negocios, emisores de tarjetas de pago y empresas de servicios transaccionales, causando cada año pérdidas financieras sustanciales y crecientes. Según el Informe Nilson de 2019, las pérdidas por fraude con tarjetas en todo el mundo han aumentado de 9.84 mil millones de dólares en 2011 a 27.85 mil millones de dólares en 2018, y se prevé que superen los 40 mil millones de dólares en 2027 {cite}`NilsonReport2019`.

Detectar patrones de fraude en transacciones con tarjetas de pago es conocido por ser un problema muy difícil. Con la cantidad cada vez mayor de datos generados por las transacciones con tarjetas de pago, se ha vuelto imposible para un analista humano detectar patrones fraudulentos en conjuntos de datos de transacciones, a menudo caracterizados por una gran cantidad de muestras, muchas dimensiones y actualizaciones en línea. Como resultado, el diseño de técnicas de detección de fraude con tarjetas de pago se ha centrado cada vez más en la última década en enfoques basados en técnicas de aprendizaje automático (ML), que automatizan el proceso de identificación de patrones fraudulentos a partir de grandes volúmenes de datos {cite}`priscilla2019credit,carcillo2019combining,sadgali2018detection,dal2015adaptive`.

La integración de técnicas de ML en los sistemas de detección de fraude con tarjetas de pago ha mejorado enormemente su capacidad para detectar fraudes de manera más eficiente y ayudar a los intermediarios de procesamiento de pagos a identificar transacciones ilícitas. Aunque en los últimos años el número de transacciones fraudulentas siguió aumentando, el porcentaje de pérdidas debidas al fraude comenzó a disminuir en 2016, una tendencia inversa que se asocia con la creciente adopción de soluciones de ML {cite}`NilsonReport2019`. Además de ayudar a ahorrar dinero, la implementación de sistemas de detección de fraude basados en ML se está convirtiendo hoy en día en una obligación para las instituciones y empresas para ganarse la confianza de sus clientes.

Un problema ampliamente reconocido y recurrente en este nuevo campo del ML para la detección de fraude con tarjetas es la falta de reproducibilidad de la mayoría de los trabajos de investigación publicados sobre el tema {cite}`lucas2020credit,priscilla2019credit,patil2018survey,zojaji2016survey`. Por un lado, hay una falta de disponibilidad de datos de transacciones con tarjetas de pago, que no pueden compartirse públicamente por razones de confidencialidad. Por otro lado, los autores no hacen suficientes esfuerzos para proporcionar su código y hacer que sus resultados sean reproducibles.

Este libro tiene como objetivo dar un primer paso en la dirección de la reproducibilidad en la evaluación comparativa (benchmarking) de técnicas de detección de fraude con tarjetas de pago. Debido a la gran cantidad de investigaciones publicadas en el dominio, no fue posible revisar e implementar exhaustivamente todas las técnicas existentes. Más bien, optamos por centrarnos en algunas de las técnicas que nos parecieron más esenciales, basándonos en nuestra colaboración de 10 años con nuestro socio industrial Worldline.

Algunas de las técnicas presentadas, como las que tratan con el desequilibrio de clases o los conjuntos de modelos, son ampliamente reconocidas como partes esenciales del diseño de un sistema de detección de fraude con tarjetas de crédito. Además, cubrimos temas menos documentados que creemos que merecen más atención. Estos incluyen, en particular, aspectos de diseño del proceso de modelado, como la elección de métricas de rendimiento y estrategias de validación, y estrategias prometedoras de preprocesamiento y aprendizaje, como incrustaciones de características (embeddings) y redes neuronales en general.

Si bien el libro se centra en el fraude con tarjetas de pago, creemos que la mayoría de las técnicas y discusiones presentadas en este libro pueden ser útiles para otros profesionales que trabajan en el tema más amplio de la detección de fraude.

Con la reproducibilidad de los experimentos como un impulsor clave para este libro, la elección de un formato de Jupyter Book pareció más adecuada que un formato de libro impreso tradicional. En particular, todas las secciones de este libro que incluyen código son notebooks de Jupyter, que pueden ejecutarse de forma independiente, ya sea en la computadora del lector clonando el repositorio del libro, o en línea utilizando Google Colab o Binder. Además, la naturaleza de código abierto del libro - totalmente disponible en un repositorio público de Github - permite a los lectores abrir discusiones sobre el contenido del libro gracias a los "issues" de Github, o proponer enmiendas o mejoras con "pull requests".

## Licencia

El código en los notebooks se publica bajo una [licencia GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). La prosa y las imágenes se publican bajo una [licencia CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


Si desea citar este libro, puede usar lo siguiente:

<pre>
@book{leborgne2022fraud,
title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
author={Le Borgne, Yann-A{\"e}l and Siblini, Wissam and Lebichot, Bertrand and Bontempi, Gianluca},
url={https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook},
year={2022},
publisher={Universit{\'e} Libre de Bruxelles}
}
</pre>

## Autores

* [Yann-Aël Le Borgne](https://yannael.github.io/) (Autor de contacto - yann-ael.le.borgne@ulb.be) - [Machine Learning Group - Université Libre de Bruxelles, Bélgica](http://mlg.ulb.ac.be).
* [Wissam Siblini](https://www.linkedin.com/in/wissam-siblini) - [Machine Learning Research - Worldline Labs](https://worldline.com)
* [Bertrand Lebichot](https://b-lebichot.github.io/) - [Interdisciplinary Centre for Security, Reliability and Trust  - Université du Luxembourg, Luxemburgo](https://wwwfr.uni.lu/snt)
* [Gianluca Bontempi](https://mlg.ulb.ac.be/wordpress/members-2/gianluca-bontempi/) - [Machine Learning Group - Université Libre de Bruxelles, Bélgica](http://mlg.ulb.ac.be)

## Agradecimientos

Este libro es el resultado de diez años de colaboración entre el [Machine Learning Group,  Université Libre de Bruxelles, Bélgica](http://mlg.ulb.ac.be) y [Worldline](https://worldline.com).

* ULB-MLG, Investigador principal: Gianluca Bontempi
* Worldline, Gerente de I+D: Frédéric Oblé

Deseamos agradecer a todos los colegas que trabajaron en este tema durante esta colaboración: Olivier Caelen (ULB-MLG/Worldline), Fabrizio Carcillo (ULB-MLG), Guillaume Coter (Worldline), Andrea Dal Pozzolo (ULB-MLG), Jacopo De Stefani (ULB-MLG), Rémy Fabry (Worldline), Liyun He-Guelton (Worldline), Gian Marco Paldino (ULB-MLG), Théo Verhelst (ULB-MLG).

La colaboración fue posible gracias a [Innoviris](https://innoviris.brussels), el Instituto de Investigación e Innovación de la Región de Bruselas, a través de una serie de subvenciones que comenzaron en 2012 y terminaron en 2021.

* 2018 a 2021. *DefeatFraud: Evaluación y validación de ingeniería de características profundas y soluciones de aprendizaje para la detección de fraudes*. Programa Team Up de Innoviris.
* 2015 a 2018. *BruFence: Aprendizaje automático escalable para automatizar el sistema de defensa*. Programa Bridge de Innoviris.
* 2012 a 2015. *Aprendizaje automático adaptable en tiempo real para la detección de fraude con tarjetas de crédito*. Programa Doctiris de Innoviris.

La colaboración continúa en el contexto del [proyecto Ingeniería de Datos para Ciencia de Datos (DEDS)](https://deds.ulb.ac.be/) - bajo el marco Horizon 2020 - Marie Skłodowska-Curie Innovative Training Networks (H2020-MSCA-ITN-2020).