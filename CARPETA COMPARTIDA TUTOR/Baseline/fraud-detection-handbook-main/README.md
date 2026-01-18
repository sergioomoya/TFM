# Aprendizaje Automático Reproducible para la Detección de Fraude con Tarjetas de Crédito - Manual Práctico

## Acceso anticipado

Versión preliminar disponible en [https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html).

## Motivaciones

El aprendizaje automático para la detección de fraude con tarjetas de crédito (ML para CCFD) se ha convertido en un campo de investigación activo. Esto se ilustra por la [notable cantidad de publicaciones sobre el tema en la última década](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/MachineLearningForFraudDetection.html).

No cabe duda de que la integración de técnicas de aprendizaje automático en los sistemas de detección de fraude en pagos con tarjeta ha mejorado enormemente su capacidad para detectar fraudes de manera más eficiente. Al mismo tiempo, un problema importante en este nuevo campo de investigación es la falta de reproducibilidad. No existen benchmarks ni metodologías reconocidas para comparar y evaluar las técnicas propuestas.

Este libro tiene como objetivo dar un primer paso en esta dirección. Todas las técnicas y resultados proporcionados en este libro son reproducibles. Las secciones que incluyen código son notebooks de Jupyter, que pueden ejecutarse localmente o en la nube utilizando [Google Colab](https://colab.research.google.com/) o [Binder](https://mybinder.org/).

El público objetivo son estudiantes o profesionales interesados en el problema específico de la detección de fraude con tarjetas de crédito desde un punto de vista práctico. De manera más general, creemos que el libro también es de interés para profesionales de datos y científicos de datos que se ocupan de problemas de aprendizaje automático que involucran datos secuenciales y/o problemas de clasificación desequilibrada.

Tabla de contenidos provisional:

* Capítulo 1: Descripción general del libro
* Capítulo 2: Antecedentes
* Capítulo 3: Primeros pasos
* Capítulo 4: Métricas de rendimiento
* Capítulo 5: Selección de modelos
* Capítulo 6: Aprendizaje desequilibrado
* Capítulo 7: Aprendizaje profundo
* Capítulo 8: Interpretabilidad*

(*): Aún no publicado.

## Borrador actual

La redacción del libro está en curso. Proporcionamos a través de este repositorio de Github un acceso temprano al libro. A partir de enero de 2022, los primeros siete capítulos están disponibles.

La versión en línea del borrador actual de este libro está disponible [aquí](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).

Cualquier comentario o sugerencia es bienvenido. Recomendamos usar los "issues" de Github para iniciar una discusión sobre un tema y usar "pull requests" para corregir errores tipográficos.


## Compilar el libro

Para leer y/o ejecutar este libro en su computadora, necesitará clonar este repositorio y compilar el libro.

Este libro es un libro de Jupyter. Por lo tanto, primero necesitará [instalar Jupyter Book](https://jupyterbook.org/intro.html#install-jupyter-book).

La compilación fue probada con las siguientes versiones de paquetes:

```
sphinxcontrib-bibtex==2.2.1
Sphinx==4.2.0
jupyter-book==0.11.2
```

Una vez hecho esto, es un proceso de dos pasos:

1. Clonar este repositorio:

```
git clone https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
```

2. Compilar el libro

```
jupyter-book build fraud-detection-handbook
```

El libro estará disponible localmente en `fraud-detection-handbook/_build/html/index.html`.

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

Este libro es el resultado de diez años de colaboración entre el [Machine Learning Group, Université Libre de Bruxelles, Bélgica](http://mlg.ulb.ac.be) y [Worldline](https://worldline.com).

* ULB-MLG, Investigador principal: Gianluca Bontempi
* Worldline, Gerente de I+D: Frédéric Oblé

Deseamos agradecer a todos los colegas que trabajaron en este tema durante esta colaboración: Olivier Caelen (ULB-MLG/Worldline), Fabrizio Carcillo (ULB-MLG), Guillaume Coter (Worldline), Andrea Dal Pozzolo (ULB-MLG), Jacopo De Stefani (ULB-MLG), Rémy Fabry (Worldline), Liyun He-Guelton (Worldline), Gian Marco Paldino (ULB-MLG), Théo Verhelst (ULB-MLG).

La colaboración fue posible gracias a [Innoviris](https://innoviris.brussels), el Instituto de Investigación e Innovación de la Región de Bruselas, a través de una serie de subvenciones que comenzaron en 2012 y terminaron en 2021.

* 2018 a 2021. *DefeatFraud: Evaluación y validación de ingeniería de características profundas y soluciones de aprendizaje para la detección de fraudes*. Programa Team Up de Innoviris.
* 2015 a 2018. *BruFence: Aprendizaje automático escalable para automatizar el sistema de defensa*. Programa Bridge de Innoviris.
* 2012 a 2015. *Aprendizaje automático adaptable en tiempo real para la detección de fraude con tarjetas de crédito*. Programa Doctiris de Innoviris.

La colaboración continúa en el contexto del [proyecto Ingeniería de Datos para Ciencia de Datos (DEDS)](https://deds.ulb.ac.be/) - bajo el marco Horizon 2020 - Marie Skłodowska-Curie Innovative Training Networks (H2020-MSCA-ITN-2020).