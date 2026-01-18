(Credit_Card_Fraud_Scenarios)=
# Escenarios de fraude con tarjetas de crédito

Las pérdidas financieras mundiales causadas por actividades fraudulentas con tarjetas de crédito ascienden a decenas de miles de millones de dólares. Uno de cada diez estadounidenses ha sido víctima de fraude con tarjetas de crédito (monto medio de $399), según el Statistic Brain Research Institute {cite}`StatisticBrain2018`. Según el último informe del Banco Central Europeo (ECB) {cite}`ECB2020`, el nivel total de pérdidas por fraude con tarjetas ascendió a €1.8 mil millones en 2018 en la Zona Única de Pagos en Euros (SEPA).

Existe una amplia variedad de escenarios que pueden conducir a un estafador a realizar con éxito pagos fraudulentos con tarjeta de crédito. Actualmente no existe una taxonomía definida sobre los tipos de fraude con tarjetas de crédito, aunque se sabe que ciertos patrones ocurren con más frecuencia que otros. También debe tenerse en cuenta que la detección de fraude es un juego del gato y el ratón, donde los patrones fraudulentos cambian con el tiempo. A medida que evoluciona la tecnología, tanto en términos de prevención del fraude como de facilidad de uso de los sistemas de pago, también lo hacen las técnicas de los estafadores. Se adaptan pasando de los objetivos antiguos (y ahora corregidos) a la vulnerabilidad de las nuevas tecnologías. También se benefician de los cambios constantes en el volumen y las características de las transacciones genuinas.

## Fraudes con tarjeta presente vs Fraudes con tarjeta no presente

Es útil distinguir dos escenarios de transacción. El primero, llamado escenarios de *tarjeta presente* (CP), se refiere a escenarios donde se necesita una tarjeta física, como transacciones en una tienda (también denominado punto de venta - POS) o transacciones en un cajero automático (por ejemplo, en un cajero automático - ATM). El segundo, llamado escenarios de *tarjeta no presente* (CNP), se refiere a escenarios donde no es necesario utilizar una tarjeta física, lo que abarca los pagos realizados en Internet, por teléfono o por correo. 

Esta distinción es importante ya que las técnicas utilizadas para comprometer una tarjeta varían, dependiendo de si es necesario producir una copia física de la tarjeta o no. Más importante aún, los estafadores tienen recientemente más probabilidades de explotar las deficiencias de los escenarios CNP que los de CP, probablemente porque los escenarios CP han existido durante más de dos décadas y se han vuelto bastante robustos a los ataques de fraude, notablemente gracias a la tecnología EMV (Europay Mastercard y Visa, es decir, tarjetas con chip integrado). Otra razón es que las consideraciones simples sobre las barreras físicas a menudo pueden ayudar a prevenir fraudes de CP. Como se indica en el informe Nilson de 2019, los escenarios CNP representaron el 54% de todas las pérdidas por fraude para el año 2018, mientras que solo representaron menos del 15% de todo el volumen de compras a nivel mundial (CNP+POS+ATM) {cite}`NilsonReport2019`. La proporción de fraude CNP es aún mayor en Europa y se informó que representó el 79% de todas las transacciones de tarjetas emitidas dentro de SEPA en el informe de 2020 sobre fraude con tarjetas del Banco Central Europeo {cite}`ECB2020`, como se informa en la figura a continuación. 

![alt text](./images/SEPA_FraudVolumePerType.png)
<p style="text-align: center;">
Fig. 1. Evolución del valor total del fraude con tarjetas utilizando tarjetas emitidas dentro de SEPA. <br>Los fraudes con tarjeta no presente representan la mayoría de los fraudes reportados.
</p>

### Fraudes con tarjeta presente

Los fraudes con tarjeta presente ocurren cuando un estafador logra realizar una transacción fraudulenta exitosa utilizando una tarjeta de pago física, ya sea en un cajero automático o en un POS. En este entorno, los escenarios de fraude generalmente se clasifican como *tarjetas perdidas o robadas*, *tarjetas falsificadas* y *tarjeta no recibida*.

**Tarjeta perdida o robada**: La tarjeta pertenece a un cliente legítimo y llega a manos de un estafador después de una pérdida o un robo. Este es el tipo de fraude más común en el entorno de fraude con tarjeta presente y permite a un estafador realizar transacciones siempre que la tarjeta no sea bloqueada por su propietario legítimo. En este escenario, el estafador generalmente intenta gastar tanto como sea posible y tan rápido como sea posible.

**Tarjeta falsificada**: Un estafador produce una tarjeta falsa imprimiendo la información de una tarjeta. Dicha información generalmente se obtiene mediante el *skimming* (clonación) de la tarjeta del cliente legítimo, sin que este se dé cuenta. Dado que los propietarios legítimos no son conscientes de la existencia de una copia de su tarjeta, la fuente del fraude podría ser más difícil de identificar, ya que el estafador puede esperar mucho tiempo antes de hacer uso de la tarjeta falsa. El mayor uso de la tecnología de chip y PIN (también conocida como EMV) ha reducido este tipo de fraude. 

**Tarjeta no recibida**: La tarjeta fue interceptada por un estafador en el buzón de un cliente legítimo. Esto puede suceder si un cliente solicita una nueva tarjeta, que es interceptada, o si un estafador logra solicitar una nueva tarjeta sin el conocimiento del cliente legítimo (por ejemplo, accediendo fraudulentamente a su cuenta bancaria) y hacer que se entregue a una dirección diferente. En el primer caso, los clientes pueden advertir rápidamente al banco que no recibieron su tarjeta y hacer que la bloqueen. El último caso puede ser más difícil de detectar ya que el cliente no sabe que se solicitó una nueva tarjeta. 

Las estadísticas sobre la proporción de estos tipos de fraude en escenarios de tarjeta presente fueron reportadas por el Banco Central Europeo para 2018, ver el gráfico a continuación {cite}`ECB2020`.

![alt text](./images/SEPA_FraudType_CardPresent.png)
<p style="text-align: center;">
Fig. 2. Evolución y desglose del valor del fraude con tarjeta presente por categoría dentro de SEPA.
</p>

Las principales categorías de fraudes son *perdidas y robadas* y *tarjetas falsificadas*, mientras que los escenarios de *tarjeta no recibida* representan una proporción muy pequeña de las pérdidas por fraude. Vale la pena señalar que estas proporciones de fraude son aproximadamente las mismas ya sea que los pagos se hayan realizado en un cajero automático o en un punto de venta, y que, en general, la cantidad de fraudes en entornos de tarjeta presente tiende a disminuir.

### Fraudes con tarjeta no presente

Tarjeta no presente se refiere a la categoría general de fraudes realizados de forma remota, ya sea por correo, teléfono o en Internet, utilizando solo parte de la información presente en una tarjeta. 

En general, hay menos estadísticas disponibles sobre la causa de tales fraudes. Por ejemplo, a diferencia de los fraudes con tarjeta presente, el Banco Central Europeo solo requiere que los operadores de esquemas de pago con tarjeta informen sobre las pérdidas generales por fraude CNP.

Sin embargo, se sabe que la mayoría de los fraudes CNP son una consecuencia directa de credenciales de pago obtenidas ilegalmente (por ejemplo, números de tarjeta), ya sea por violaciones de datos o, a veces, directamente de los titulares de la tarjeta (por ejemplo, a través de phishing, mensajes de texto fraudulentos). También vale la pena señalar que dichas credenciales generalmente no se utilizan directamente, sino que se ponen a la venta en mercados web clandestinos (la *dark web*) y luego son utilizadas por grupos delictivos. Los delincuentes que roban datos suelen ser un grupo diferente a los delincuentes que perpetran fraudes {cite}`ECB2020,NilsonReport2019`. 

Los datos que generalmente están involucrados en el fraude de tarjeta no presente involucran el número de tarjeta, la fecha de vencimiento de la tarjeta, el código de seguridad de la tarjeta y la información de facturación personal (como la dirección del titular de la tarjeta).

