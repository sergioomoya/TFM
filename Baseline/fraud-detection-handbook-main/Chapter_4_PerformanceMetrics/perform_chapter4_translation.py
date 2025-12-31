
import json
import os

def translate_notebook(file_path, replacements):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        new_source = []
        for line in cell['source']:
            new_line = line
            for en, es in replacements.items():
                if en in new_line:
                    new_line = new_line.replace(en, es)
                    changed = True
            new_source.append(new_line)
        cell['source'] = new_source

    if changed:
        print(f"Writing changes to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
    else:
        print(f"No changes made to {file_path}")

# Assessment_RealWorldData.ipynb translations
assessment_replacements = {
    "Fig. 2 reports the PR curves for the same five baseline classifiers on real-world data. The curves are at first sight quite different from those obtained on simulated data ({ref}`Precision_Recall_Curve`).": "La Fig. 2 reporta las curvas PR para los mismos cinco clasificadores base en datos del mundo real. Las curvas son a primera vista bastante diferentes de las obtenidas con datos simulados ({ref}`Precision_Recall_Curve`).",
    "We finally report, as in {ref}`Precision_Top_K_Metrics`, the number of daily detected frauded cards using real-world data. The results are provided in Fig. 3, for a decision tree of depth 2.": "Finalmente reportamos, como en {ref}`Precision_Top_K_Metrics`, el número de tarjetas fraudulentas detectadas diariamente utilizando datos del mundo real. Los resultados se proporcionan en la Fig. 3, para un árbol de decisión de profundidad 2."
}

# TopKBased.ipynb translations
topk_replacements = {
    "We get a precision of 0.26, that is, 26 out of the one hundred most highly suspicious transactions were indeed fraudulent transactions.": "Obtenemos una precisión de 0.26, es decir, 26 de las cien transacciones más sospechosas fueron de hecho transacciones fraudulentas.",
    "It should be noted that the highest value that is achievable for the $P@k(d)$ may be lower than one. This is the case when $k$ is higher than the number of fraudulent transactions for a given day. In the example above, the number of fraudulent transactions at day 129 is 55.": "Cabe señalar que el valor más alto alcanzable para el $P@k(d)$ puede ser menor que uno. Este es el caso cuando $k$ es mayor que el número de transacciones fraudulentas para un día dado. En el ejemplo anterior, el número de transacciones fraudulentas en el día 129 es 55.",
    "As a result, the maximum $P@100(129)$ which can be achieved is 55/100=0.55.": "Como resultado, el máximo $P@100(129)$ que se puede lograr es 55/100=0.55.",
    "When a test set spans multiple days, let $P@k$ be the mean of $P@k(d)$ for a set of days $d \\in \\mathcal{D}$ {cite}`dal2017credit,dal2015adaptive`, that is:": "Cuando un conjunto de prueba abarca varios días, sea $P@k$ la media de $P@k(d)$ para un conjunto de días $d \\in \\mathcal{D}$ {cite}`dal2017credit,dal2015adaptive`, es decir:",
    "Let us implement this score with a function `precision_top_k`, which takes as input a `predictions_df` DataFrame, and a `top_k` threshold. The function loops over all days of the DataFrame, and computes for each day the number of fraudulent transactions and the precision top-$k$. It returns these values for all days as lists, together with the resulting mean of the precisions top-$k$.": "Implementemos esta puntuación con una función `precision_top_k`, que toma como entrada un DataFrame `predictions_df` y un umbral `top_k`. La función itera sobre todos los días del DataFrame y calcula para cada día el número de transacciones fraudulentas y la precisión top-$k$. Devuelve estos valores para todos los días como listas, junto con la media resultante de las precisiones top-$k$.",
    "# Initialization: Load shared functions and simulated data": "# Inicialización: Cargar funciones compartidas y datos simulados",
    "# Load shared functions": "# Cargar funciones compartidas",
    "# Get simulated data from Github repository": "# Obtener datos simulados del repositorio de Github",
    "# Load data from the 2018-07-25 to the 2018-08-14": "# Cargar datos del 2018-07-25 al 2018-08-14",
    'print("Load  files")': 'print("Cargar archivos")',
    'print("{0} transactions loaded, containing {1} fraudulent transactions".format': 'print("{0} transacciones cargadas, conteniendo {1} transacciones fraudulentas".format',
    "# Order transactions by decreasing probabilities of frauds": "# Ordenar transacciones por probabilidades decrecientes de fraude",
    "# Get the top k most suspicious transactions": "# Obtener las top k transacciones más sospechosas",
    "# Compute precision top k": "# Calcular precisión top k",
    "# Sort days by increasing order": "# Ordenar días en orden creciente",
    "# For each day, compute precision top k": "# Para cada día, calcular precisión top k",
    "# Compute the mean": "# Calcular la media",
    "# Returns number of fraudulent transactions per day,": "# Devuelve el número de transacciones fraudulentas por día,",
    "# precision top k per day, and resulting mean": "# precisión top k por día, y la media resultante",
    'print("Number of remaining fraudulent transactions: "': 'print("Número de transacciones fraudulentas restantes: "',
    'print("Precision top-k: "': 'print("Precisión top-k: "',
    'print("Average Precision top-k: "': 'print("Precisión promedio top-k: "',
    "# Compute the number of transactions per day, ": "# Calcular el número de transacciones por día, ",
    "#fraudulent transactions per day and fraudulent cards per day": "#transacciones fraudulentas por día y tarjetas fraudulentas por día",
    "# Add the remaining number of fraudulent transactions for the last 7 days (test period)": "# Agregar el número restante de transacciones fraudulentas para los últimos 7 días (período de prueba)",
    "# Add precision top k for the last 7 days (test period) ": "# Agregar precisión top k para los últimos 7 días (período de prueba) ",
    "# Plot the number of transactions per day, fraudulent transactions per day and fraudulent cards per day": "# Graficar el número de transacciones por día, transacciones fraudulentas por día y tarjetas fraudulentas por día",
    "# Training period": "# Período de entrenamiento",
    "# Test period": "# Período de prueba",
    "title='Number of fraudulent transactions per day \\n and number of detected fraudulent transactions'": "title='Número de transacciones fraudulentas por día \\n y número de transacciones fraudulentas detectadas'",
    "label = '# fraudulent txs per day - Original'": "label = '# txs fraudulentas por día - Original'",
    "label = '# fraudulent txs per day - Remaining'": "label = '# txs fraudulentas por día - Restantes'",
    "label = '# detected fraudulent txs per day'": "label = '# txs fraudulentas detectadas por día'"
}

base_path = r"c:\Programacion\fraud-detection-handbook-main\Chapter_4_PerformanceMetrics"

translate_notebook(os.path.join(base_path, "Assessment_RealWorldData.ipynb"), assessment_replacements)
translate_notebook(os.path.join(base_path, "TopKBased.ipynb"), topk_replacements)
