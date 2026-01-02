const fs = require('fs');
const path = require('path');

const rootDir = "c:\\Programacion\\fraud-detection-handbook-main";

const replacements = [
    // Exact matches
    { from: "# Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers", to: "# Las tarjetas comprometidas de ese día de prueba, menos el período de retraso, se agregan al grupo de clientes defraudados conocidos" },
    { from: "# Plot locations of terminals", to: "# Graficar ubicaciones de terminales" },
    { from: "# Plot location of the last customer", to: "# Graficar ubicación del último cliente" },
    { from: "# Plot the region within a radius of 50 of the last customer", to: "# Graficar la región dentro de un radio de 50 del último cliente" },
    { from: /# Load data from the (\d{4}-\d{2}-\d{2}) to the (\d{4}-\d{2}-\d{2})/g, to: "# Cargar datos del $1 al $2" },
    { from: "# This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,", to: "# Esto toma el máximo de las predicciones Y el máximo de la etiqueta TX_FRAUD para cada CUSTOMER_ID," },
    { from: "# Get the top k most suspicious cards", to: "# Obtener las top k tarjetas más sospechosas" },
    { from: "# Returns precision top k per day as a list, and resulting mean", to: "# Devuelve la precisión top k por día como una lista, y el promedio resultante" },
    { from: "# Note: n_jobs set to one for getting true execution times", to: "# Nota: n_jobs establecido en uno para obtener tiempos de ejecución reales" },
    { from: "# Get the performance plot for a single performance metric", to: "# Obtener el gráfico de rendimiento para una sola métrica de rendimiento" },
    { from: "# Plot data on graph", to: "# Graficar datos en el gráfico" },
    { from: /# Rename to ([\w_]+) for model performance comparison at the end of this notebook/g, to: "# Renombrar a $1 para comparación de rendimiento de modelos al final de este cuaderno" },
    { from: "# For each model class", to: "# Para cada clase de modelo" },
    { from: "# Get the performances for the default paramaters", to: "# Obtener los rendimientos para los parámetros predeterminados" },
    { from: "# Get the performances for the best estimated parameters", to: "# Obtener los rendimientos para los mejores parámetros estimados" },
    { from: "# Get the performances for the boptimal parameters", to: "# Obtener los rendimientos para los parámetros subóptimos" },
    { from: "# Return the mean performances and their standard deviations", to: "# Devolver los rendimientos medios y sus desviaciones estándar" },
    { from: "# Create Default parameters bars (Orange)", to: "# Crear barras de parámetros predeterminados (Naranja)" },
    { from: "# will require up to three months of data", to: "# requerirá hasta tres meses de datos" },
    { from: "# Get the training and test sets", to: "# Obtener los conjuntos de entrenamiento y prueba" },
    { from: "# Fit model", to: "# Ajustar modelo" },
    { from: "# Compute fraud detection performances", to: "# Calcular rendimientos de detección de fraude" },
    { from: "# Inicialización: Cargar funciones compartidas and simulated data", to: "# Inicialización: Cargar funciones compartidas y datos simulados" },
    { from: "# Get simulated data from Github repository", to: "# Obtener datos simulados del repositorio Github" },
    { from: "# Plot the training points", to: "# Graficar los puntos de entrenamiento" },
    { from: "# Recreate the train and test DafaFrames from these indices", to: "# Recrear los DataFrames de entrenamiento y prueba a partir de estos índices" },
    { from: "# Set the starting day for the training period, and the deltas", to: "# Establecer el día de inicio para el período de entrenamiento y los deltas" },
    { from: "# Select parameter of interest (n_estimators)", to: "# Seleccionar parámetro de interés (n_estimators)" },
    { from: /# Rename to ([\w_]+) for model performance comparison later in this section/g, to: "# Renombrar a $1 para comparación de rendimiento de modelos más adelante en esta sección" },
    { from: "# random_state is set to 0 for reproducibility", to: "# random_state se establece en 0 para reproducibilidad" },
    { from: "# Create a pipeline with the list of samplers, and the estimator", to: "# Crear un pipeline con la lista de muestreadores y el estimador" },
    { from: "# Get performances on the validation set using prequential validation", to: "# Obtener rendimientos en el conjunto de validación usando validación prequential" },
    { from: "# Get performances on the test set using prequential validation", to: "# Obtener rendimientos en el conjunto de prueba usando validación prequential" },
    { from: "# And return as a single DataFrame", to: "# Y devolver como un solo DataFrame" },
    { from: "# By default, scales input data", to: "# Por defecto, escala los datos de entrada" },
    { from: "# By default, scaling the input data", to: "# Por defecto, escalando los datos de entrada" },
    { from: "#Setting the model in training mode", to: "#Estableciendo el modelo en modo de entrenamiento" },
    { from: "#evaluating the model on the test set after each epoch", to: "#evaluando el modelo en el conjunto de prueba después de cada época" },
    { from: "#we did not call the function scaleData this time", to: "#no llamamos a la función scaleData esta vez" },
    { from: "# Let us rescale data for the next parts", to: "# Reescalemos los datos para las siguientes partes" },
    { from: "#categorical variables : encoding valid according to train", to: "#variables categóricas: codificación válida según entrenamiento" },
    { from: "# Only keep columns that are needed as argument to custom scoring function", to: "# Mantener solo las columnas necesarias como argumento para la función de puntuación personalizada" },
    { from: "# Only keep columns that are needed as argument to custome scoring function", to: "# Mantener solo las columnas necesarias como argumento para la función de puntuación personalizada" },
    { from: "# to reduce serialization time of transaction dataset", to: "# para reducir el tiempo de serialización del conjunto de datos de transacciones" },
    { from: "# storing the features x in self.feature and adding the \"padding\" transaction at the end", to: "# almacenando las características x en self.feature y agregando la transacción de \"relleno\" al final" },
    { from: "#these will get normalized but it should still work", to: "#estos se normalizarán pero debería seguir funcionando" },
    { from: "# We first train the classifier using the `fit` method, and pass as arguments the input and output features", to: "# Primero entrenamos el clasificador usando el método `fit`, y pasamos como argumentos las características de entrada y salida" },
    { from: "# We then get the predictions on the training and test data using the `predict_proba` method", to: "# Luego obtenemos las predicciones en los datos de entrenamiento y prueba usando el método `predict_proba`" },
    { from: "# Load shared functions", to: "# Cargar funciones compartidas" }
];

function scanDirectory(directory) {
    const files = fs.readdirSync(directory);

    files.forEach(file => {
        const fullPath = path.join(directory, file);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            if (file !== 'node_modules' && file !== '.git' && file !== '_build' && file !== '.ipynb_checkpoints' && file !== 'images') {
                scanDirectory(fullPath);
            }
        } else if (path.extname(file) === '.ipynb') {
            processNotebook(fullPath);
        }
    });
}

function processNotebook(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        let initialContent = content;
        let modified = false;

        // Naive string replace for JSON content can be dangerous if we match keys, but comments should be safe if specific enough
        // However, better to modify JSON object to be safe against accidental corruption of JSON structure

        const nb = JSON.parse(content);

        nb.cells.forEach(cell => {
            if (cell.cell_type === 'code') {
                cell.source = cell.source.map(line => {
                    let newLine = line;
                    for (const replacement of replacements) {
                        if (replacement.from instanceof RegExp) {
                            if (replacement.from.test(newLine)) {
                                newLine = newLine.replace(replacement.from, replacement.to);
                                modified = true;
                            }
                        } else {
                            if (newLine.includes(replacement.from)) {
                                // Match leading whitespace and # if possible to keep indentation, but simple replace usually works for comments
                                newLine = newLine.replace(replacement.from, replacement.to);
                                modified = true;
                            }
                        }
                    }
                    return newLine;
                });
            }
        });

        if (modified) {
            fs.writeFileSync(filePath, JSON.stringify(nb, null, 1), 'utf8'); // re-formatting with indent 1 to match typically
            console.log(`Updated ${filePath}`);
        } else {
            // console.log(`No changes for ${filePath}`);
        }

    } catch (e) {
        console.error(`Error processing ${filePath}: ${e.message}`);
    }
}

console.log("Applying final translations...");
scanDirectory(rootDir);
