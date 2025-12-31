const fs = require('fs');
const path = require('path');

const notebookPath = path.join(__dirname, 'ModelSelection_RealWorldData.ipynb');

if (!fs.existsSync(notebookPath)) {
    console.error(`Notebook file not found at: ${notebookPath}`);
    process.exit(1);
}

let content = fs.readFileSync(notebookPath, 'utf8');

// Define replacements
const replacements = [
    // Code Comments
    {
        search: '# Initialization: Load shared functions and simulated data',
        replace: '# Inicialización: Cargar funciones compartidas y datos simulados'
    },
    {
        search: '# Load shared functions',
        replace: '# Cargar funciones compartidas'
    },
    {
        search: '# Get simulated data from Github repository',
        replace: '# Obtener datos simulados del repositorio Github'
    },
    // Markdown Text
    {
        search: 'Increasing the number of trees allows to slighlty increase the Average Precicision and the CP@100. It however has little influence on the AUC ROC, for which 25 trees already provide optimal performances.',
        replace: 'Aumentar el número de árboles permite incrementar ligeramente la Precisión Promedio y el CP@100. Sin embargo, tiene poca influencia en el AUC ROC, para el cual 25 árboles ya proporcionan rendimientos óptimos.'
    },
    {
        search: 'Three different parameters are assessed for boosting: The tree depth (`max_depth` parameter) taking values in the set [3,6,9], the number of trees (`n_estimators` parameter) taking values in the set [25,50,100] and the learning rate (`learning_rate` parameter) taking values in the set [0.1, 0.3]. The optimal parameters are a combination of 100 trees with a maximum depth of 6, and a learning rate of 0.1, except for the CP@100 where 50 trees provide the best performance.',
        replace: 'Se evalúan tres parámetros diferentes para el boosting: La profundidad del árbol (parámetro `max_depth`) tomando valores en el conjunto [3,6,9], el número de árboles (parámetro `n_estimators`) tomando valores en el conjunto [25,50,100] y la tasa de aprendizaje (parámetro `learning_rate`) tomando valores en el conjunto [0.1, 0.3]. Los parámetros óptimos son una combinación de 100 árboles con una profundidad máxima de 6, y una tasa de aprendizaje de 0.1, excepto para el CP@100 donde 50 árboles proporcionan el mejor rendimiento.'
    },
    {
        search: 'For better visualization, we follow the same approach as with the [simulated dataset](Model_Selection_XGBoost). Let us first report the performances as a function of the tree depth, for a fixed number of 100 trees and a learning rate of 0.1.',
        replace: 'Para una mejor visualización, seguimos el mismo enfoque que con el [conjunto de datos simulado](Model_Selection_XGBoost). Reportemos primero los rendimientos en función de la profundidad del árbol, para un número fijo de 100 árboles y una tasa de aprendizaje de 0.1.'
    },
    {
        search: 'Similar to the [simulated dataset](Model_Selection_XGBoost), the peformances first increase with the tree depth, before reaching an optimum and decreasing. The optimal tree depth is found around 6.\\n",\n    "\\n",\n    "Let us then report the performances as a function of the number of trees, for a fixed depth of 6 and a learning rate 0.1.',
        replace: 'De manera similar al [conjunto de datos simulado](Model_Selection_XGBoost), los rendimientos primero aumentan con la profundidad del árbol, antes de alcanzar un óptimo y disminuir. La profundidad óptima del árbol se encuentra alrededor de 6.\\n",\n    "\\n",\n    "Reportemos entonces los rendimientos en función del número de árboles, para una profundidad fija de 6 y una tasa de aprendizaje de 0.1.'
    },
    // Code Strings (Plotting/Labels)
    {
        search: 'parameter_name="Number of trees/Maximum tree depth"',
        replace: 'parameter_name="Número de árboles/Profundidad máxima del árbol"'
    },
    {
        search: "expe_type_list=['Test','Validation']",
        replace: "expe_type_list=['Prueba','Validación']"
    },
    // Table Headers/Output
    {
        search: '<th>Execution time</th>',
        replace: '<th>Tiempo de ejecución</th>'
    },
    {
        search: '<th>Parameters summary</th>',
        replace: '<th>Resumen de parámetros</th>'
    },
    {
        search: '<th>Best estimated parameters</th>',
        replace: '<th>Mejores parámetros estimados</th>'
    },
    {
        search: '<th>Validation performance</th>',
        replace: '<th>Rendimiento de validación</th>'
    },
    {
        search: '<th>Test performance</th>',
        replace: '<th>Rendimiento de prueba</th>'
    },
    {
        search: '<th>Optimal parameter(s)</th>',
        replace: '<th>Parámetro(s) óptimo(s)</th>'
    },
    {
        search: '<th>Optimal test performance</th>',
        replace: '<th>Rendimiento de prueba óptimo</th>'
    },
    {
        search: '<th>Average precision</th>',
        replace: '<th>Precisión promedio</th>'
    },
    {
        search: '<th>Card Precision@100</th>',
        replace: '<th>Precisión Tarjeta@100</th>'
    },
    // Plain text table headers (some might appear in text/plain output fields)
    {
        search: 'Best estimated parameters',
        replace: 'Mejores parámetros estimados'
    },
    {
        search: 'Validation performance',
        replace: 'Rendimiento de validación'
    },
    {
        search: 'Test performance',
        replace: 'Rendimiento de prueba'
    },
    {
        search: 'Optimal parameter(s)',
        replace: 'Parámetro(s) óptimo(s)'
    },
    {
        search: 'Optimal test performance',
        replace: 'Rendimiento de prueba óptimo'
    },
    {
        search: 'AUC ROC Average precision Card Precision@100',
        replace: 'AUC ROC Precisión promedio Precisión Tarjeta@100'
    }
];

let modifiedCount = 0;
replacements.forEach(item => {
    // Use a global replace to catch multiple occurrences
    const regex = new RegExp(item.search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
    if (regex.test(content)) {
        content = content.replace(regex, item.replace);
        modifiedCount++;
        console.log(`Applied translation: "${item.search.substring(0, 30)}..."`);
    }
});

fs.writeFileSync(notebookPath, content, 'utf8');
console.log(`Successfully finished translations. Applied ${modifiedCount} patterns.`);
