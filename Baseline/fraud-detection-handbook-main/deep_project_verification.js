const fs = require('fs');
const path = require('path');

const rootDir = "c:\\Programacion\\fraud-detection-handbook-main";

// Advanced heuristics for English detection in code comments
const suspiciousPatterns = [
    // Common English verbs/nouns in comments
    /\bLoad\b/i,
    /\bDefine\b/i,
    /\bSetup\b/i,
    /\bCalculate\b/i,
    /\bCompute\b/i,
    /\bPlot\b/i,
    /\bSplit\b/i,
    /\bTrain\b/i,
    /\bTest\b/i,
    /\bValidate\b/i,
    /\bPerform\b/i,
    /\bCheck\b/i,
    /\bCreate\b/i,
    /\bUpdate\b/i,
    /\bSave\b/i,
    /\bRead\b/i,
    /\bWrite\b/i,
    /\bGet\b/i,
    /\bSet\b/i,
    /\bInitialize\b/i,
    /\bReturn\b/i,
    /\bFunction\b/i,
    /\bClass\b/i,
    /\bObject\b/i,
    /\bVariable\b/i,
    /\bParameter\b/i,
    /\bModel\b/i,
    /\bData\b/i,
    /\bSet of\b/i,
    /\bWidth of\b/i,
    /\bPosition of\b/i,
    /\bExecution time\b/i,
    /\bLabel\b/i,
    /\bFeature\b/i,
    /\bColumn\b/i,
    /\bRow\b/i,
    /\bDataset\b/i,
    /\bTransformation\b/i,
    /\bMatrix\b/i,
    /\bAccuracy\b/i,
    /\bPrecision\b/i,
    /\bRecall\b/i,
    /\bScore\b/i,
    /\bMetric\b/i
];

// Whitelist for false positives (words that are same in Spanish or technical terms)
const whitelist = [
    "Test", "Train", "Validation", // Often used in variable names or technical context
    "DataFrame", "Dataset",
    "Plot", "Matplotlib", "Seaborn",
    "Python", "Jupyter", "Notebook",
    "Google", "Colab", "Github",
    "Fraud", "Detection", "Handbook",
    "Recall", "Precision", "F1", "AUC", "ROC",
    "XGBoost", "LightGBM", "CatBoost",
    "Scikit-learn", "Sklearn",
    "Pandas", "Numpy",
    "Class", "Def", "Return", "Import", "From", // Python keywords
    "True", "False", "None", // Python keywords
    "Todo", "Fixme", // Standard markers
    "URL", "HTTP", "HTTPS", "API",
    "Id", "ID", "Risk", "Terminal", "Customer", "Transaction", // Field names
    "Day", "Night", "Weekend", // Field names often kept
    "Mean", "Std", "Min", "Max", "Count", "Sum" // Aggregation names often kept
];

let warnings = [];

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
            checkNotebook(fullPath);
        }
    });
}

function checkNotebook(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const nb = JSON.parse(content);

        nb.cells.forEach((cell, index) => {
            if (cell.cell_type === 'code') {
                cell.source.forEach((line, lineIndex) => {
                    const trimmedLine = line.trim();
                    if (trimmedLine.startsWith('#')) {
                        const commentText = trimmedLine.substring(1).trim();
                        // Check against patterns
                        for (const pattern of suspiciousPatterns) {
                            if (pattern.test(commentText)) {
                                // Check whitelist
                                const words = commentText.split(/\s+/);
                                const isWhitelisted = words.every(word => {
                                    const cleanWord = word.replace(/[^a-zA-Z0-9]/g, '');
                                    return whitelist.includes(cleanWord) || whitelist.some(w => w.toLowerCase() === cleanWord.toLowerCase());
                                });

                                // Special case: Allow mixed sentences if they look Spanish (heuristic: contains 'de', 'la', 'el', 'en', 'para', 'por')
                                const spanishIndicators = ['de', 'la', 'el', 'en', 'para', 'por', 'con', 'un', 'una', 'los', 'las'];
                                const hasSpanish = words.some(word => spanishIndicators.includes(word.toLowerCase()));

                                if (!isWhitelisted && !hasSpanish && commentText.length > 5) {
                                    // Filter out some really common technical short comments
                                    if (commentText !== "TODO" && !commentText.startsWith("------")) {
                                        warnings.push({
                                            file: path.relative(rootDir, filePath),
                                            cell: index,
                                            line: lineIndex,
                                            text: trimmedLine,
                                            reason: `Matched pattern: ${pattern}`
                                        });
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
    } catch (e) {
        console.error(`Error reading ${filePath}: ${e.message}`);
    }
}

console.log("Starting Deep Verification Scan...");
scanDirectory(rootDir);

const reportPath = path.join(rootDir, 'verification_report.txt');
let reportContent = "";

if (warnings.length > 0) {
    reportContent += `Found ${warnings.length} potential issues:\n`;
    warnings.forEach(w => {
        reportContent += `[${w.file}] Cell ${w.cell}, Line ${w.line}: "${w.text}"\n`;
    });
} else {
    reportContent += "No suspicious English comments found.\n";
}

fs.writeFileSync(reportPath, reportContent);
console.log(`Report written to ${reportPath}`);
