
const fs = require('fs');
const path = require('path');

const directoryPath = __dirname;
const outputFile = path.join(directoryPath, 'strings_to_translate_ch7.txt');

const files = fs.readdirSync(directoryPath).filter(file => file.endsWith('.ipynb'));

let outputContent = '';

files.forEach(file => {
    const filePath = path.join(directoryPath, file);
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const nb = JSON.parse(content);

        outputContent += `\n\n--- FILE: ${file} ---\n\n`;

        if (nb.cells) {
            nb.cells.forEach((cell, index) => {
                if (cell.cell_type === 'markdown') {
                    if (cell.source) {
                        outputContent += `[MARKDOWN CELL ${index}]\n`;
                        cell.source.forEach(line => {
                            if (line.trim().length > 0) {
                                outputContent += line + '\n';
                            }
                        });
                    }
                } else if (cell.cell_type === 'code') {
                    if (cell.source) {
                        let hastext = false;
                        cell.source.forEach(line => {
                            if (line.includes('#') || line.includes('"') || line.includes("'")) {
                                if (!hastext) {
                                    outputContent += `[CODE CELL ${index} - POTENTIAL TEXT]\n`;
                                    hastext = true;
                                }
                                outputContent += line + '\n';
                            }
                        });
                    }
                }
            });
        }

    } catch (e) {
        console.error(`Error reading ${file}: ${e.message}`);
    }
});

fs.writeFileSync(outputFile, outputContent, 'utf8');
console.log(`Strings extracted to ${outputFile}`);
