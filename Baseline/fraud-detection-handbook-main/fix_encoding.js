
const fs = require('fs');
const path = 'c:\\Programacion\\fraud-detection-handbook-main\\Chapter_References\\shared_functions.json';

try {
    let content = fs.readFileSync(path, 'utf8');

    // Fix encoding artifacts (Mojibake)
    // UTF-8 bytes interpreted as Windows-1252/Latin-1
    const replacements = [
        { from: /Ã¡/g, to: 'á' },
        { from: /Ã©/g, to: 'é' },
        { from: /Ã­/g, to: 'í' },
        { from: /Ã³/g, to: 'ó' },
        { from: /Ãº/g, to: 'ú' },
        { from: /Ã±/g, to: 'ñ' },
        { from: /Ã/g, to: 'Á' } // Be careful with this one, handled last if needed, or check mostly used for Á (C3 81) which might show as Ã<control>
    ];

    // Better approach: decoding manually trick? 
    // If the file was written as UTF-8 but read as Latin-1 then written back as UTF-8, we have double encoded chars.
    // If I just replace the common ones I see in Spanish text 'áéíóúñ', it should be sufficient for this context.
    
    // Specific correct mappings for common spanish chars found in mojibake
    // á: \xC3\xA1 -> Ã¡
    // é: \xC3\xA9 -> Ã©
    // í: \xC3\xAD -> Ã­
    // ó: \xC3\xB3 -> Ã³
    // ú: \xC3\xBA -> Ãº
    // ñ: \xC3\xB1 -> Ã±
    // Á: \xC3\x81 -> Ã (plus invisible char? 0x81 is control in 1252)
    // É: \xC3\x89 -> Ã‰
    // Í: \xC3\x8D -> Ã
    // Ó: \xC3\x93 -> Ã“
    // Ú: \xC3\x9A -> Ãš
    // Ñ: \xC3\x91 -> Ã‘
    
    const map = {
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã‰': 'É',
        'Ã“': 'Ó',
        'Ãš': 'Ú',
        'Ã‘': 'Ñ'
        // 'Ã': 'Á' // 0x81 is often not printable or mapped differently. Let's skip for now unless seen.
        // Found 'Ã' in previous step? No, saw 'CapÃ­tulo' etc.
    };

    Object.keys(map).forEach(key => {
        content = content.split(key).join(map[key]);
    });

    fs.writeFileSync(path, content, 'utf8');
    console.log('Encoding fixed.');

} catch (err) {
    console.error('Error:', err);
}
