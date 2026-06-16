let session = null;
let modelReady = false;

const classNames = {
    0: "key",
    1: "saint Paul",
    2: "saint Peter",
    3: "sword"
};

const displayNames = {
    "key": "Ключ",
    "sword": "Меч",
    "saint Paul": "Апостол Павел",
    "saint Peter": "Апостол Пётр"
};

async function loadModel() {
    try {
        console.log('Загрузка модели...');
        const modelUrl = 'model/best.onnx';
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        modelReady = true;
        console.log('Модель успешно загружена!');
        document.getElementById('saintName').textContent = 'Модель готова';
        document.getElementById('saintDescription').textContent = 'Загрузите изображение для распознавания';
        return true;
    } catch (error) {
        console.error('Ошибка загрузки модели:', error);
        modelReady = false;
        document.getElementById('saintName').textContent = 'Ошибка модели';
        document.getElementById('saintDescription').textContent = 'Проверьте консоль для деталей';
        return false;
    }
}

async function processImage(image) {
    if (!modelReady) {
        alert('Модель еще не загружена, подождите...');
        return null;
    }
    
    try {
        console.log('Обработка изображения...');
        
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, 640, 640);
        
        const imageData = ctx.getImageData(0, 0, 640, 640);
        const data = imageData.data;
        const floatData = new Float32Array(3 * 640 * 640);
        
        for (let i = 0; i < data.length; i += 4) {
            const idx = i / 4;
            floatData[idx] = data[i] / 255.0;
            floatData[640*640 + idx] = data[i+1] / 255.0;
            floatData[2*640*640 + idx] = data[i+2] / 255.0;
        }
        
        const inputTensor = new ort.Tensor('float32', floatData, [1, 3, 640, 640]);
        const results = await session.run({ images: inputTensor });
        
        console.log('Инференс выполнен');
        
        const detections = results.output0.data;
        const numDetections = 8400;
        const numClasses = 4;
        const numCoords = 4;
        const numMaskCoeffs = 32;
        const totalAttrs = numCoords + numMaskCoeffs + numClasses;
        
        let maxConf = {
            "saint Peter": 0.0,
            "saint Paul": 0.0,
            "key": 0.0,
            "sword": 0.0
        };
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * totalAttrs;
            const classStart = offset + numCoords + numMaskCoeffs;
            
            for (let c = 0; c < numClasses; c++) {
                const score = detections[classStart + c];
                const prob = 1 / (1 + Math.exp(-score));
                const className = classNames[c];
                if (maxConf[className] !== undefined) {
                    if (prob > maxConf[className]) {
                        maxConf[className] = prob;
                    }
                }
            }
        }
        
        console.log('Максимальные уверенности:', maxConf);
        
        const peterRaw = Math.max(maxConf["saint Peter"] || 0, 0);
        const paulRaw = Math.max(maxConf["saint Paul"] || 0, 0);
        const keyConf = maxConf["key"] || 0;
        const swordConf = maxConf["sword"] || 0;
        
        const peterScore = 1 - (1 - peterRaw) * (1 - keyConf);
        const paulScore = 1 - (1 - paulRaw) * (1 - swordConf);
        const total = peterScore + paulScore;
        
        let peterProb = 0;
        let paulProb = 0;
        
        if (total > 0) {
            peterProb = peterScore / total;
            paulProb = paulScore / total;
        }
        
        console.log('Итоговые вероятности - Петр:', peterProb, 'Павел:', paulProb);
        
        let verdict = 'Неопределенно';
        let verdictColor = '#777';
        
        if (total === 0) {
            verdict = 'Апостол не определён';
        } else if (peterProb >= 0.65) {
            verdict = 'Апостол Пётр';
        } else if (paulProb >= 0.65) {
            verdict = 'Апостол Павел';
        } else if (peterProb >= 0.35 && paulProb >= 0.35) {
            verdict = 'Оба: Пётр и Павел';
        } else if (peterProb > paulProb) {
            verdict = 'Скорее Пётр';
        } else {
            verdict = 'Скорее Павел';
        }
        
        let evidence = [];
        if (peterRaw > 0.1) evidence.push('Фигура Петра ' + (peterRaw * 100).toFixed(0) + '%');
        if (keyConf > 0.1) evidence.push('Ключ ' + (keyConf * 100).toFixed(0) + '%');
        if (paulRaw > 0.1) evidence.push('Фигура Павла ' + (paulRaw * 100).toFixed(0) + '%');
        if (swordConf > 0.1) evidence.push('Меч ' + (swordConf * 100).toFixed(0) + '%');
        
        const evidenceText = evidence.length > 0 ? evidence.join(' + ') : 'Нет уверенных обнаружений';
        
        return {
            verdict: verdict,
            peter_probability: peterProb,
            paul_probability: paulProb,
            peterRaw: peterRaw,
            paulRaw: paulRaw,
            keyConf: keyConf,
            swordConf: swordConf,
            evidence: evidenceText
        };
    } catch (error) {
        console.error('Ошибка обработки:', error);
        return null;
    }
}

function loadInfo(verdict, callback) {
    if (verdict === 'Апостол Пётр' || verdict === 'Скорее Пётр') {
        fetch('assets/peter_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Пётр с ключами');
            });
    } else if (verdict === 'Апостол Павел' || verdict === 'Скорее Павел') {
        fetch('assets/paul_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Павел с мечом');
            });
    } else {
        callback('Святой не определён');
    }
}

const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('previewContainer');
const saintName = document.getElementById('saintName');
const saintDescription = document.getElementById('saintDescription');
const probabilities = document.getElementById('probabilities');

loadModel();

imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    saintName.textContent = 'Обработка...';
    saintDescription.textContent = 'Анализ изображения';
    probabilities.innerHTML = '';
    
    const reader = new FileReader();
    reader.onload = function(event) {
        preview.src = event.target.result;
        previewContainer.style.display = 'block';
        
        preview.onload = function() {
            processImage(preview).then(function(result) {
                if (!result) {
                    saintName.textContent = 'Ошибка';
                    saintDescription.textContent = 'Не удалось обработать изображение';
                    return;
                }
                
                saintName.textContent = result.verdict;
                
                loadInfo(result.verdict, function(desc) {
                    saintDescription.textContent = desc;
                });
                
                const peterPercent = (result.peter_probability * 100).toFixed(1);
                const paulPercent = (result.paul_probability * 100).toFixed(1);
                
                probabilities.innerHTML = 
                    '<div style="text-align:left; padding:10px;">' +
                    '<p><strong>Пётр:</strong> ' + peterPercent + '%</p>' +
                    '<p style="font-size:0.9rem; color:#aaa; margin-left:15px;">Доказательства: ' + result.evidence + '</p>' +
                    '<p><strong>Павел:</strong> ' + paulPercent + '%</p>' +
                    '</div>';
            });
        };
    };
    reader.readAsDataURL(file);
});