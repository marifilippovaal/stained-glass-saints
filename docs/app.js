let session = null;
let modelReady = false;

const classNames = {
    0: "key",
    1: "sword", 
    2: "saint Peter",
    3: "saint Paul"
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
        
        let classConfs = {
            "saint Peter": 0.0,
            "saint Paul": 0.0,
            "key": 0.0,
            "sword": 0.0
        };
        
        const numDetections = 8400;
        const numClasses = 4;
        const numCoords = 4;
        const numMaskCoeffs = 32;
        const totalAttrs = numCoords + numMaskCoeffs + numClasses;
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * totalAttrs;
            const classStart = offset + numCoords + numMaskCoeffs;
            
            for (let c = 0; c < numClasses; c++) {
                const score = detections[classStart + c];
                const prob = 1 / (1 + Math.exp(-score));
                
                const className = classNames[c];
                if (classConfs[className] !== undefined) {
                    if (prob > classConfs[className]) {
                        classConfs[className] = prob;
                    }
                }
            }
        }
        
        console.log('Детекции:', classConfs);
        
        const paulConf = classConfs["saint Paul"];
        const peterConf = classConfs["saint Peter"];
        const keyConf = classConfs["key"];
        const swordConf = classConfs["sword"];
        
        const peterScore = 1 - (1 - peterConf) * (1 - keyConf);
        const paulScore = 1 - (1 - paulConf) * (1 - swordConf);
        
        const totalScore = peterScore + paulScore;
        
        let peterProb = 0;
        let paulProb = 0;
        
        if (totalScore > 0) {
            peterProb = peterScore / totalScore;
            paulProb = paulScore / totalScore;
        }
        
        console.log('Итоговые вероятности - Петр:', peterProb, 'Павел:', paulProb);
        
        let verdict = 'Неопределенно';
        
        if (totalScore === 0) {
            verdict = 'Не определено';
        } else if (peterProb >= 0.65 && paulProb < 0.35) {
            verdict = 'Апостол Петр';
        } else if (paulProb >= 0.65 && peterProb < 0.35) {
            verdict = 'Апостол Павел';
        } else if (peterProb >= 0.35 && paulProb >= 0.35) {
            verdict = 'Возможно оба';
        } else if (peterProb > paulProb) {
            verdict = 'Скорее Петр';
        } else {
            verdict = 'Скорее Павел';
        }
        
        let evidencePeter = [];
        let evidencePaul = [];
        
        if (peterConf > 0.1) evidencePeter.push('Пётр ' + (peterConf * 100).toFixed(0) + '%');
        if (keyConf > 0.1) evidencePeter.push('ключ ' + (keyConf * 100).toFixed(0) + '%');
        if (paulConf > 0.1) evidencePaul.push('Павел ' + (paulConf * 100).toFixed(0) + '%');
        if (swordConf > 0.1) evidencePaul.push('меч ' + (swordConf * 100).toFixed(0) + '%');
        
        return {
            verdict: verdict,
            peter_probability: peterProb,
            paul_probability: paulProb,
            peterConf: peterConf,
            paulConf: paulConf,
            keyConf: keyConf,
            swordConf: swordConf,
            evidencePeter: evidencePeter.join(' + ') || 'нет признаков',
            evidencePaul: evidencePaul.join(' + ') || 'нет признаков'
        };
    } catch (error) {
        console.error('Ошибка обработки:', error);
        return null;
    }
}

function loadInfo(verdict, callback) {
    if (verdict === 'Апостол Петр' || verdict === 'Скорее Петр') {
        fetch('assets/peter_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Петр с ключами');
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
    } else if (verdict === 'Возможно оба') {
        callback('На изображении могут быть оба апостола');
    } else {
        callback('Святой не определен');
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
                    '<p style="font-size:0.9rem; color:#aaa; margin-left:15px;">' + result.evidencePeter + '</p>' +
                    '<p><strong>Павел:</strong> ' + paulPercent + '%</p>' +
                    '<p style="font-size:0.9rem; color:#aaa; margin-left:15px;">' + result.evidencePaul + '</p>' +
                    '</div>';
            });
        };
    };
    reader.readAsDataURL(file);
});