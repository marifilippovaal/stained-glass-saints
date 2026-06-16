let session = null;
let modelReady = false;
let modelLoading = false;

const classNames = {
    0: "key",
    1: "saint Paul",
    2: "saint Peter",
    3: "sword"
};

async function loadModel() {
    if (modelLoading || modelReady) {
        return;
    }
    modelLoading = true;
    try {
        console.log('Загрузка модели...');
        document.getElementById('saintName').textContent = 'Загрузка модели...';
        document.getElementById('saintDescription').textContent = 'Пожалуйста, подождите';
        
        const modelUrl = 'model/best.onnx';
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        modelReady = true;
        modelLoading = false;
        console.log('Модель успешно загружена!');
        document.getElementById('saintName').textContent = 'Модель готова';
        document.getElementById('saintDescription').textContent = 'Загрузите изображение для распознавания';
        return true;
    } catch (error) {
        console.error('Ошибка загрузки модели:', error);
        modelReady = false;
        modelLoading = false;
        document.getElementById('saintName').textContent = 'Ошибка модели';
        document.getElementById('saintDescription').textContent = 'Перезагрузите страницу';
        return false;
    }
}

async function processImage(image) {
    if (!modelReady) {
        document.getElementById('saintName').textContent = 'Загрузка модели...';
        document.getElementById('saintDescription').textContent = 'Подождите, модель загружается';
        return null;
    }
    
    try {
        console.log('Обработка изображения...');
        document.getElementById('saintName').textContent = 'Обработка...';
        document.getElementById('saintDescription').textContent = 'Анализ изображения';
        
        var canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, 640, 640);
        
        var imageData = ctx.getImageData(0, 0, 640, 640);
        var data = imageData.data;
        var floatData = new Float32Array(3 * 640 * 640);
        
        for (var i = 0; i < data.length; i += 4) {
            var idx = i / 4;
            floatData[idx] = data[i] / 255.0;
            floatData[640*640 + idx] = data[i+1] / 255.0;
            floatData[2*640*640 + idx] = data[i+2] / 255.0;
        }
        
        var inputTensor = new ort.Tensor('float32', floatData, [1, 3, 640, 640]);
        var results = await session.run({ images: inputTensor });
        
        console.log('Инференс выполнен');
        
        var detections = results.output0.data;
        var numDetections = detections.length / 38;
        var numClasses = 4;
        var numCoords = 4;
        var numMaskCoeffs = 32;
        var totalAttrs = 38;
        
        console.log('Детекций после NMS:', numDetections);
        
        var maxConf = {
            "key": 0.0,
            "saint Paul": 0.0,
            "saint Peter": 0.0,
            "sword": 0.0
        };
        
        for (var i2 = 0; i2 < numDetections; i2++) {
            var offset = i2 * totalAttrs;
            var classStart = offset + numCoords + numMaskCoeffs;
            
            for (var c = 0; c < numClasses; c++) {
                var score = detections[classStart + c];
                var prob = 1 / (1 + Math.exp(-score));
                var className = classNames[c];
                if (maxConf[className] !== undefined) {
                    if (prob > maxConf[className]) {
                        maxConf[className] = prob;
                    }
                }
            }
        }
        
        console.log('Максимальные уверенности:', maxConf);
        
        var peterRaw = maxConf["saint Peter"] || 0;
        var paulRaw = maxConf["saint Paul"] || 0;
        var keyConf = maxConf["key"] || 0;
        var swordConf = maxConf["sword"] || 0;
        
        console.log('peterRaw:', peterRaw, 'paulRaw:', paulRaw, 'keyConf:', keyConf, 'swordConf:', swordConf);
        
        var peterScore = 1 - (1 - peterRaw) * (1 - keyConf);
        var paulScore = 1 - (1 - paulRaw) * (1 - swordConf);
        var total = peterScore + paulScore;
        
        var peterProb = 0;
        var paulProb = 0;
        
        if (total > 0) {
            peterProb = peterScore / total;
            paulProb = paulScore / total;
        }
        
        console.log('Пётр:', peterProb, 'Павел:', paulProb);
        
        var verdict = 'Неопределенно';
        
        if (total === 0 || (peterProb < 0.3 && paulProb < 0.3)) {
            verdict = 'Апостол не определён';
        } else if (peterProb >= 0.65 && paulProb < 0.35) {
            verdict = 'Апостол Пётр';
        } else if (paulProb >= 0.65 && peterProb < 0.35) {
            verdict = 'Апостол Павел';
        } else if (peterProb >= 0.35 && paulProb >= 0.35) {
            if (peterProb > paulProb) {
                verdict = 'Скорее Пётр (оба присутствуют)';
            } else {
                verdict = 'Скорее Павел (оба присутствуют)';
            }
        } else if (peterProb > paulProb) {
            verdict = 'Скорее Пётр';
        } else {
            verdict = 'Скорее Павел';
        }
        
        var evidence = [];
        if (peterRaw > 0.3) evidence.push('Фигура Петра ' + (peterRaw * 100).toFixed(0) + '%');
        if (keyConf > 0.3) evidence.push('Ключ ' + (keyConf * 100).toFixed(0) + '%');
        if (paulRaw > 0.3) evidence.push('Фигура Павла ' + (paulRaw * 100).toFixed(0) + '%');
        if (swordConf > 0.3) evidence.push('Меч ' + (swordConf * 100).toFixed(0) + '%');
        
        var evidenceText = evidence.length > 0 ? evidence.join(' + ') : 'Нет уверенных обнаружений';
        
        console.log('Вердикт:', verdict);
        
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
    if (verdict.includes('Пётр') && !verdict.includes('Павел')) {
        fetch('assets/peter_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Пётр с ключами');
            });
    } else if (verdict.includes('Павел') && !verdict.includes('Пётр')) {
        fetch('assets/paul_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Павел с мечом');
            });
    } else if (verdict.includes('оба')) {
        callback('На изображении присутствуют оба апостола');
    } else {
        callback('Святой не определён');
    }
}

var imageInput = document.getElementById('imageInput');
var preview = document.getElementById('preview');
var previewContainer = document.getElementById('previewContainer');
var saintName = document.getElementById('saintName');
var saintDescription = document.getElementById('saintDescription');
var probabilities = document.getElementById('probabilities');

loadModel();

imageInput.addEventListener('change', function(e) {
    var file = e.target.files[0];
    if (!file) return;
    
    saintName.textContent = 'Обработка...';
    saintDescription.textContent = 'Анализ изображения';
    probabilities.innerHTML = '';
    
    var reader = new FileReader();
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
                
                var peterPercent = (result.peter_probability * 100).toFixed(1);
                var paulPercent = (result.paul_probability * 100).toFixed(1);
                
                probabilities.innerHTML = 
                    '<div style="text-align:left; padding:10px;">' +
                    '<p><strong>Пётр:</strong> ' + peterPercent + '%</p>' +
                    '<p><strong>Павел:</strong> ' + paulPercent + '%</p>' +
                    '<p style="font-size:0.9rem; color:#aaa; margin-top:10px;">Доказательства: ' + result.evidence + '</p>' +
                    '</div>';
            });
        };
    };
    reader.readAsDataURL(file);
});