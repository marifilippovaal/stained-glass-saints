let session = null;
let classes = {};
let modelReady = false;

async function loadModel() {
    try {
        console.log('Загрузка модели...');
        
        const response = await fetch('model/classes.json');
        classes = await response.json();
        console.log('Классы загружены:', classes);
        
        const modelUrl = 'model/best.onnx';
        console.log('Загрузка ONNX модели...');
        
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

function getResult(peterProb, paulProb) {
    if (peterProb > paulProb && peterProb > 0.3) return 'Апостол Петр';
    if (paulProb > peterProb && paulProb > 0.3) return 'Апостол Павел';
    return 'Неопределенно';
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
        let peterProb = 0;
        let paulProb = 0;
        
        for (let i = 0; i < 8400; i++) {
            const offset = i * 40;
            const classStart = offset + 36;
            const scorePeter = detections[classStart + 2];
            const scorePaul = detections[classStart + 3];
            if (scorePeter > peterProb) peterProb = scorePeter;
            if (scorePaul > paulProb) paulProb = scorePaul;
        }
        
        console.log('Вероятности - Петр:', peterProb, 'Павел:', paulProb);
        
        return {
            verdict: getResult(peterProb, paulProb),
            peter: peterProb,
            paul: paulProb
        };
    } catch (error) {
        console.error('Ошибка обработки:', error);
        return null;
    }
}

function loadInfo(verdict, callback) {
    if (verdict === 'Апостол Петр') {
        fetch('assets/peter_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Петр с ключами');
            });
    } else if (verdict === 'Апостол Павел') {
        fetch('assets/paul_info.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                callback(data.description);
            })
            .catch(function() {
                callback('Апостол Павел с мечом');
            });
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
                
                const peterPercent = (result.peter * 100).toFixed(1);
                const paulPercent = (result.paul * 100).toFixed(1);
                
                probabilities.innerHTML = 
                    '<div class="prob-bar">' +
                    '<span>Св. Петр</span>' +
                    '<div class="bar"><div style="width:' + peterPercent + '%; background:#4CAF50;"></div></div>' +
                    '<span>' + peterPercent + '%</span>' +
                    '</div>' +
                    '<div class="prob-bar">' +
                    '<span>Св. Павел</span>' +
                    '<div class="bar"><div style="width:' + paulPercent + '%; background:#2196F3;"></div></div>' +
                    '<span>' + paulPercent + '%</span>' +
                    '</div>';
            });
        };
    };
    reader.readAsDataURL(file);
});