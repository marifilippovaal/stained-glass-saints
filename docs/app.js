let session = null;
let classes = {};
let classColors = {};

async function loadModel() {
    try {
        const classResponse = await fetch('model/classes.json');
        classes = await classResponse.json();
        
        Object.keys(classes).forEach(key => {
            classColors[key] = `hsl(${Math.random() * 360}, 80%, 60%)`;
        });

        const modelUrl = 'model/best.onnx';
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log('Модель загружена');
        return true;
    } catch (error) {
        console.error('Ошибка загрузки:', error);
        return false;
    }
}

async function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 640, 640);
    
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const data = imageData.data;
    
    const float32Data = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        float32Data[idx] = data[i] / 255.0;
        float32Data[640*640 + idx] = data[i+1] / 255.0;
        float32Data[2*640*640 + idx] = data[i+2] / 255.0;
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

function postprocess(outputs, confThreshold = 0.25) {
    const boxes = [];
    const scores = [];
    const classIds = [];
    const masks = [];

    const detections = outputs[0].data;
    const maskCoeffs = outputs[1].data;

    const numDetections = 8400;
    const numClasses = 4;
    const numCoords = 4;
    const numMaskCoeffs = 32;

    for (let i = 0; i < numDetections; i++) {
        const offset = i * (numCoords + numMaskCoeffs + numClasses);
        const x1 = detections[offset];
        const y1 = detections[offset + 1];
        const x2 = detections[offset + 2];
        const y2 = detections[offset + 3];
        
        let maxScore = -Infinity;
        let bestClass = -1;
        const classStart = offset + numCoords + numMaskCoeffs;
        for (let c = 0; c < numClasses; c++) {
            const score = detections[classStart + c];
            if (score > maxScore) {
                maxScore = score;
                bestClass = c;
            }
        }
        
        if (maxScore > confThreshold) {
            const maskData = [];
            const maskStart = offset + numCoords;
            for (let m = 0; m < numMaskCoeffs; m++) {
                maskData.push(detections[maskStart + m]);
            }
            
            boxes.push([x1, y1, x2, y2]);
            scores.push(maxScore);
            classIds.push(bestClass);
            masks.push(maskData);
        }
    }
    
    return { boxes, scores, classIds, masks };
}

async function predict(image) {
    if (!session) {
        alert('Модель еще не загружена');
        return;
    }

    try {
        const inputTensor = await preprocessImage(image);
        const results = await session.run({ images: inputTensor });
        
        const detections = postprocess([results.output0, results.output1]);
        
        let peterProb = 0;
        let paulProb = 0;
        
        for (let i = 0; i < detections.classIds.length; i++) {
            const classId = detections.classIds[i];
            const score = detections.scores[i];
            
            if (classes[classId] === 'saint Peter') peterProb = score;
            if (classes[classId] === 'saint Paul') paulProb = score;
        }
        
        let verdict = 'Неопределенно';
        if (peterProb > paulProb && peterProb > 0.3) verdict = 'Апостол Петр';
        else if (paulProb > peterProb && paulProb > 0.3) verdict = 'Апостол Павел';
        
        return {
            verdict: verdict,
            peter_probability: peterProb,
            paul_probability: paulProb,
            detections: detections
        };
    } catch (error) {
        console.error('Ошибка инференса:', error);
        return null;
    }
}

const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('previewContainer');
const saintName = document.getElementById('saintName');
const saintDescription = document.getElementById('saintDescription');
const probabilities = document.getElementById('probabilities');

loadModel();

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (event) => {
        preview.src = event.target.result;
        previewContainer.style.display = 'block';
        
        await new Promise(resolve => preview.onload = resolve);
        
        const result = await predict(preview);
        if (!result) return;
        
        saintName.textContent = result.verdict;
        
        let info = {};
        if (result.verdict === 'Апостол Петр') {
            const resp = await fetch('assets/peter_info.json');
            info = await resp.json();
        } else if (result.verdict === 'Апостол Павел') {
            const resp = await fetch('assets/paul_info.json');
            info = await resp.json();
        }
        saintDescription.textContent = info.description || '';
        
        probabilities.innerHTML = `
            <div class="prob-bar">
                <span>Св. Петр</span>
                <div class="bar"><div style="width:${result.peter_probability * 100}%; background:#4CAF50;"></div></div>
                <span>${(result.peter_probability * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-bar">
                <span>Св. Павел</span>
                <div class="bar"><div style="width:${result.paul_probability * 100}%; background:#2196F3;"></div></div>
                <span>${(result.paul_probability * 100).toFixed(1)}%</span>
            </div>
        `;
    };
    reader.readAsDataURL(file);
});