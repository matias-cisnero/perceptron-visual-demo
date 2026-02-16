const canvas = document.getElementById("grafico");
const ctx = canvas.getContext("2d");
const trainButton = document.getElementById("btn-entrenar");
const resultDisplay = document.getElementById("resultado");
const controls = document.querySelectorAll('.controls-container input, .controls-container select');

let stopTrainingFlag = false;
let lastWeights = {}; 

const offScreenCanvas = document.createElement('canvas');
const offCtx = offScreenCanvas.getContext('2d');

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
}

async function perceptronSimple(x, y, eta, cota) {
    let w = [0, 0, 1];
    let error = Infinity, i = 0, errorMin = Infinity, wMin = [...w];
    const xExt = x.map(p => [...p, 1]);
    const inicio_tiempo = performance.now();

    while (error > 0 && i < cota && !stopTrainingFlag) {
        const idx = Math.floor(Math.random() * x.length);
        const salida = signo(dot(xExt[idx], w));
        const delta = xExt[idx].map(val => eta * (y[idx] - salida) * val);
        w = w.map((wi, j) => wi + delta[j]);

        error = calcularError(x, y, "simple", { w_simple: w });
        if (error < errorMin) {
            errorMin = error;
            wMin = [...w];
        }

        lastWeights = { w_simple: w };
        graficar(x, y, lastWeights, "simple");
        await new Promise(res => setTimeout(res, 50));
        i++;
    }

    const fin_total = round((performance.now() - inicio_tiempo) / 1000, 5);
    let output_messages = [`Épocas necesarias: ${i}`, `Error final: ${errorMin}`, `Tiempo total: ${fin_total} s.`];
    
    if (stopTrainingFlag) output_messages.push("Detenido manualmente.");
    else if (errorMin > 0) output_messages.push("No convergió (problema no lineal).");

    return { weights: wMin, output_text: output_messages.join("\n") };
}

async function perceptronMulticapa(x, y, eta, cota) {
    const num_hidden = 2;
    const num_output = 1;
    const num_input_bias = x[0].length + 1;

    let wjk = Array(num_input_bias).fill(0).map(() => Array(num_hidden).fill(0).map(() => (Math.random() * 2) - 1));
    let wij = Array(num_hidden + 1).fill(0).map(() => Array(num_output).fill(0).map(() => (Math.random() * 2) - 1));

    let error_min = Infinity;
    let wjk_min = JSON.parse(JSON.stringify(wjk));
    let wij_min = JSON.parse(JSON.stringify(wij));
    
    const xExt = x.map(p => [...p, 1]);
    const y_scaled = y.map(val => [val]);
    const inicio_tiempo = performance.now();
    let epochs = 0;
    let perfect = false;

    for (let e = 0; e < cota && !stopTrainingFlag; e++) {
        epochs = e + 1;
        const indices = Array.from({ length: x.length }, (_, i) => i).sort(() => Math.random() - 0.5);

        for (const idx of indices) {
            if (stopTrainingFlag) break;
            const Vk = toRowVector(xExt[idx]);
            
            const h_j = matMul(Vk, wjk)[0];
            const Vj_bias = toRowVector([...h_j.map(v => g(v)), 1]);
            const h_i = matMul(Vj_bias, wij)[0][0];
            const Vi = g(h_i);

            const delta_i = toColumnVector([g_derivada(h_i) * (y_scaled[idx][0] - Vi)]);
            const delta_j = toColumnVector(h_j.map((val, j) => g_derivada(val) * matMul(delta_i, transpose(wij.slice(0, num_hidden)))[0][j]));

            wij = matAdd(wij, scalarMul(eta, matMul(transpose(Vj_bias), delta_i)));
            wjk = matAdd(wjk, transpose(scalarMul(eta, matMul(delta_j, Vk))));
        }

        const curr_error = calcularError(x, y, "multicapa", { wjk, wij });
        if (curr_error < error_min) {
            error_min = curr_error;
            wij_min = JSON.parse(JSON.stringify(wij));
            wjk_min = JSON.parse(JSON.stringify(wjk));
        }

        lastWeights = { wjk, wij };
        if (checkAllClassifiedCorrectly(x, y, wjk, wij)) {
            perfect = true;
            break;
        }

        graficar(x, y, lastWeights, "multicapa");
        await new Promise(res => setTimeout(res, 10));
    }

    const fin_total = round((performance.now() - inicio_tiempo) / 1000, 5);
    const final_error = calcularError(x, y, "multicapa", { wjk: wjk_min, wij: wij_min });
    let msgs = [`Épocas: ${epochs}`, `Error final: ${final_error.toFixed(6)}`, `Tiempo: ${fin_total} s.`];
    
    if (perfect) msgs.push("Clasificación perfecta.");
    else if (stopTrainingFlag) msgs.push("Detenido manualmente.");

    return { wjk: wjk_min, wij: wij_min, output_text: msgs.join("\n") };
}

function graficar(x, y, weights, modelo) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width;
    const height = canvas.height;
    
    // 1. ÁREA DE DIBUJO
    const size = Math.min(width, height);
    const offsetX = (width - size) / 2;
    const offsetY = (height - size) / 2;

    ctx.clearRect(0, 0, width, height);

    // 2. GENERAR HEATMAP (Un poco más grande que 6x6 para evitar bordes blancos internos)
    const range = 7; // Generamos rango 7 (-3.5 a 3.5)
    
    const heatW = 150; 
    const heatH = 150;
    offScreenCanvas.width = heatW;
    offScreenCanvas.height = heatH;
    
    const imgData = offCtx.createImageData(heatW, heatH);
    const data = imgData.data;
    
    const stepX = range / heatW;
    const stepY = range / heatH;
    
    if ((modelo === "simple" && weights.w_simple) || (modelo === "multicapa" && weights.wjk)) {
        for (let iy = 0; iy < heatH; iy++) {
            const py = (range / 2) - (iy * stepY); 
            for (let ix = 0; ix < heatW; ix++) {
                const px = -(range / 2) + (ix * stepX);
                
                let val;
                if (modelo === "simple") {
                    val = signo(dot([px, py, 1], weights.w_simple));
                } else {
                    const h_j = matMul(toRowVector([px, py, 1]), weights.wjk)[0].map(v => g(v));
                    val = g(matMul(toRowVector([...h_j, 1]), weights.wij)[0][0]);
                }

                const idx = (iy * heatW + ix) * 4;
                if (val >= 0) {
                    data[idx] = 76; data[idx+1] = 175; data[idx+2] = 80;
                } else {
                    data[idx] = 244; data[idx+1] = 67; data[idx+2] = 54;
                }
                data[idx+3] = 100; // Opacidad
            }
        }
        offCtx.putImageData(imgData, 0, 0);
        // Dibujamos el heatmap centrado
        ctx.drawImage(offScreenCanvas, offsetX, offsetY, size, size);
    }

    // 3. VECTORES Y PUNTOS
    ctx.save();
    ctx.translate(width / 2, height / 2);
    
    // La escala debe coincidir con el 'range' usado para el heatmap (7)
    const scale = size / range; 
    ctx.scale(scale, -scale);

    // --- GRILLA (De -3 a 3) ---
    ctx.beginPath();
    ctx.strokeStyle = "rgba(0,0,0,0.15)";
    ctx.lineWidth = 0.02; 
    for (let i = -3; i <= 3; i++) {
        // Líneas verticales
        ctx.moveTo(i, -3); ctx.lineTo(i, 3);
        // Líneas horizontales
        ctx.moveTo(-3, i); ctx.lineTo(3, i);
    }
    ctx.stroke();

    // --- PUNTOS ---
    for (let i = 0; i < x.length; i++) {
        ctx.beginPath();
        ctx.arc(x[i][0], x[i][1], 0.15, 0, 2 * Math.PI);
        ctx.fillStyle = y[i] === 1 ? "#4CAF50" : "#F44336";
        ctx.fill();
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 0.04;
        ctx.stroke();
    }

    // --- MÁSCARA BLANCA (EL MARCO) ---
    // Esto tapa todo lo que sobra fuera de -3 a 3
    ctx.beginPath();
    // 1. Rectángulo gigante exterior (más grande que el canvas visible)
    ctx.rect(-50, -50, 100, 100); 
    // 2. Rectángulo interior (el área visible de 6x6)
    ctx.rect(-3, -3, 6, 6);
    
    ctx.fillStyle = "white";
    // 'evenodd' rellena el espacio entre el rect gigante y el pequeño
    ctx.fill("evenodd"); 


    ctx.restore();
}

async function toggleTraining() {
    if (trainButton.textContent === "Iniciar entrenamiento") {
        trainButton.textContent = "Parar entrenamiento";
        trainButton.style.backgroundColor = "#dc3545";
        controls.forEach(c => c.disabled = true);
        stopTrainingFlag = false;
        lastWeights = {};

        const tipo = document.getElementById("dataset").value;
        const eta = parseFloat(document.getElementById("eta").value);
        const cota = parseInt(document.getElementById("cota").value);
        const modelo = document.getElementById("modelo").value;

        actualizarSVG(modelo);
        resultDisplay.innerText = "Entrenando...";
        
        try {
            const data = datasets[tipo];
            let res;
            if (modelo === "simple") {
                res = await perceptronSimple(data.x, data.y, eta, cota);
                lastWeights = { w_simple: res.weights };
            } else {
                res = await perceptronMulticapa(data.x, data.y, eta, cota);
                lastWeights = { wjk: res.wjk, wij: res.wij };
            }
            graficar(data.x, data.y, lastWeights, modelo);
            resultDisplay.innerText = res.output_text;
        } catch (e) {
            console.error(e);
            resultDisplay.innerText = "Error: " + e.message;
        } finally {
            trainButton.textContent = "Iniciar entrenamiento";
            trainButton.style.backgroundColor = "#007bff";
            controls.forEach(c => c.disabled = false);
        }
    } else {
        stopTrainingFlag = true;
        resultDisplay.innerText = "Deteniendo...";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    resizeCanvas();
    trainButton.addEventListener('click', toggleTraining);
    actualizarSVG(document.getElementById("modelo").value);
    graficar(datasets.and.x, datasets.and.y, {}, "simple");
});

window.addEventListener('resize', () => {
    resizeCanvas();
    const mod = document.getElementById("modelo").value;
    const dat = document.getElementById("dataset").value;
    graficar(datasets[dat].x, datasets[dat].y, lastWeights, mod);
});

document.getElementById("modelo").addEventListener('change', (e) => {
    if (trainButton.textContent === "Iniciar entrenamiento") {
        lastWeights = {};
        actualizarSVG(e.target.value);
        resultDisplay.innerText = "Resultados aquí.";
        const dat = document.getElementById("dataset").value;
        graficar(datasets[dat].x, datasets[dat].y, {}, e.target.value);
    }
});

document.getElementById("dataset").addEventListener('change', (e) => {
    if (trainButton.textContent === "Iniciar entrenamiento") {
        const mod = document.getElementById("modelo").value;
        graficar(datasets[e.target.value].x, datasets[e.target.value].y, lastWeights, mod);
    }
});