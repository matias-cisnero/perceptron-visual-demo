const canvas = document.getElementById("grafico");
const ctx = canvas.getContext("2d");
const trainButton = document.getElementById("btn-entrenar");
const resultDisplay = document.getElementById("resultado");
const controls = document.querySelectorAll('.controls-container input, .controls-container select');

let stopTrainingFlag = false; // Bandera para detener el entrenamiento

// Función auxiliar para redondear números
function round(value, decimals) {
    return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

// Definición de los datasets
const datasets = {
    and: {
        x: [[-1, -1], [-1, 1], [1, -1], [1, 1]], // Entradas
        y: [-1, -1, -1, 1] // Salidas esperadas
    },
    xor: {
        x: [[-1, -1], [-1, 1], [1, -1], [1, 1]], // Entradas
        y: [-1, 1, 1, -1] // Salidas esperadas
    }
};

// --- Funciones Auxiliares para Álgebra Lineal ---
// Producto punto de dos vectores 1D
function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Los vectores deben tener la misma longitud para el producto punto.");
    }
    return a.reduce((acc, val, i) => acc + val * b[i], 0);
}

// Multiplicación de matrices (A @ B)
function matMul(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
        throw new Error(`Dimension mismatch for matrix multiplication: A(${rowsA}x${colsA}) vs B(${rowsB}x${colsB})`);
    }

    const result = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Transponer una matriz
function transpose(matrix) {
    if (matrix.length === 0 || matrix[0].length === 0) return [[]];
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

// Multiplicación escalar de una matriz
function scalarMul(scalar, matrix) {
    return matrix.map(row => row.map(val => scalar * val));
}

// Suma de matrices
function matAdd(A, B) {
    if (A.length !== B.length || A[0].length !== B[0].length) {
        throw new Error("Dimension mismatch for matrix addition.");
    }
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
}

// Convertir un array 1D a un vector fila 2D
function toRowVector(arr) {
    return [arr];
}

// Convertir un array 1D a un vector columna 2D
function toColumnVector(arr) {
    return arr.map(val => [val]);
}

// --- Funciones de Activación ---
function signo(h) {
    return h >= 0 ? 1 : -1;
}

function g(h, beta = 1) { // Función de activación tanh
    return Math.tanh(beta * h);
}

function g_derivada(h, beta = 1) { // Derivada de tanh
    const val_g = g(h, beta);
    return beta * (1 - val_g * val_g);
}

// --- Función de Error ---
function calcularError(x, y, modelo, weights_obj) {
    if (modelo === "simple") {
        const w = weights_obj.w_simple;
        let error = 0;
        for (let i = 0; i < x.length; i++) {
            const entrada = [...x[i], 1]; // Añadir sesgo
            const salida_predicha = signo(dot(entrada, w));
            error += Math.abs(y[i] - salida_predicha); // Error absoluto
        }
        return error;
    } else if (modelo === "multicapa") {
        const wjk = weights_obj.wjk; // Pesos entrada a oculta
        const wij = weights_obj.wij; // Pesos oculta a salida
        let total_squared_error = 0;
        const xExt = x.map(p => [...p, 1]); // Entradas con sesgo

        for (let i = 0; i < x.length; i++) {
            const Vk = toRowVector(xExt[i]); // Entrada actual

            // Propagación hacia adelante (con capa oculta tanh)
            const h_j_matrix = matMul(Vk, wjk);
            const Vj_activations = h_j_matrix[0].map(val => g(val)); // Activaciones capa oculta (tanh)
            const Vj_with_bias = toRowVector([...Vj_activations, 1]); // Salida capa oculta con sesgo

            const h_i_matrix = matMul(Vj_with_bias, wij);
            const Vi = g(h_i_matrix[0][0]); // Salida de la red (tanh)

            // Error cuadrático medio
            total_squared_error += 0.5 * Math.pow(y[i] - Vi, 2);
        }
        return total_squared_error;
    }
    return Infinity;
}

// --- Función para verificar clasificación perfecta ---
function checkAllClassifiedCorrectly(x, y, wjk, wij) {
    const num_hidden_neurons = wjk[0].length; // Obtener de los pesos para consistencia
    const xExt = x.map(p => [...p, 1]); // Entradas con sesgo

    for (let i = 0; i < x.length; i++) {
        const Vk = toRowVector(xExt[i]);

        // Propagación hacia adelante
        const h_j_matrix = matMul(Vk, wjk);
        const Vj_activations = h_j_matrix[0].map(val => g(val));
        const Vj_with_bias = toRowVector([...Vj_activations, 1]);

        const h_i_matrix = matMul(Vj_with_bias, wij);
        const Vi = g(h_i_matrix[0][0]); // Activación de salida final (tanh)

        const predicted_class = signo(Vi); // Aplicar signo para obtener -1 o 1

        if (predicted_class !== y[i]) {
            return false; // Se encontró un punto mal clasificado
        }
    }
    return true; // Todos los puntos clasificados correctamente
}

// --- Funciones de Entrenamiento del Perceptrón ---
async function perceptronSimple(x, y, eta, cota) {
    let w = [0, 0, 1]; // Pesos iniciales [w1, w2, bias]
    let error = Infinity, i = 0, errorMin = Infinity, wMin = [...w];
    const xExt = x.map(p => [...p, 1]); // Entradas con sesgo

    const inicio_tiempo = performance.now(); // Usar performance.now() para mayor precisión

    while (error > 0 && i < cota && !stopTrainingFlag) { // Añadir condición de parada
        const idx = Math.floor(Math.random() * x.length); // Índice aleatorio
        const salida = signo(dot(xExt[idx], w)); // Predicción

        // Actualización de pesos
        const delta = xExt[idx].map(val => eta * (y[idx] - salida) * val);
        w = w.map((wi, j) => wi + delta[j]);

        error = calcularError(x, y, "simple", { w_simple: w });
        if (error < errorMin) {
            errorMin = error;
            wMin = [...w];
        }

        graficar(x, y, { w_simple: w }, "simple");
        await new Promise(res => setTimeout(res, 50)); // Pequeña pausa para visualización
        i++;
    }

    const fin_tiempo = performance.now();
    const fin_total = round((fin_tiempo - inicio_tiempo) / 1000, 5);

    let output_messages = [];
    output_messages.push(`Épocas necesarias: ${i}`);
    output_messages.push(`Error final: ${errorMin}`);
    output_messages.push(`Tiempo total: ${fin_total} segundos.`);
    
    if (stopTrainingFlag) {
        output_messages.push("Entrenamiento detenido manualmente.");
    } else if (errorMin > 0) {
        output_messages.push("El perceptrón simple no pudo converger a error 0. Esto es esperado para problemas no linealmente separables (como XOR).");
    }

    return { weights: wMin, output_text: output_messages.join("\n") };
}

async function perceptronMulticapa(x, y, eta, cota) {
    const num_hidden_neurons = 2; // 2 neuronas en la capa oculta (la 3ra es el sesgo)
    const num_output_neurons = 1; // 1 neurona en la capa de salida

    const num_input_features = x[0].length; // 2 para AND/XOR
    const num_input_with_bias = num_input_features + 1; // 3 (x1, x2, sesgo)

    // Pesos de la capa de entrada (k) a la capa oculta (j)
    // wjk: (num_input_with_bias x num_hidden_neurons)
    let wjk = Array(num_input_with_bias).fill(0).map(() => 
                Array(num_hidden_neurons).fill(0).map(() => (Math.random() * 2) - 1));

    // Pesos de la capa oculta (j + sesgo) a la capa de salida (i)
    // wij: (num_hidden_neurons + 1 x num_output_neurons)
    let wij = Array(num_hidden_neurons + 1).fill(0).map(() => 
                Array(num_output_neurons).fill(0).map(() => (Math.random() * 2) - 1));

    let error_min = Infinity;
    let wjk_min = JSON.parse(JSON.stringify(wjk));
    let wij_min = JSON.parse(JSON.stringify(wij));

    const xExt = x.map(p => [...p, 1]); // Entradas con sesgo
    const y_scaled = y.map(val => [val]); // Asegurar que y sea 2D para operaciones matriciales

    const output_messages = [];
    const inicio_tiempo = performance.now();

    let actual_epochs_completed = 0; // Variable para rastrear las épocas completadas
    let perfectly_classified = false; // Bandera para clasificación perfecta

    for (let epoch_counter = 0; epoch_counter < cota && !stopTrainingFlag; epoch_counter++) { // Usar epoch_counter para el bucle
        actual_epochs_completed = epoch_counter + 1; // Actualizar el contador real
        const indices = Array.from({ length: x.length }, (_, i) => i).sort(() => Math.random() - 0.5); // Mezclar índices

        for (const idx of indices) {
            if (stopTrainingFlag) break; // Detener si la bandera es true
            const Vk = toRowVector(xExt[idx]); // Entrada actual con sesgo

            // --- Propagación hacia adelante (Forward Pass) ---
            const h_j_matrix = matMul(Vk, wjk); // Entrada ponderada a la capa oculta
            const h_j = h_j_matrix[0]; // Convertir a array 1D para g()

            const Vj_activations = h_j.map(val => g(val)); // Activaciones capa oculta (TANH)
            const Vj_with_bias = toRowVector([...Vj_activations, 1]); // Salida capa oculta con sesgo

            const h_i_matrix = matMul(Vj_with_bias, wij);
            const h_i = h_i_matrix[0][0]; // Convertir a escalar para g()

            const Vi = g(h_i); // Activación de la neurona de salida (TANH)

            // --- Retropropagación (Backward Pass) ---
            // Delta para la capa de salida
            const error_output = y_scaled[idx][0] - Vi; // Error escalar
            const delta_i_scalar = g_derivada(h_i) * error_output;
            const delta_i = toColumnVector([delta_i_scalar]); // Vector columna (1x1)

            // Delta para la capa oculta (excluyendo el peso del sesgo de wij para esta propagación)
            const wij_no_bias = wij.slice(0, num_hidden_neurons); // Eliminar la fila de sesgo de wij
            const delta_j_matrix = matMul(delta_i, transpose(wij_no_bias)); // (1x1) @ (1x2) = (1x2)
            const delta_j_activations_scalar = delta_j_matrix[0]; // Array 1D de escalares

            const delta_j = h_j.map((hj_val, j_idx) => g_derivada(hj_val) * delta_j_activations_scalar[j_idx]); // Derivada TANH
            const delta_j_col_vector = toColumnVector(delta_j); // Vector columna (2x1)

            // --- Actualización de Pesos ---
            // Delta para wij (oculta a salida)
            const Delta_wij_matrix = matMul(transpose(Vj_with_bias), delta_i);
            const Delta_wij_scaled = scalarMul(eta, Delta_wij_matrix);
            wij = matAdd(wij, Delta_wij_scaled);

            // Delta para wjk (entrada a oculta)
            const Delta_wjk_matrix = matMul(delta_j_col_vector, Vk);
            const Delta_wjk_scaled = scalarMul(eta, Delta_wjk_matrix);
            wjk = matAdd(wjk, transpose(Delta_wjk_scaled)); 
        }

        // Calcular error total después de cada época
        const current_error = calcularError(x, y, "multicapa", { wjk: wjk, wij: wij });

        if (current_error < error_min) {
            error_min = current_error;
            wij_min = JSON.parse(JSON.stringify(wij));
            wjk_min = JSON.parse(JSON.stringify(wjk));
        }

        // Verificar clasificación perfecta después de cada época
        if (checkAllClassifiedCorrectly(x, y, wjk, wij)) {
            perfectly_classified = true;
            // Asegurarse de que los pesos finales sean los que lograron la clasificación perfecta
            wij_min = JSON.parse(JSON.stringify(wij));
            wjk_min = JSON.parse(JSON.stringify(wjk));
            break; // Detener el entrenamiento
        }

        // Graficar el estado actual (puede ser lento si hay muchas épocas)
        graficar(x, y, { wjk: wjk, wij: wij }, "multicapa");
        await new Promise(res => setTimeout(res, 10)); // Pequeña pausa para visualización
    }

    const fin_tiempo = performance.now();
    const fin_total = round((fin_tiempo - inicio_tiempo) / 1000, 5);

    // Calcular el error final con los mejores pesos encontrados (o los de clasificación perfecta)
    const final_error_overall = calcularError(x, y, "multicapa", { wjk: wjk_min, wij: wij_min });

    output_messages.push(`Épocas necesarias: ${actual_epochs_completed}`); // Usar el contador real
    output_messages.push(`Error final: ${final_error_overall.toFixed(6)}`);
    output_messages.push(`Tiempo total: ${fin_total} segundos.`);
    
    if (perfectly_classified) {
        output_messages.push("Clasificación perfecta alcanzada.");
    } else if (stopTrainingFlag) {
        output_messages.push("Entrenamiento detenido manualmente.");
    }

    return { wjk: wjk_min, wij: wij_min, output_text: output_messages.join("\n") };
}

// --- Función de Graficación ---
function graficar(x, y, weights_obj, modelo) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2); // Centrar el origen
    ctx.scale(50, -50); // Escalar y voltear el eje Y

    // Dibujar la cuadrícula de fondo
    ctx.beginPath();
    ctx.strokeStyle = "#ccc";
    ctx.lineWidth = 0.01;
    for (let xg = -4; xg <= 4; xg++) {
        ctx.moveTo(xg, -4);
        ctx.lineTo(xg, 4);
    }
    for (let yg = -4; yg <= 4; yg++) {
        ctx.moveTo(-4, yg);
        ctx.lineTo(4, yg);
    }
    ctx.stroke();

    // Rango para la cuadrícula de decisión
    const resolution = 100;
    const x_range = [-2, 2];
    const y_range = [-2, 2];
    const step_x = (x_range[1] - x_range[0]) / resolution;
    const step_y = (y_range[1] - y_range[0]) / resolution;

    // Dibujar las zonas de clasificación (mapa de calor)
    if (modelo === "simple" && weights_obj.w_simple) {
        const w = weights_obj.w_simple;
        for (let ix = 0; ix < resolution; ix++) {
            for (let iy = 0; iy < resolution; iy++) {
                const px = x_range[0] + ix * step_x;
                const py = y_range[0] + iy * step_y;
                const input_point = [px, py, 1]; // Añadir sesgo
                const prediction = signo(dot(input_point, w));

                ctx.fillStyle = prediction >= 0 ? "rgba(76, 175, 80, 0.5)" : "rgba(244, 67, 54, 0.5)"; // Verde claro/Rojo claro
                ctx.fillRect(px, py, step_x, step_y);
            }
        }
    } else if (modelo === "multicapa" && weights_obj.wjk && weights_obj.wij) {
        const wjk = weights_obj.wjk;
        const wij = weights_obj.wij;

        for (let ix = 0; ix < resolution; ix++) {
            for (let iy = 0; iy < resolution; iy++) {
                const px = x_range[0] + ix * step_x;
                const py = y_range[0] + iy * step_y;

                // Realizar pasada hacia adelante para este punto de la cuadrícula
                const input_point = toRowVector([px, py, 1]); // Añadir sesgo

                const h_j_matrix = matMul(input_point, wjk);
                const Vj_activations = h_j_matrix[0].map(val => g(val)); // Activaciones capa oculta (tanh)
                const Vj_with_bias = toRowVector([...Vj_activations, 1]);

                const h_i_matrix = matMul(Vj_with_bias, wij);
                const Vi = g(h_i_matrix[0][0]); // Salida de la red (tanh)

                // Determinar el color de la región basado en la salida
                ctx.fillStyle = Vi >= 0 ? "rgba(76, 175, 80, 0.5)" : "rgba(244, 67, 54, 0.5)"; // Verde claro/Rojo claro
                ctx.fillRect(px, py, step_x, step_y);
            }
        }
    }
    ctx.setLineDash([]);
    ctx.restore();

    // Dibujar los puntos de datos (siempre encima de las regiones)
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2); // Centrar el origen
    ctx.scale(50, -50); // Escalar y voltear el eje Y
    for (let i = 0; i < x.length; i++) {
        ctx.beginPath();
        ctx.arc(x[i][0], x[i][1], 0.15, 0, 2 * Math.PI);
        ctx.fillStyle = y[i] === 1 ? "#4CAF50" : "#F44336"; // Verde para 1, Rojo para -1
        ctx.fill();
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 0.02;
        ctx.stroke();
    }
    ctx.restore();
}

// --- Actualizar Diagrama SVG del Perceptrón ---
function actualizarSVG(modelo) {
    const svg = document.getElementById("svg-diagram");
    svg.innerHTML = `
        <defs>
            <radialGradient id="inputRadial" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" style="stop-color:#e0f2f7;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#a7d9ed;stop-opacity:1" />
            </radialGradient>
            <radialGradient id="hiddenRadial" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" style="stop-color:#f3e5f5;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#d1c4e9;stop-opacity:1" />
            </radialGradient>
            <radialGradient id="outputRadial" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" style="stop-color:#fffde7;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#ffe082;stop-opacity:1" />
            </radialGradient>

            <filter id="drop-shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceAlpha" stdDeviation="1"/>
                <feOffset dx="1" dy="1" result="offsetblur"/>
                <feFlood flood-color="rgba(0,0,0,0.2)"/>
                <feComposite in2="offsetblur" operator="in"/>
                <feMerge>
                    <feMergeNode/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>

            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="0" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#777" /> </marker>
        </defs>
    `;

    if (modelo === "simple") {
        svg.innerHTML += `
            <line class="connection" x1="50" y1="50" x2="250" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="100" x2="250" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="160" x2="250" y2="100" marker-end="url(#arrowhead)" />

            <circle class="neuron" cx="50" cy="50" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="50" cy="100" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="50" cy="160" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" /> 

            <circle class="neuron" cx="250" cy="100" r="20" fill="url(#outputRadial)" style="filter:url(#drop-shadow);" />

            <text x="50" y="55" text-anchor="middle" class="neuron-label">x₁</text>
            <text x="50" y="105" text-anchor="middle" class="neuron-label">x₂</text>
            <text x="50" y="165" text-anchor="middle" class="neuron-label">1</text>
            <text x="250" y="105" text-anchor="middle" class="neuron-label">O</text>`;
    } else { // Multicapa
        svg.innerHTML += `
            <line class="connection" x1="50" y1="40" x2="150" y2="40" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="100" x2="150" y2="40" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="160" x2="150" y2="40" marker-end="url(#arrowhead)" />

            <line class="connection" x1="50" y1="40" x2="150" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="100" x2="150" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="160" x2="150" y2="100" marker-end="url(#arrowhead)" />

            <line class="connection" x1="50" y1="40" x2="150" y2="160" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="100" x2="150" y2="160" marker-end="url(#arrowhead)" />
            <line class="connection" x1="50" y1="160" x2="150" y2="160" marker-end="url(#arrowhead)" />

            <line class="connection" x1="150" y1="40" x2="250" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="150" y1="100" x2="250" y2="100" marker-end="url(#arrowhead)" />
            <line class="connection" x1="150" y1="160" x2="250" y2="100" marker-end="url(#arrowhead)" />
            
            <circle class="neuron" cx="50" cy="40" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="50" cy="100" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="50" cy="160" r="20" fill="url(#inputRadial)" style="filter:url(#drop-shadow);" />
            
            <circle class="neuron" cx="150" cy="40" r="20" fill="url(#hiddenRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="150" cy="100" r="20" fill="url(#hiddenRadial)" style="filter:url(#drop-shadow);" />
            <circle class="neuron" cx="150" cy="160" r="20" fill="url(#hiddenRadial)" style="filter:url(#drop-shadow);" />
            
            <circle class="neuron" cx="250" cy="100" r="20" fill="url(#outputRadial)" style="filter:url(#drop-shadow);" />
            
            <text x="50" y="45" text-anchor="middle" class="neuron-label">x₁</text>
            <text x="50" y="105" text-anchor="middle" class="neuron-label">x₂</text>
            <text x="50" y="165" text-anchor="middle" class="neuron-label">1</text>
            <text x="250" y="105" text-anchor="middle" class="neuron-label">O</text>
        `;
    }
}

// --- Controlador de Interfaz: Toggle Training ---
async function toggleTraining() {
    if (trainButton.innerText === "Detener") {
        stopTrainingFlag = true;
        trainButton.innerText = "Deteniendo...";
        trainButton.disabled = true;
        return;
    }

    const modelo = document.getElementById("modelo").value;
    const datasetKey = document.getElementById("dataset").value;
    const eta = parseFloat(document.getElementById("eta").value);
    const cota = parseInt(document.getElementById("cota").value);

    // Configurar estado inicial
    stopTrainingFlag = false;
    trainButton.innerText = "Detener";
    resultDisplay.innerText = "Entrenando...";
    
    // Actualizar el diagrama según la selección
    actualizarSVG(modelo);
    
    // Obtener datos
    const { x, y } = datasets[datasetKey];
    
    let result;
    
    // Ejecutar entrenamiento
    if (modelo === "simple") {
        result = await perceptronSimple(x, y, eta, cota);
    } else {
        result = await perceptronMulticapa(x, y, eta, cota);
    }

    // Mostrar resultados y restaurar botón
    resultDisplay.innerText = result.output_text;
    trainButton.innerText = "Iniciar entrenamiento";
    trainButton.disabled = false;
}

// Inicializar gráfico y diagrama al cargar
actualizarSVG(document.getElementById("modelo").value);
graficar(datasets.and.x, datasets.and.y, {}, document.getElementById("modelo").value);

// Listener para actualizar el diagrama cuando cambia el select
document.getElementById("modelo").addEventListener("change", (e) => {
    actualizarSVG(e.target.value);
    graficar(datasets[document.getElementById("dataset").value].x, datasets[document.getElementById("dataset").value].y, {}, e.target.value);
});

document.getElementById("dataset").addEventListener("change", (e) => {
    graficar(datasets[e.target.value].x, datasets[e.target.value].y, {}, document.getElementById("modelo").value);
});