// --- utils.js ---

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

// Función auxiliar para redondear números
function round(value, decimals) {
    return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

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