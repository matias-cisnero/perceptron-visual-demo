// --- diagram.js ---

// --- Actualizar Diagrama SVG del Perceptrón ---
function actualizarSVG(modelo) {
    const svg = document.getElementById("svg-diagram");
    // Definiciones de gradientes y filtros
    const defs = `
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

    svg.innerHTML = defs; 

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
    } else { // Multicapa (3 neuronas ocultas en el diagrama)
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
            <text x="150" y="45" text-anchor="middle" class="neuron-label">h₁</text>
            <text x="150" y="105" text-anchor="middle" class="neuron-label">h₂</text>
            <text x="150" y="165" text-anchor="middle" class="neuron-label">h₃</text>
            <text x="250" y="105" text-anchor="middle" class="neuron-label">O</text>`;
    }
}