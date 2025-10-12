# 🔠 Perceptron Visual

Este proyecto es una herramienta interactiva para explorar el concepto de **separabilidad lineal** y visualizar la arquitectura y comportamiento de un **perceptrón simple** y un **perceptrón multicapa** (MLP).

* 🌐 Visualización en vivo: [GitHub Pages](https://matias-cisnero.github.io/perceptron-visual-demo/)
* 📁 Repositorio: [matias-cisnero/perceptron-visual](https://github.com/matias-cisnero/perceptron-visual)

![Demo](demo-perceptron.gif)

---

## 🔍 ¿Qué hace este proyecto?

* Permite entender la **separabilidad lineal** de conjuntos de datos clásicos como `AND` y `XOR`.
* Visualiza los puntos en 2D y el proceso de separación mediante:

  * Un **perceptrón simple** (una capa, activación escalón)
  * Un **perceptrón multicapa** (una capa oculta, activación `tanh`)
* Muestra la **arquitectura del modelo** de forma interactiva.
* Permite al usuario modificar:

  * El **dataset** (`AND` o `XOR`)
  * El **modelo** (simple o multicapa)
  * La **tasa de aprendizaje** (`η`)
  * La **cantidad máxima de iteraciones** de entrenamiento

---

## ⚙️ Tecnologías utilizadas

* [JavaScript](https://developer.mozilla.org/es/docs/Web/JavaScript)
* [HTML/CSS](https://developer.mozilla.org/es/docs/Web/HTML)
* [GitHub Pages](https://pages.github.com/) para despliegue
* Visualización con `<canvas>` para mostrar los puntos y la arquitectura

---

## 🚀 ¿Cómo lo uso?

### 🌐 Frontend interactivo

Accedé directamente desde:

🔗 [https://matias-cisnero.github.io/perceptron-visual/](https://matias-cisnero.github.io/perceptron-visual-demo/)

1. Elegí el dataset: `AND` o `XOR`
2. Seleccioná el modelo: perceptrón simple o multicapa
3. Ajustá la tasa de aprendizaje y las iteraciones
4. Observá:

   * Cómo cambia la arquitectura del modelo
   * Si puede o no separar los puntos en el plano

---

## 📁 Estructura del proyecto

```
perceptron-visual/
├── index.html            <- Interfaz principal
```

> El entrenamiento del modelo se realiza completamente en el navegador, sin librerías externas.

---

## 🧠 Conceptos clave

* **Separabilidad lineal**: propiedad de un dataset que puede ser separado por una línea recta.
* **Perceptrón simple**: modelo lineal de una capa.
* **Perceptrón multicapa (MLP)**: red neuronal con al menos una capa oculta, usando `tanh` como activación.
* **Tasa de aprendizaje** (`η`): controla el tamaño de los pasos de ajuste en el entrenamiento.
* **Iteraciones**: cantidad de veces que se actualizan los pesos.

---

## ✨ Créditos

Creado por [Matías Cisnero](https://github.com/matias-cisnero) para experimentar de forma visual con perceptrones y redes neuronales simples.

---

## 📌 Licencia

Este proyecto está bajo la licencia MIT.
