# Sistema Automático de Evaluación de Calidad de Imágenes de Productos

**Universidad Sergio Arboleda — Ciencias de la Computación**
Juan Pablo Joya · Jefferson Gutierrez

---

## Descripción
Sistema que evalúa y clasifica la calidad visual de fotografías de productos
para uso en plataformas de e-commerce y redes sociales (Instagram).

Categorías de clasificación: **Profesional**, **Aceptable**, **Deficiente**

---

## Requisitos
- CMake >= 3.16
- OpenCV >= 4.5
- C++17
- (Para entrenamiento CNN) Python 3.9+, PyTorch, ONNX

---

## Compilación
```bash
mkdir build && cd build
cmake ..
make
```

## Uso
```bash
# Solo métricas clásicas
./evaluador fotos/pro.jpg

# Con modelo CNN
./evaluador fotos/pro.jpg models/quality_model.onnx
```

---

## Métricas implementadas
| Métrica | Técnica | Peso |
|---|---|---|
| Nitidez | Varianza del Laplaciano | 30% |
| Iluminación | Luminancia media | 25% |
| Contraste | Desviación estándar | 20% |
| Uniformidad del fondo | Varianza perimetral | 15% |
| Exposición | % píxeles saturados | 10% |