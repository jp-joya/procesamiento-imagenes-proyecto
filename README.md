# Sistema Automático de Evaluación de Calidad de Imágenes de Productos

**Universidad Sergio Arboleda · Ciencias de la Computación**  
Juan Pablo Joya · Jefferson Gutierrez

---

## Descripción

Herramienta que evalúa automáticamente la calidad visual de fotografías de productos para e-commerce y redes sociales (Instagram). Orientada a microempresas y emprendimientos que gestionan su propio contenido sin conocimientos técnicos de fotografía.

El sistema analiza una imagen y entrega:
- Un **score de calidad** de 0 a 100
- Una **categoría**: Profesional, Aceptable o Deficiente
- **Recomendaciones concretas** para mejorar la imagen antes de publicarla

---

## Instalación

### Requisitos
- Python 3.10+

### Pasos

```bash
# Clonar o descargar el proyecto
cd procesamientoImagenes

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

```bash
streamlit run app.py
```

Se abre la aplicación en el navegador (`http://localhost:8501`). Sube una imagen JPG, PNG o WEBP y el sistema la analiza automáticamente.

---

## Métricas implementadas (v0.1 — técnicas clásicas)

| Métrica | Técnica | Peso en score |
|---|---|---|
| Nitidez | Varianza del Laplaciano | 30% |
| Iluminación | Luminancia media (escala de grises) | 25% |
| Contraste | Desviación estándar de píxeles | 20% |
| Uniformidad del fondo | Varianza en región perimetral (15%) | 15% |
| Exposición | Fracción de píxeles saturados | 10% |

### Clasificación

| Categoría | Score |
|---|---|
| 🟢 Profesional | ≥ 72 |
| 🟡 Aceptable | 45 – 71 |
| 🔴 Deficiente | < 45 |

---

## Estructura del proyecto

```
procesamientoImagenes/
├── app.py            # Interfaz web (Streamlit)
├── evaluator.py      # Lógica de análisis de calidad
├── requirements.txt  # Dependencias Python
└── README.md
```

---

## Roadmap

| Fase | Descripción | Estado |
|---|---|---|
| v0.1 | Métricas clásicas de visión + interfaz Streamlit | ✅ Completado |
| v0.2 | Recolección y etiquetado del dataset | ⏳ Pendiente |
| v0.3 | Preprocesamiento y extracción de características | ⏳ Pendiente |
| v0.4 | Entrenamiento CNN con transfer learning (MobileNet/ResNet) | ⏳ Pendiente |
| v0.5 | Integración CNN + métricas clásicas + evaluación final | ⏳ Pendiente |

---

## Referencias

1. dcx.lett.digital. *Fotos en el e-commerce: los pros y contras de producir tus propias imágenes.* 2020.
2. Tbaileh, I. & Bagriyanik, S. *Visual quality assessment of E-commerce product images using convolutional neural networks.* Multimedia Systems, 31(6), 2025. https://doi.org/10.1007/s00530-025-02009-8
