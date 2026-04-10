"""
Evaluador de Calidad de Imágenes de Productos
Sistema Automático de Evaluación de Calidad - Versión Preliminar (Técnicas Clásicas)

Métricas implementadas:
- Nitidez: Varianza del Laplaciano
- Iluminación: Análisis de luminancia (espacio LAB)
- Contraste: Desviación estándar de valores de píxel
- Uniformidad del fondo: Varianza en regiones periféricas
- Sobreexposición / Subexposición: Porcentaje de píxeles saturados
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ── Umbrales de clasificación ──────────────────────────────────────────────────
SHARPNESS_GOOD   = 120.0   # varianza del Laplaciano
SHARPNESS_OK     = 50.0

BRIGHTNESS_LOW   = 60.0    # luminancia media (0-255)
BRIGHTNESS_HIGH  = 200.0
BRIGHTNESS_IDEAL_LOW  = 90.0
BRIGHTNESS_IDEAL_HIGH = 170.0

CONTRAST_GOOD    = 55.0    # desviación estándar
CONTRAST_OK      = 30.0

BG_UNIFORMITY_GOOD = 20.0  # desv. estándar en bordes (menor = más uniforme)
BG_UNIFORMITY_OK   = 40.0

SATURATION_MAX   = 0.05    # fracción máxima de píxeles saturados aceptable


@dataclass
class MetricResult:
    name: str
    value: float
    score: float          # 0-100
    label: str            # "Bueno", "Aceptable", "Deficiente"
    recommendation: str   # texto de retroalimentación


@dataclass
class EvaluationResult:
    metrics: list = field(default_factory=list)
    overall_score: float = 0.0
    category: str = ""    # "Profesional", "Aceptable", "Deficiente"
    summary: str = ""


# ── Funciones de métricas individuales ────────────────────────────────────────

def compute_sharpness(gray: np.ndarray) -> MetricResult:
    """Varianza del Laplaciano: mayor valor = imagen más nítida."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if lap_var >= SHARPNESS_GOOD:
        score, label, rec = 100.0, "Bueno", "La nitidez es excelente."
    elif lap_var >= SHARPNESS_OK:
        pct = (lap_var - SHARPNESS_OK) / (SHARPNESS_GOOD - SHARPNESS_OK)
        score = 50.0 + pct * 50.0
        label = "Aceptable"
        rec = "La imagen podría ser un poco más nítida. Usa un trípode o aumenta la velocidad de obturación."
    else:
        score = max(0.0, (lap_var / SHARPNESS_OK) * 50.0)
        label = "Deficiente"
        rec = "La imagen está desenfocada. Asegúrate de enfocar bien el producto y evitar movimiento durante la captura."

    return MetricResult("Nitidez", round(lap_var, 2), round(score, 1), label, rec)


def compute_brightness(gray: np.ndarray) -> MetricResult:
    """Luminancia media en escala de grises."""
    mean_brightness = float(gray.mean())

    if BRIGHTNESS_IDEAL_LOW <= mean_brightness <= BRIGHTNESS_IDEAL_HIGH:
        score, label, rec = 100.0, "Bueno", "La iluminación es adecuada."
    elif BRIGHTNESS_LOW <= mean_brightness < BRIGHTNESS_IDEAL_LOW:
        pct = (mean_brightness - BRIGHTNESS_LOW) / (BRIGHTNESS_IDEAL_LOW - BRIGHTNESS_LOW)
        score = 50.0 + pct * 50.0
        label = "Aceptable"
        rec = "La imagen está un poco oscura. Añade más luz natural o artificial al producto."
    elif BRIGHTNESS_IDEAL_HIGH < mean_brightness <= BRIGHTNESS_HIGH:
        pct = (BRIGHTNESS_HIGH - mean_brightness) / (BRIGHTNESS_HIGH - BRIGHTNESS_IDEAL_HIGH)
        score = 50.0 + pct * 50.0
        label = "Aceptable"
        rec = "La imagen está un poco sobreexpuesta. Reduce la intensidad de la fuente de luz."
    elif mean_brightness < BRIGHTNESS_LOW:
        score = max(0.0, (mean_brightness / BRIGHTNESS_LOW) * 50.0)
        label = "Deficiente"
        rec = "La imagen está muy oscura. Mejora significativamente la iluminación del producto."
    else:
        score = max(0.0, (BRIGHTNESS_HIGH / mean_brightness) * 40.0)
        label = "Deficiente"
        rec = "La imagen está muy sobreexpuesta. Aleja las fuentes de luz o usa un difusor."

    return MetricResult("Iluminación", round(mean_brightness, 2), round(score, 1), label, rec)


def compute_contrast(gray: np.ndarray) -> MetricResult:
    """Desviación estándar de los valores de píxel."""
    std_dev = float(gray.std())

    if std_dev >= CONTRAST_GOOD:
        score, label, rec = 100.0, "Bueno", "El contraste es adecuado."
    elif std_dev >= CONTRAST_OK:
        pct = (std_dev - CONTRAST_OK) / (CONTRAST_GOOD - CONTRAST_OK)
        score = 50.0 + pct * 50.0
        label = "Aceptable"
        rec = "El contraste podría mejorarse. Intenta con un fondo de color más contrastante al producto."
    else:
        score = max(0.0, (std_dev / CONTRAST_OK) * 50.0)
        label = "Deficiente"
        rec = "Contraste muy bajo. El producto no se distingue bien del fondo. Usa un fondo de color opuesto."

    return MetricResult("Contraste", round(std_dev, 2), round(score, 1), label, rec)


def compute_background_uniformity(gray: np.ndarray) -> MetricResult:
    """
    Analiza la uniformidad del fondo tomando una franja perimetral del 15%
    del ancho/alto de la imagen.
    """
    h, w = gray.shape
    border = max(1, int(min(h, w) * 0.15))

    top    = gray[:border, :]
    bottom = gray[h - border:, :]
    left   = gray[:, :border]
    right  = gray[:, w - border:]
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(),
                                    left.flatten(), right.flatten()])
    std_bg = float(border_pixels.std())

    if std_bg <= BG_UNIFORMITY_GOOD:
        score, label, rec = 100.0, "Bueno", "El fondo es uniforme y limpio."
    elif std_bg <= BG_UNIFORMITY_OK:
        pct = 1.0 - (std_bg - BG_UNIFORMITY_GOOD) / (BG_UNIFORMITY_OK - BG_UNIFORMITY_GOOD)
        score = 50.0 + pct * 50.0
        label = "Aceptable"
        rec = "El fondo tiene algunas variaciones. Considera usar un fondo liso de un solo color."
    else:
        score = max(0.0, (BG_UNIFORMITY_OK / std_bg) * 50.0)
        label = "Deficiente"
        rec = "El fondo es muy heterogéneo y distrae del producto. Usa un fondo blanco, negro o de color sólido."

    return MetricResult("Uniformidad del Fondo", round(std_bg, 2), round(score, 1), label, rec)


def compute_overexposure(gray: np.ndarray) -> MetricResult:
    """Fracción de píxeles saturados (muy oscuros o muy claros)."""
    total = gray.size
    overexposed  = float(np.sum(gray >= 250)) / total
    underexposed = float(np.sum(gray <= 5))   / total
    sat_frac = overexposed + underexposed

    if sat_frac <= SATURATION_MAX:
        score, label, rec = 100.0, "Bueno", "No hay zonas quemadas ni áreas completamente oscuras."
    elif sat_frac <= SATURATION_MAX * 3:
        pct = 1.0 - (sat_frac - SATURATION_MAX) / (SATURATION_MAX * 2)
        score = 50.0 + max(0.0, pct) * 50.0
        label = "Aceptable"
        rec = "Hay algunas zonas quemadas o muy oscuras. Ajusta la exposición para recuperar detalle."
    else:
        score = max(0.0, 50.0 * (1 - sat_frac))
        label = "Deficiente"
        rec = "Muchas zonas quemadas o completamente negras. La exposición está muy mal calibrada."

    return MetricResult(
        "Exposición",
        round(sat_frac * 100, 2),
        round(score, 1),
        label,
        rec
    )


# ── Función principal ──────────────────────────────────────────────────────────

# Pesos de cada métrica en el score global
WEIGHTS = {
    "Nitidez":               0.30,
    "Iluminación":           0.25,
    "Contraste":             0.20,
    "Uniformidad del Fondo": 0.15,
    "Exposición":            0.10,
}

CATEGORY_LABELS = {
    "Profesional": "🟢 Profesional",
    "Aceptable":   "🟡 Aceptable",
    "Deficiente":  "🔴 Deficiente",
}


def evaluate_image(image_bgr: np.ndarray) -> EvaluationResult:
    """
    Recibe una imagen en formato BGR (OpenCV) y devuelve un EvaluationResult
    con todas las métricas, score global y categoría.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    metrics = [
        compute_sharpness(gray),
        compute_brightness(gray),
        compute_contrast(gray),
        compute_background_uniformity(gray),
        compute_overexposure(gray),
    ]

    # Score global ponderado
    overall = sum(WEIGHTS[m.name] * m.score for m in metrics)

    if overall >= 72:
        category = "Profesional"
        summary = (
            "La imagen tiene calidad profesional. "
            "Es adecuada para publicar en e-commerce o redes sociales."
        )
    elif overall >= 45:
        category = "Aceptable"
        summary = (
            "La imagen tiene calidad aceptable pero puede mejorarse. "
            "Revisa las recomendaciones para optimizarla antes de publicar."
        )
    else:
        category = "Deficiente"
        summary = (
            "La imagen no cumple los estándares mínimos de calidad. "
            "Se recomienda retomar la fotografía aplicando las correcciones indicadas."
        )

    return EvaluationResult(
        metrics=metrics,
        overall_score=round(overall, 1),
        category=category,
        summary=summary,
    )


def load_image(path: str) -> np.ndarray | None:
    """Carga una imagen desde disco. Retorna None si falla."""
    img = cv2.imread(path)
    return img


def load_image_from_bytes(data: bytes) -> np.ndarray | None:
    """Carga una imagen desde bytes (útil para uploads en Streamlit)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
