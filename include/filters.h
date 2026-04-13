#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ── Estructura de resultado por métrica ───────────────────────────────────────
struct MetricResult {
    std::string name;
    double      value;           // Valor numérico calculado
    double      score;           // Score normalizado 0-100
    std::string label;           // "Bueno", "Aceptable", "Deficiente"
    std::string recommendation;  // Retroalimentación textual
};

// ── Umbrales de clasificación ─────────────────────────────────────────────────

// Nitidez (varianza del Laplaciano)
constexpr double SHARPNESS_GOOD = 120.0;
constexpr double SHARPNESS_OK   = 50.0;

// Iluminación (luminancia media 0-255)
constexpr double BRIGHTNESS_LOW          = 60.0;
constexpr double BRIGHTNESS_HIGH         = 200.0;
constexpr double BRIGHTNESS_IDEAL_LOW    = 90.0;
constexpr double BRIGHTNESS_IDEAL_HIGH   = 170.0;

// Contraste (desviación estándar de píxeles)
constexpr double CONTRAST_GOOD = 55.0;
constexpr double CONTRAST_OK   = 30.0;

// Uniformidad del fondo (desv. estándar en bordes)
constexpr double BG_UNIFORMITY_GOOD = 20.0;
constexpr double BG_UNIFORMITY_OK   = 40.0;

// Exposición (fracción máxima de píxeles saturados)
constexpr double SATURATION_MAX = 0.05;

// ── Declaraciones de funciones ────────────────────────────────────────────────
MetricResult computeSharpness(const cv::Mat& gray);
MetricResult computeIllumination(const cv::Mat& gray);
MetricResult computeContrast(const cv::Mat& gray);
MetricResult computeBackground(const cv::Mat& gray);
MetricResult computeExposure(const cv::Mat& gray);