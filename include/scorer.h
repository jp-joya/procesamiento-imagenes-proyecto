#pragma once
#include "filters.h"
#include <vector>
#include <string>

// ── Resultado global de evaluación ───────────────────────────────────────────
struct EvaluationResult {
    std::vector<MetricResult> metrics;
    double      overallScore;   // Score global ponderado 0-100
    std::string category;       // "Profesional", "Aceptable", "Deficiente"
    std::string summary;        // Texto resumen para el usuario
};

// ── Pesos de cada métrica (deben sumar 1.0) ───────────────────────────────────
// Nitidez:      0.30
// Iluminación:  0.25
// Contraste:    0.20
// Fondo:        0.15
// Exposición:   0.10

// ── Umbrales de categoría ─────────────────────────────────────────────────────
constexpr double SCORE_PROFESIONAL = 72.0;
constexpr double SCORE_ACEPTABLE   = 45.0;

// ── Declaración ───────────────────────────────────────────────────────────────
EvaluationResult computeGlobalScore(const std::vector<MetricResult>& metrics);