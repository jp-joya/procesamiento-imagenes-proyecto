#include "scorer.h"
#include <map>

EvaluationResult computeGlobalScore(const std::vector<MetricResult>& metrics) {
    // Pesos por métrica (deben sumar 1.0)
    const std::map<std::string, double> weights = {
        { "Nitidez",               0.30 },
        { "Iluminacion",           0.25 },
        { "Contraste",             0.20 },
        { "Uniformidad del Fondo", 0.15 },
        { "Exposicion",            0.10 }
    };

    double overall = 0.0;
    for (const auto& m : metrics) {
        auto it = weights.find(m.name);
        if (it != weights.end()) {
            overall += it->second * m.score;
        }
    }

    std::string category, summary;

    if (overall >= SCORE_PROFESIONAL) {
        category = "Profesional";
        summary  = "La imagen tiene calidad profesional. Es adecuada para publicar en e-commerce o redes sociales.";
    } else if (overall >= SCORE_ACEPTABLE) {
        category = "Aceptable";
        summary  = "La imagen tiene calidad aceptable pero puede mejorarse. Revisa las recomendaciones antes de publicar.";
    } else {
        category = "Deficiente";
        summary  = "La imagen no cumple los estandares minimos. Se recomienda retomar la fotografia aplicando las correcciones indicadas.";
    }

    return { metrics, overall, category, summary };
}