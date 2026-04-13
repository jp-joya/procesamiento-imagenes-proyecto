#include "filters.h"
#include <algorithm>

MetricResult computeExposure(const cv::Mat& gray) {
    double total       = static_cast<double>(gray.total());
    double overexposed = cv::countNonZero(gray >= 250) / total;
    double underexposed = cv::countNonZero(gray <= 5)  / total;
    double satFrac     = overexposed + underexposed;

    double score;
    std::string label, rec;

    if (satFrac <= SATURATION_MAX) {
        score = 100.0;
        label = "Bueno";
        rec   = "No hay zonas quemadas ni areas completamente oscuras.";
    } else if (satFrac <= SATURATION_MAX * 3) {
        double pct = 1.0 - (satFrac - SATURATION_MAX) / (SATURATION_MAX * 2);
        score = 50.0 + std::max(0.0, pct) * 50.0;
        label = "Aceptable";
        rec   = "Hay algunas zonas quemadas o muy oscuras. Ajusta la exposicion para recuperar detalle.";
    } else {
        score = std::max(0.0, 50.0 * (1.0 - satFrac));
        label = "Deficiente";
        rec   = "Muchas zonas quemadas o negras. La exposicion esta muy mal calibrada.";
    }

    return { "Exposicion", satFrac * 100.0, score, label, rec };
}