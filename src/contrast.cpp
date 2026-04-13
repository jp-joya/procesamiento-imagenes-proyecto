#include "filters.h"
#include <algorithm>

MetricResult computeContrast(const cv::Mat& gray) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    double stdDev = stddev[0];

    double score;
    std::string label, rec;

    if (stdDev >= CONTRAST_GOOD) {
        score = 100.0;
        label = "Bueno";
        rec   = "El contraste es adecuado.";
    } else if (stdDev >= CONTRAST_OK) {
        double pct = (stdDev - CONTRAST_OK) / (CONTRAST_GOOD - CONTRAST_OK);
        score = 50.0 + pct * 50.0;
        label = "Aceptable";
        rec   = "El contraste podria mejorarse. Usa un fondo de color mas contrastante al producto.";
    } else {
        score = std::max(0.0, (stdDev / CONTRAST_OK) * 50.0);
        label = "Deficiente";
        rec   = "Contraste muy bajo. El producto no se distingue del fondo. Usa un fondo de color opuesto.";
    }

    return { "Contraste", stdDev, score, label, rec };
}