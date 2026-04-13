#include "filters.h"
#include <algorithm>

MetricResult computeSharpness(const cv::Mat& gray) {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double lapVar = stddev[0] * stddev[0];

    double score;
    std::string label, rec;

    if (lapVar >= SHARPNESS_GOOD) {
        score = 100.0;
        label = "Bueno";
        rec   = "La nitidez es excelente.";
    } else if (lapVar >= SHARPNESS_OK) {
        double pct = (lapVar - SHARPNESS_OK) / (SHARPNESS_GOOD - SHARPNESS_OK);
        score = 50.0 + pct * 50.0;
        label = "Aceptable";
        rec   = "La imagen podria ser mas nitida. Usa un tripode o aumenta la velocidad de obturacion.";
    } else {
        score = std::max(0.0, (lapVar / SHARPNESS_OK) * 50.0);
        label = "Deficiente";
        rec   = "La imagen esta desenfocada. Enfoca bien el producto y evita movimiento durante la captura.";
    }

    return { "Nitidez", lapVar, score, label, rec };
}