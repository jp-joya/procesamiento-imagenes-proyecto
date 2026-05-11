#include "filters.h"
#include <algorithm>

MetricResult computeSharpness(const cv::Mat& gray) {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double lapVar = stddev[0] * stddev[0];

    // Penalización por oscuridad: si la imagen es oscura,
    // el ruido nocturno infla el Laplaciano falsamente.
    // Reducimos el lapVar proporcionalmente al brillo real.
    cv::Scalar brightMean = cv::mean(gray);
    double brightness = brightMean[0]; // 0-255

    // Una imagen oscura (< 80) tiene su lapVar penalizado fuertemente
    // Una imagen bien iluminada (>= 120) no se toca
    if (brightness < 120.0) {
        double penalty = brightness / 120.0; // entre 0.0 y 1.0
        lapVar *= (penalty * penalty);        // penalización cuadrática
    }

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