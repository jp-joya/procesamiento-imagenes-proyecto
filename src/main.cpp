#include "filters.h"
#include "scorer.h"
#include "classifier.h"
#include "utils.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: ./evaluador <ruta_imagen> [ruta_modelo.onnx]" << std::endl;
        return 1;
    }

    // ── Cargar imagen ─────────────────────────────────────────────────────────
    std::string imagePath = argv[1];
    cv::Mat image = loadImage(imagePath);
    if (image.empty()) return 1;

    cv::Mat imageResized = resizeImage(image, 800);
    cv::Mat gray = toGray(imageResized);

    // ── Calcular métricas clásicas ────────────────────────────────────────────
    std::vector<MetricResult> metrics = {
        computeSharpness(gray),
        computeIllumination(gray),
        computeContrast(gray),
        computeBackground(gray),
        computeExposure(gray)
    };

    // ── Score global ──────────────────────────────────────────────────────────
    EvaluationResult result = computeGlobalScore(metrics);

    // ── Mostrar resultados ────────────────────────────────────────────────────
    std::cout << std::endl;
    std::cout << "=== Evaluacion de Calidad de Imagen ===" << std::endl;
    std::cout << "Archivo: " << imagePath << std::endl;
    printSeparator();

    for (const auto& m : result.metrics) {
        printMetric(m.name, m.value, m.score, m.label, m.recommendation);
    }

    printSeparator();
    std::cout << "Score global: " << result.overallScore << "/100" << std::endl;
    std::cout << "Categoria:    " << result.category << std::endl;
    std::cout << "Resumen:      " << result.summary << std::endl;
    printSeparator();

    // ── Clasificación CNN (opcional, si se pasa el modelo) ────────────────────
    if (argc >= 3) {
        std::string modelPath = argv[2];
        ImageClassifier classifier;
        if (classifier.loadModel(modelPath)) {
            std::string cnnCategory = classifier.classify(imageResized);
            std::cout << "Categoria CNN: " << cnnCategory << std::endl;
            printSeparator();
        }
    }

    return 0;
}