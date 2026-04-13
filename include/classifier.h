#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

// ── Clasificador CNN basado en modelo ONNX ────────────────────────────────────
// Carga un modelo pre-entrenado (MobileNet / ResNet exportado desde Python)
// y clasifica una imagen en: "Profesional", "Aceptable", "Deficiente"
//
// NOTA: Esta clase requiere que el archivo models/quality_model.onnx exista.
//       Si el modelo no está cargado, classify() retorna "Sin modelo".

class ImageClassifier {
public:
    ImageClassifier();

    // Carga el modelo .onnx desde disco
    // Retorna true si se cargó correctamente
    bool loadModel(const std::string& modelPath);

    // Clasifica una imagen BGR (OpenCV) y retorna la categoría
    std::string classify(const cv::Mat& image);

    // Indica si el modelo fue cargado exitosamente
    bool isLoaded() const;

private:
    cv::dnn::Net net_;
    bool         modelLoaded_;

    static const int INPUT_SIZE = 224;  // MobileNet / ResNet esperan 224x224

    // Prepara la imagen para la red: resize + normalización ImageNet
    cv::Mat preprocess(const cv::Mat& image) const;
};