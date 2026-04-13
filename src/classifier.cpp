#include "classifier.h"
#include <iostream>

ImageClassifier::ImageClassifier() : modelLoaded_(false) {}

bool ImageClassifier::loadModel(const std::string& modelPath) {
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
        modelLoaded_ = true;
        std::cout << "[CNN] Modelo cargado: " << modelPath << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[CNN] Error al cargar modelo: " << e.what() << std::endl;
        modelLoaded_ = false;
    }
    return modelLoaded_;
}

std::string ImageClassifier::classify(const cv::Mat& image) {
    if (!modelLoaded_) {
        return "Sin modelo CNN";
    }

    cv::Mat blob = preprocess(image);
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    // output shape: [1, 3] — clases: 0=Deficiente, 1=Aceptable, 2=Profesional
    cv::Point classIdPoint;
    cv::minMaxLoc(output.reshape(1, 1), nullptr, nullptr, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    switch (classId) {
        case 0: return "Deficiente";
        case 1: return "Aceptable";
        case 2: return "Profesional";
        default: return "Desconocido";
    }
}

bool ImageClassifier::isLoaded() const {
    return modelLoaded_;
}

cv::Mat ImageClassifier::preprocess(const cv::Mat& image) const {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    // Normalización ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    cv::Mat blob = cv::dnn::blobFromImage(
        resized,
        1.0 / 255.0,
        cv::Size(INPUT_SIZE, INPUT_SIZE),
        cv::Scalar(0.485, 0.456, 0.406),
        true,   // swapRB (BGR -> RGB)
        false
    );
    return blob;
}