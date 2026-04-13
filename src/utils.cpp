#include "utils.h"
#include <iostream>
#include <iomanip>

cv::Mat loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[ERROR] No se pudo cargar la imagen: " << path << std::endl;
    }
    return image;
}

cv::Mat toGray(const cv::Mat& imageBGR) {
    cv::Mat gray;
    cv::cvtColor(imageBGR, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat resizeImage(const cv::Mat& image, int targetSize) {
    cv::Mat resized;
    int maxDim = std::max(image.cols, image.rows);
    if (maxDim <= targetSize) return image.clone();
    double scale = static_cast<double>(targetSize) / maxDim;
    cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

cv::Mat normalizeImage(const cv::Mat& image) {
    cv::Mat norm;
    image.convertTo(norm, CV_64F, 1.0 / 255.0);
    return norm;
}

void printSeparator() {
    std::cout << std::string(50, '-') << std::endl;
}

void printMetric(const std::string& name, double value, double score,
                 const std::string& label, const std::string& recommendation) {
    std::cout << std::left << std::setw(22) << name
              << "valor: " << std::setw(8) << std::fixed << std::setprecision(2) << value
              << "score: " << std::setw(7) << score
              << "[" << label << "]" << std::endl;
    std::cout << "  -> " << recommendation << std::endl;
}