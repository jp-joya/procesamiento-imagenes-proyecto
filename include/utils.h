#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// ── Carga y preprocesamiento ───────────────────────────────────────────────────

// Carga una imagen desde disco en formato BGR (OpenCV)
// Retorna una Mat vacía si la ruta no existe o no es una imagen válida
cv::Mat loadImage(const std::string& path);

// Convierte una imagen BGR a escala de grises
cv::Mat toGray(const cv::Mat& imageBGR);

// Redimensiona una imagen manteniendo la relación de aspecto
// targetSize: lado máximo en píxeles
cv::Mat resizeImage(const cv::Mat& image, int targetSize = 800);

// Normaliza los valores de píxel al rango [0, 1]
cv::Mat normalizeImage(const cv::Mat& image);

// ── Salida en consola ─────────────────────────────────────────────────────────

// Imprime una línea separadora
void printSeparator();

// Imprime el resultado de una métrica con formato
void printMetric(const std::string& name,
                 double             value,
                 double             score,
                 const std::string& label,
                 const std::string& recommendation);