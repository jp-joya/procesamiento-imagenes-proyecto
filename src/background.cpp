#include "filters.h"
#include <algorithm>

MetricResult computeBackground(const cv::Mat& gray) {
    int h = gray.rows;
    int w = gray.cols;
    int border = std::max(1, static_cast<int>(std::min(h, w) * 0.15));

    cv::Mat top    = gray(cv::Rect(0,        0,           w,      border));
    cv::Mat bottom = gray(cv::Rect(0,        h - border,  w,      border));
    cv::Mat left   = gray(cv::Rect(0,        0,           border, h));
    cv::Mat right  = gray(cv::Rect(w-border, 0,           border, h));

    std::vector<uchar> pixels;
    pixels.insert(pixels.end(), top.begin<uchar>(),    top.end<uchar>());
    pixels.insert(pixels.end(), bottom.begin<uchar>(), bottom.end<uchar>());
    pixels.insert(pixels.end(), left.begin<uchar>(),   left.end<uchar>());
    pixels.insert(pixels.end(), right.begin<uchar>(),  right.end<uchar>());

    cv::Mat borderMat(pixels);
    cv::Scalar mean, stddev;
    cv::meanStdDev(borderMat, mean, stddev);
    double stdBg = stddev[0];

    double score;
    std::string label, rec;

    if (stdBg <= BG_UNIFORMITY_GOOD) {
        score = 100.0;
        label = "Bueno";
        rec   = "El fondo es uniforme y limpio.";
    } else if (stdBg <= BG_UNIFORMITY_OK) {
        double pct = 1.0 - (stdBg - BG_UNIFORMITY_GOOD) / (BG_UNIFORMITY_OK - BG_UNIFORMITY_GOOD);
        score = 50.0 + pct * 50.0;
        label = "Aceptable";
        rec   = "El fondo tiene algunas variaciones. Considera usar un fondo liso de un solo color.";
    } else {
        score = std::max(0.0, (BG_UNIFORMITY_OK / stdBg) * 50.0);
        label = "Deficiente";
        rec   = "El fondo es muy heterogeneo. Usa un fondo blanco, negro o de color solido.";
    }

    return { "Uniformidad del Fondo", stdBg, score, label, rec };
}