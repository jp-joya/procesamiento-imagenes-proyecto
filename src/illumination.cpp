#include "filters.h"
#include <algorithm>

MetricResult computeIllumination(const cv::Mat& gray) {
    double meanBrightness = cv::mean(gray)[0];

    double score;
    std::string label, rec;

    if (meanBrightness >= BRIGHTNESS_IDEAL_LOW && meanBrightness <= BRIGHTNESS_IDEAL_HIGH) {
        score = 100.0;
        label = "Bueno";
        rec   = "La iluminacion es adecuada.";
    } else if (meanBrightness >= BRIGHTNESS_LOW && meanBrightness < BRIGHTNESS_IDEAL_LOW) {
        double pct = (meanBrightness - BRIGHTNESS_LOW) / (BRIGHTNESS_IDEAL_LOW - BRIGHTNESS_LOW);
        score = 50.0 + pct * 50.0;
        label = "Aceptable";
        rec   = "La imagen esta un poco oscura. Añade mas luz natural o artificial.";
    } else if (meanBrightness > BRIGHTNESS_IDEAL_HIGH && meanBrightness <= BRIGHTNESS_HIGH) {
        double pct = (BRIGHTNESS_HIGH - meanBrightness) / (BRIGHTNESS_HIGH - BRIGHTNESS_IDEAL_HIGH);
        score = 50.0 + pct * 50.0;
        label = "Aceptable";
        rec   = "La imagen esta un poco sobreexpuesta. Reduce la intensidad de la fuente de luz.";
    } else if (meanBrightness < BRIGHTNESS_LOW) {
        score = std::max(0.0, (meanBrightness / BRIGHTNESS_LOW) * 50.0);
        label = "Deficiente";
        rec   = "La imagen esta muy oscura. Mejora significativamente la iluminacion.";
    } else {
        score = std::max(0.0, (BRIGHTNESS_HIGH / meanBrightness) * 40.0);
        label = "Deficiente";
        rec   = "La imagen esta muy sobreexpuesta. Aleja las fuentes de luz o usa un difusor.";
    }

    return { "Iluminacion", meanBrightness, score, label, rec };
}