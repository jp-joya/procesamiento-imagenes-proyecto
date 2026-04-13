#include "scorer.h"
#include <iostream>
#include <vector>

// Imprime en consola todas las recomendaciones de los métricas deficientes o aceptables
void printRecommendations(const EvaluationResult& result) {
    std::vector<std::string> recs;
    for (const auto& m : result.metrics) {
        if (m.label == "Aceptable" || m.label == "Deficiente") {
            recs.push_back(m.recommendation);
        }
    }

    if (recs.empty()) {
        std::cout << "  ¡No hay recomendaciones! La imagen cumple todos los criterios." << std::endl;
    } else {
        for (const auto& r : recs) {
            std::cout << "  -> " << r << std::endl;
        }
    }
}