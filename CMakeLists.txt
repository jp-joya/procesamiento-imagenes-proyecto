cmake_minimum_required(VERSION 3.16)
project(EvaluadorCalidadImagenes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Buscar OpenCV ─────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Directorios de cabeceras ──────────────────────────────────────────────────
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ── Archivos fuente ───────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/sharpness.cpp
    src/illumination.cpp
    src/contrast.cpp
    src/background.cpp
    src/exposure.cpp
    src/scorer.cpp
    src/classifier.cpp
    src/recommendation.cpp
    src/utils.cpp
)

# ── Ejecutable ────────────────────────────────────────────────────────────────
add_executable(evaluador ${SOURCES})

target_link_libraries(evaluador ${OpenCV_LIBS})

# ── Mensaje de configuración ──────────────────────────────────────────────────
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Ejecutable: evaluador")