import os
import subprocess
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)


BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EVALUADOR     = os.path.join(BASE_DIR, "..", "build", "evaluador")
MODELO        = os.path.join(BASE_DIR, "..", "models", "quality_model.onnx")
ALLOWED       = {"jpg", "jpeg", "png", "webp"}


app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def calibrar_score(score_bruto):
    return round(score_bruto, 1)


def determinar_estado(score_calibrado):
    if score_calibrado >= 85:
        return "Bueno"
    elif score_calibrado >= 60:
        return "Aceptable"
    else:
        return "Deficiente"


def determinar_categoria(score_calibrado):
    if score_calibrado >= 85:
        return "Profesional"
    elif score_calibrado >= 60:
        return "Aceptable"
    else:
        return "Deficiente"


def generar_resumen(categoria):
    if categoria == "Profesional":
        return "La imagen cumple con un nivel alto de calidad visual y puede usarse en publicaciones principales."
    elif categoria == "Aceptable":
        return "La imagen tiene calidad aceptable pero conviene mejorar algunos aspectos antes de publicarla."
    return "La imagen presenta problemas importantes de calidad y debería corregirse antes de publicarla."


def parse_output(output):
    result = {
        "metricas": [],
        "score_global": 0.0,
        "categoria": "",
        "resumen": "",
        "categoria_cnn": "",
        "recomendaciones": []
    }

    lines = output.strip().split("\n")
    i = 0
    current_metric = None

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Score global:"):
            try:
                score_raw_global = float(line.split(":")[1].replace("/100", "").strip())
                result["score_global"] = calibrar_score(score_raw_global)
                result["categoria"] = determinar_categoria(result["score_global"])
                result["resumen"] = generar_resumen(result["categoria"])
            except:
                pass

        elif line.startswith("Categoria CNN:"):
            result["categoria_cnn"] = line.split(":", 1)[1].strip()

        elif "valor:" in line and "score:" in line:
            try:
                parts = line.split()
                nombre    = parts[0]
                val_idx   = parts.index("valor:") + 1
                score_idx = parts.index("score:") + 1

                score_raw = float(parts[score_idx])
                score_calibrado = calibrar_score(score_raw)

                current_metric = {
                    "nombre": nombre,
                    "valor": float(parts[val_idx]),
                    "score": score_calibrado,
                    "estado": determinar_estado(score_calibrado),
                    "recomendacion": ""
                }
                result["metricas"].append(current_metric)
            except:
                pass

        elif line.startswith("->") and current_metric is not None:
            rec = line.lstrip("->").strip()
            current_metric["recomendacion"] = rec
            result["recomendaciones"].append({
                "metrica": current_metric["nombre"],
                "texto": rec
            })

        i += 1

    return result


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def generar_filtros(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return []

    h, w = img.shape[:2]
    max_dim = 480
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtros = []

    filtros.append({
        "nombre": "Original",
        "descripcion": "Imagen tal como se recibió, base del análisis.",
        "imagen": img_to_b64(img)
    })

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    filtros.append({
        "nombre": "Escala de grises",
        "descripcion": "Conversión usada para calcular nitidez y contraste.",
        "imagen": img_to_b64(gray_bgr)
    })

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.uint8(np.clip(np.abs(lap) * 4, 0, 255))
    lap_bgr = cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)
    filtros.append({
        "nombre": "Nitidez (Laplaciano)",
        "descripcion": "Varianza del Laplaciano. Mayor activación = mayor nitidez.",
        "imagen": img_to_b64(lap_bgr)
    })

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brillo = hsv[:, :, 2]
    brillo_color = cv2.applyColorMap(brillo, cv2.COLORMAP_HOT)
    filtros.append({
        "nombre": "Mapa de brillo",
        "descripcion": "Canal V del espacio HSV. Detecta zonas oscuras o sobreexpuestas.",
        "imagen": img_to_b64(brillo_color)
    })

    eq = cv2.equalizeHist(gray)
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    filtros.append({
        "nombre": "Contraste (ecualizado)",
        "descripcion": "Histograma ecualizado para visualizar el rango tonal disponible.",
        "imagen": img_to_b64(eq_bgr)
    })

    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, blurred)
    diff_color = cv2.applyColorMap(diff * 4, cv2.COLORMAP_COOL)
    filtros.append({
        "nombre": "Uniformidad del fondo",
        "descripcion": "Diferencia con versión difuminada. Zonas activas = fondo no uniforme.",
        "imagen": img_to_b64(diff_color)
    })

    _, quemado = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    _, oscuro  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    expo_vis = img.copy()
    expo_vis[quemado > 0] = [0, 220, 255]
    expo_vis[oscuro > 0] = [0, 0, 200]
    filtros.append({
        "nombre": "Exposición",
        "descripcion": "Amarillo = zonas quemadas. Rojo = zonas muy oscuras.",
        "imagen": img_to_b64(expo_vis)
    })

    return filtros


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analizar", methods=["POST"])
def analizar():
    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files["imagen"]
    if not file.filename or not allowed(file.filename):
        return jsonify({"error": "Formato no válido. Usa JPG, PNG o WEBP."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    try:
        proc = subprocess.run(
            [EVALUADOR, filepath, MODELO],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            return jsonify({"error": proc.stderr or "Error del evaluador"}), 500

        data = parse_output(proc.stdout)
        data["filtros"] = generar_filtros(filepath)
        return jsonify(data)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "El análisis tardó demasiado."}), 500
    except FileNotFoundError:
        return jsonify({"error": f"Ejecutable no encontrado: {EVALUADOR}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)