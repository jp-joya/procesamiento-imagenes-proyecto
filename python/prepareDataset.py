"""
prepareDataset.py
-----------------
Descarga imágenes de productos reales desde URLs públicas (sin registro),
las evalúa con los mismos filtros del proyecto y genera versiones
degradadas para crear un dataset balanceado de 3 clases.

Estructura de salida:
    data/labeled/profesional/   → imágenes originales de alta calidad
    data/labeled/aceptable/     → versión con degradación leve
    data/labeled/deficiente/    → versión con degradación fuerte

Uso:
    python python/prepareDataset.py
    python python/prepareDataset.py --raw-dir data/raw   (si ya tienes fotos)
    python python/prepareDataset.py --count 200          (más imágenes)
"""

import os
import cv2
import numpy as np
import urllib.request
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--raw-dir",  default="data/raw",     help="Carpeta con imágenes ya descargadas")
parser.add_argument("--out-dir",  default="data/labeled", help="Carpeta de salida con clases")
parser.add_argument("--count",    type=int, default=300,  help="Número de imágenes a descargar")
parser.add_argument("--min-score",type=float, default=65.0, help="Score mínimo para considerar imagen como buena")
args = parser.parse_args()

# ── Umbrales (mismos que filters.h) ───────────────────────────────────────────
SHARPNESS_GOOD       = 120.0
SHARPNESS_OK         = 50.0
BRIGHTNESS_LOW       = 60.0
BRIGHTNESS_HIGH      = 200.0
BRIGHTNESS_IDEAL_LOW = 90.0
BRIGHTNESS_IDEAL_HIGH= 170.0
CONTRAST_GOOD        = 55.0
CONTRAST_OK          = 30.0
BG_UNIFORMITY_GOOD   = 20.0
BG_UNIFORMITY_OK     = 40.0
SATURATION_MAX       = 0.05

WEIGHTS = {
    "sharpness":    0.30,
    "illumination": 0.25,
    "contrast":     0.20,
    "background":   0.15,
    "exposure":     0.10,
}

# ── URLs de productos reales (Picsum con seeds fijas = imágenes reproducibles) ─
# Picsum entrega fotografías reales variadas; con seeds distintos obtenemos
# productos, personas, objetos cotidianos — suficiente variedad para el modelo.
def build_urls(count: int) -> list:
    base = "https://picsum.photos/seed/{seed}/600/600"
    return [base.format(seed=i) for i in range(1, count + 1)]

# ── Funciones de score (espejo de los .cpp) ───────────────────────────────────
def score_sharpness(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = float(lap.var())
    if v >= SHARPNESS_GOOD:   return 100.0
    if v >= SHARPNESS_OK:     return 50.0 + (v - SHARPNESS_OK) / (SHARPNESS_GOOD - SHARPNESS_OK) * 50.0
    return max(0.0, v / SHARPNESS_OK * 50.0)

def score_illumination(gray):
    m = float(cv2.mean(gray)[0])
    if BRIGHTNESS_IDEAL_LOW <= m <= BRIGHTNESS_IDEAL_HIGH: return 100.0
    if BRIGHTNESS_LOW <= m < BRIGHTNESS_IDEAL_LOW:
        return 50.0 + (m - BRIGHTNESS_LOW) / (BRIGHTNESS_IDEAL_LOW - BRIGHTNESS_LOW) * 50.0
    if BRIGHTNESS_IDEAL_HIGH < m <= BRIGHTNESS_HIGH:
        return 50.0 + (BRIGHTNESS_HIGH - m) / (BRIGHTNESS_HIGH - BRIGHTNESS_IDEAL_HIGH) * 50.0
    if m < BRIGHTNESS_LOW:   return max(0.0, m / BRIGHTNESS_LOW * 50.0)
    return max(0.0, BRIGHTNESS_HIGH / m * 40.0)

def score_contrast(gray):
    v = float(gray.astype(np.float32).std())
    if v >= CONTRAST_GOOD: return 100.0
    if v >= CONTRAST_OK:   return 50.0 + (v - CONTRAST_OK) / (CONTRAST_GOOD - CONTRAST_OK) * 50.0
    return max(0.0, v / CONTRAST_OK * 50.0)

def score_background(gray):
    h, w = gray.shape
    border = max(1, int(min(h, w) * 0.15))
    regions = [
        gray[0:border, :],
        gray[h-border:h, :],
        gray[:, 0:border],
        gray[:, w-border:w],
    ]
    pixels = np.concatenate([r.flatten() for r in regions])
    std = float(np.std(pixels.astype(np.float32)))
    if std <= BG_UNIFORMITY_GOOD: return 100.0
    if std <= BG_UNIFORMITY_OK:
        pct = 1.0 - (std - BG_UNIFORMITY_GOOD) / (BG_UNIFORMITY_OK - BG_UNIFORMITY_GOOD)
        return 50.0 + pct * 50.0
    return max(0.0, BG_UNIFORMITY_OK / std * 50.0)

def score_exposure(gray):
    total = gray.size
    over  = np.count_nonzero(gray >= 250) / total
    under = np.count_nonzero(gray <= 5)   / total
    sat   = over + under
    if sat <= SATURATION_MAX:           return 100.0
    if sat <= SATURATION_MAX * 3:
        pct = 1.0 - (sat - SATURATION_MAX) / (SATURATION_MAX * 2)
        return 50.0 + max(0.0, pct) * 50.0
    return max(0.0, 50.0 * (1.0 - sat))

def compute_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scores = {
        "sharpness":    score_sharpness(gray),
        "illumination": score_illumination(gray),
        "contrast":     score_contrast(gray),
        "background":   score_background(gray),
        "exposure":     score_exposure(gray),
    }
    return sum(WEIGHTS[k] * v for k, v in scores.items())

# ── Degradaciones ─────────────────────────────────────────────────────────────
def degrade_aceptable(bgr):
    """Desenfoque leve + oscurecer un poco + leve ruido"""
    k = random.choice([7, 9, 11])
    out = cv2.GaussianBlur(bgr, (k, k), 0)
    factor = random.uniform(0.75, 0.88)
    out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    noise = np.random.randint(-12, 12, out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

def degrade_deficiente(bgr):
    """Desenfoque fuerte + efecto aleatorio (muy oscuro, sobreexpuesto o fondo ruidoso)"""
    k = random.choice([21, 25, 31])
    out = cv2.GaussianBlur(bgr, (k, k), 0)
    choice = random.randint(0, 2)
    if choice == 0:   # muy oscuro
        out = np.clip(out.astype(np.float32) * random.uniform(0.2, 0.4), 0, 255).astype(np.uint8)
    elif choice == 1: # sobreexpuesto
        out = np.clip(out.astype(np.float32) * random.uniform(2.2, 3.0) + 60, 0, 255).astype(np.uint8)
    else:             # fondo caótico (ruido en bordes)
        h, w = out.shape[:2]
        b = int(min(h, w) * 0.18)
        noise = np.random.randint(0, 256, out.shape, dtype=np.uint8)
        mask = np.zeros((h, w, 3), dtype=bool)
        mask[:b, :] = True; mask[h-b:, :] = True
        mask[:, :b] = True; mask[:, w-b:] = True
        out = np.where(mask, noise, out)
    return out

# ── Descarga de imagen ────────────────────────────────────────────────────────
def download_image(url: str, timeout: int = 10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            arr = np.frombuffer(r.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        return None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    for cls in ["profesional", "aceptable", "deficiente"]:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    # Recopilar imágenes: primero las que ya hay en raw/, luego descargar
    images = []

    existing = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg"))
    if existing:
        print(f"[INFO] Usando {len(existing)} imágenes existentes en {raw_dir}/")
        for p in existing:
            img = cv2.imread(str(p))
            if img is not None:
                images.append((p.stem, img))

    needed = args.count - len(images)
    if needed > 0:
        print(f"[INFO] Descargando {needed} imágenes desde Picsum...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        urls = build_urls(needed + 50)  # extra por si algunas fallan
        downloaded = 0
        for i, url in enumerate(tqdm(urls, desc="Descargando")):
            if downloaded >= needed:
                break
            img = download_image(url)
            if img is None:
                continue
            fname = raw_dir / f"img_{i:04d}.jpg"
            cv2.imwrite(str(fname), img)
            images.append((f"img_{i:04d}", img))
            downloaded += 1

    print(f"\n[INFO] Total imágenes disponibles: {len(images)}")

    # Evaluar y generar variantes
    counters = {"profesional": 0, "aceptable": 0, "deficiente": 0}
    skipped = 0

    print("[INFO] Generando dataset balanceado...")
    for name, bgr in tqdm(images, desc="Procesando"):
        s = compute_score(bgr)

        if s < args.min_score:
            skipped += 1
            continue  # foto original ya es mala, no la usamos como base

        # Original → profesional
        dst = out_dir / "profesional" / f"{name}_orig.jpg"
        cv2.imwrite(str(dst), bgr)
        counters["profesional"] += 1

        # Degradación leve → aceptable
        dst = out_dir / "aceptable" / f"{name}_aceptable.jpg"
        cv2.imwrite(str(dst), degrade_aceptable(bgr))
        counters["aceptable"] += 1

        # Degradación fuerte → deficiente
        dst = out_dir / "deficiente" / f"{name}_deficiente.jpg"
        cv2.imwrite(str(dst), degrade_deficiente(bgr))
        counters["deficiente"] += 1

    print(f"\n{'='*45}")
    print(f"  Dataset generado en: {out_dir}/")
    print(f"{'='*45}")
    print(f"  Profesional : {counters['profesional']:>4} imágenes")
    print(f"  Aceptable   : {counters['aceptable']:>4} imágenes")
    print(f"  Deficiente  : {counters['deficiente']:>4} imágenes")
    print(f"  Descartadas : {skipped:>4} (score < {args.min_score})")
    print(f"  TOTAL       : {sum(counters.values()):>4} imágenes")
    print(f"{'='*45}")
    print("\n✅ Listo. Ahora ejecuta:")
    print("   python python/trainCNN.py")

if __name__ == "__main__":
    main()