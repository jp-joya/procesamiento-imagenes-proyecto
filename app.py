"""
Sistema Automático de Evaluación de Calidad de Imágenes de Productos
Versión Preliminar — Demo

Universidad Sergio Arboleda
Juan Pablo Joya · Jefferson Gutierrez
"""

import io
import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from evaluator import evaluate_image, load_image_from_bytes, CATEGORY_LABELS

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluador de Calidad de Imágenes",
    page_icon="📸",
    layout="wide",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 5px solid #ccc;
}
.metric-card.bueno    { border-left-color: #2ecc71; }
.metric-card.aceptable{ border-left-color: #f39c12; }
.metric-card.deficiente{ border-left-color: #e74c3c; }
.rec-text { color: #555; font-size: 0.9em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 72:
        return "#2ecc71"
    if score >= 45:
        return "#f39c12"
    return "#e74c3c"


def label_css_class(label: str) -> str:
    return label.lower().replace("é", "e").replace("í", "i")


def plot_histogram(image_bgr: np.ndarray) -> plt.Figure:
    """Histograma RGB de la imagen."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    colors = ("b", "g", "r")
    channel_labels = ("Azul", "Verde", "Rojo")
    for i, (col, lbl) in enumerate(zip(colors, channel_labels)):
        hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, label=lbl, linewidth=1.2, alpha=0.8)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Valor de píxel", fontsize=9)
    ax.set_ylabel("Frecuencia", fontsize=9)
    ax.set_title("Distribución de color (RGB)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_laplacian(image_bgr: np.ndarray) -> plt.Figure:
    """Mapa de bordes (Laplaciano) para visualizar zonas de nitidez."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagen original", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(lap_abs, cmap="hot")
    axes[1].set_title("Mapa de nitidez (Laplaciano)", fontsize=9)
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def gauge_chart(score: float) -> plt.Figure:
    """Gráfico tipo velocímetro para el score global."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)
    ax.axis("off")

    # Arco de fondo (rojo → amarillo → verde)
    for start, end, color in [(180, 252, "#e74c3c"),
                               (108, 180, "#f39c12"),
                               (0,   108, "#2ecc71")]:
        theta = np.linspace(np.radians(start), np.radians(end), 60)
        ax.plot(np.cos(theta), np.sin(theta), lw=14, color=color,
                solid_capstyle="butt", alpha=0.35)

    # Aguja
    angle = np.radians(180 - score * 1.8)
    ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#2c3e50", zorder=5))

    # Texto
    ax.text(0, -0.2, f"{score:.0f}/100", ha="center", va="center",
            fontsize=18, fontweight="bold", color=score_color(score))
    fig.tight_layout(pad=0)
    return fig


# ── Interfaz principal ────────────────────────────────────────────────────────

def main():
    # Encabezado
    st.markdown("## 📸 Evaluador de Calidad de Imágenes de Productos")
    st.markdown(
        "Sube una fotografía de tu producto y el sistema analizará su calidad visual "
        "con retroalimentación automática para publicaciones en e-commerce y redes sociales."
    )
    st.divider()

    uploaded = st.file_uploader(
        "Sube tu imagen (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Sube una imagen para comenzar el análisis.")
        return

    # Cargar imagen
    raw_bytes = uploaded.read()
    image_bgr = load_image_from_bytes(raw_bytes)

    if image_bgr is None:
        st.error("No se pudo leer la imagen. Intenta con otro archivo.")
        return

    # Evaluar
    with st.spinner("Analizando imagen..."):
        result = evaluate_image(image_bgr)

    # ── Layout: izquierda imagen + gauge | derecha métricas ──
    col_img, col_metrics = st.columns([1, 1], gap="large")

    with col_img:
        # Imagen original
        pil_img = Image.open(io.BytesIO(raw_bytes))
        st.image(pil_img, caption="Imagen analizada", use_container_width=True)

        # Velocímetro
        st.pyplot(gauge_chart(result.overall_score), use_container_width=False)

        # Categoría y resumen
        cat_emoji = CATEGORY_LABELS.get(result.category, result.category)
        cat_color = score_color(result.overall_score)
        st.markdown(
            f"<h3 style='color:{cat_color}; text-align:center;'>{cat_emoji}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; color:#555;'>{result.summary}</p>",
            unsafe_allow_html=True,
        )

    with col_metrics:
        st.markdown("### Análisis por métrica")
        for m in result.metrics:
            css = label_css_class(m.label)
            label_color = {"bueno": "#2ecc71", "aceptable": "#f39c12", "deficiente": "#e74c3c"}.get(css, "#aaa")
            st.markdown(f"""
<div class="metric-card {css}">
  <b>{m.name}</b>
  &nbsp;&nbsp;<span style="color:{label_color}; font-weight:bold;">{m.label}</span>
  &nbsp;—&nbsp;<span style="color:#888; font-size:0.85em;">valor: {m.value}</span>
  <div style="background:#e0e0e0; border-radius:6px; height:10px; margin:6px 0;">
    <div style="background:{label_color}; width:{m.score}%; height:10px; border-radius:6px;"></div>
  </div>
  <p class="rec-text">💡 {m.recommendation}</p>
</div>
""", unsafe_allow_html=True)

    # ── Visualizaciones extra ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Visualizaciones técnicas")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.pyplot(plot_histogram(image_bgr), use_container_width=True)

    with vcol2:
        st.pyplot(plot_laplacian(image_bgr), use_container_width=True)

    # ── Resumen numérico ───────────────────────────────────────────────────────
    with st.expander("Ver resumen numérico de métricas"):
        rows = [[m.name, m.value, f"{m.score:.1f}/100", m.label] for m in result.metrics]
        rows.append(["**SCORE GLOBAL**", "—", f"**{result.overall_score}/100**", f"**{result.category}**"])
        st.table(
            {
                "Métrica":  [r[0] for r in rows],
                "Valor":    [r[1] for r in rows],
                "Score":    [r[2] for r in rows],
                "Nivel":    [r[3] for r in rows],
            }
        )

    st.caption(
        "Sistema Automático de Evaluación de Calidad de Imágenes · "
        "Universidad Sergio Arboleda · Juan Pablo Joya & Jefferson Gutierrez · "
        "Versión preliminar — técnicas clásicas de visión por computador"
    )


if __name__ == "__main__":
    main()
