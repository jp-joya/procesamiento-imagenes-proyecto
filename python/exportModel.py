"""
Exporta el modelo entrenado (.pt) a formato ONNX
para ser usado desde classifier.cpp con cv::dnn.
Uso: python python/exportModel.py
"""

import torch
import torch.nn as nn
from torchvision import models

MODEL_PT   = "models/quality_model.pt"
MODEL_ONNX = "models/quality_model.onnx"
NUM_CLASSES = 3

# ── Cargar arquitectura y pesos ───────────────────────────────────────────────
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
model.eval()

# ── Exportar a ONNX ───────────────────────────────────────────────────────────
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_ONNX,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11
)

print(f"Modelo exportado exitosamente: {MODEL_ONNX}")
print("Ahora puedes usar: ./evaluador foto.jpg models/quality_model.onnx")