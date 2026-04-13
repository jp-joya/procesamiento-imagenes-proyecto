"""
Entrenamiento de CNN para clasificación de calidad de imágenes de productos.
Transfer learning con MobileNetV2 (PyTorch).
Salida: models/quality_model.onnx
Uso: python python/trainCNN.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_DIR    = "data/labeled"
MODEL_OUT   = "models/quality_model.pt"
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_CLASSES = 3   # profesional, aceptable, deficiente

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ── Transformaciones ──────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

# 80% entrenamiento, 20% validación
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_set.dataset.transform = val_transforms

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Clases: {full_dataset.classes}")
print(f"Total imágenes: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")

# ── Modelo: MobileNetV2 con transfer learning ─────────────────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Congelar capas base, solo entrenar el clasificador final
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(device)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Fase entrenamiento
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_acc = 100.0 * correct / total

    # Fase validación
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100.0 * val_correct / val_total
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Guardar el mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  -> Modelo guardado (val_acc: {val_acc:.2f}%)")

print(f"\nEntrenamiento completo. Mejor val_acc: {best_val_acc:.2f}%")
print(f"Modelo guardado en: {MODEL_OUT}")
print("Ejecuta exportModel.py para convertir a ONNX.")