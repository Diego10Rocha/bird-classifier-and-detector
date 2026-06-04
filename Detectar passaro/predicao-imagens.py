from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

import os
import cv2
import pandas as pd


# ==========================================
# CONFIGURAÇÕES
# ==========================================

MODEL_PATH = "Detectar passaro\\best.pt"

IMAGE_DIR = "Detectar passaro\\images\\test"

OUTPUT_DIR = "Detectar passaro\\resultados"


CONFIDENCE = 0.4

# Configurações SAHI
SLICE_HEIGHT = 640
SLICE_WIDTH = 640

OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# =====================================================
# CRIAR PASTAS
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_OUTPUT = os.path.join(OUTPUT_DIR, "imagens")

os.makedirs(IMAGE_OUTPUT, exist_ok=True)

# =====================================================
# CARREGAR MODELO YOLO + SAHI
# =====================================================

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device="cuda"
)

# =====================================================
# DATAFRAME RESULTADOS
# =====================================================

results_data = []

# =====================================================
# PROCESSAMENTO DE IMAGENS
# =====================================================

image_extensions = [".jpg", ".jpeg", ".png"]

for file in os.listdir(IMAGE_DIR):

    ext = os.path.splitext(file)[1].lower()

    if ext not in image_extensions:
        continue

    img_path = os.path.join(IMAGE_DIR, file)

    print(f"\nProcessando imagem: {file}")

    # =========================================
    # SAHI PREDICTION
    # =========================================

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO
    )

    image = cv2.imread(img_path)

    # =========================================
    # DESENHAR DETECÇÕES
    # =========================================

    for obj in result.object_prediction_list:

        bbox = obj.bbox

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)

        class_name = obj.category.name
        confidence = obj.score.value

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

        # Salvar resultados
        results_data.append({
            "tipo": "imagem",
            "arquivo": file,
            "classe": class_name,
            "confianca": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    output_path = os.path.join(IMAGE_OUTPUT, file)

    cv2.imwrite(output_path, image)

# =====================================================
# SALVAR CSV
# =====================================================

df = pd.DataFrame(results_data)

csv_path = os.path.join(OUTPUT_DIR, "deteccoes_sahi.csv")

df.to_csv(csv_path, index=False)

print("\nCSV salvo em:")
print(csv_path)

# =====================================================
# MÉTRICAS
# =====================================================

print("\n==============================")
print("MÉTRICAS")
print("==============================")

detections_per_class = df["classe"].value_counts()

print("\nDetecções por classe:")
print(detections_per_class)

mean_confidence = df.groupby("classe")["confianca"].mean()

print("\nConfiança média por classe:")
print(mean_confidence)