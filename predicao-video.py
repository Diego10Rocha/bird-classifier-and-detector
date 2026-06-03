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

VIDEO_DIR = "Detectar passaro\\videos"

OUTPUT_DIR = "Detectar passaro\\resultados"


CONFIDENCE = 0.4

# Configurações SAHI
SLICE_HEIGHT = 300
SLICE_WIDTH = 300

OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# =====================================================
# CRIAR PASTAS
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_OUTPUT = os.path.join(OUTPUT_DIR, "imagens")
VIDEO_OUTPUT = os.path.join(OUTPUT_DIR, "videos")

os.makedirs(IMAGE_OUTPUT, exist_ok=True)
os.makedirs(VIDEO_OUTPUT, exist_ok=True)

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

video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

thraupis_sayaca=VIDEO_DIR + "Thraupis sayaca.mp4"
phacellodomus_rufifrons=VIDEO_DIR + "Phacellodomus rufifrons.mp4"
cathartes_burrovianus=VIDEO_DIR + "Cathartes burrovianus.mp4"

cyanocorax_cyanopogon=VIDEO_DIR + "Cyanocorax cyanopogon.mp4"
euphonia_chlorotica=VIDEO_DIR + "Euphonia chlorotica.mp4"
falco_sparverius=VIDEO_DIR + "Falco sparverius.mp4"

file = thraupis_sayaca

#for file in os.listdir(VIDEO_DIR):
ext = os.path.splitext(file)[1].lower()

#if ext not in video_extensions:
#    continue

video_path = os.path.join(VIDEO_DIR, file)

print(f"\nProcessando vídeo: {file}")

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = os.path.join(VIDEO_OUTPUT, file)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =====================================
    # SAHI NO FRAME
    # =====================================

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO
    )

    for obj in result.object_prediction_list:

        bbox = obj.bbox

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)

        class_name = obj.category.name
        confidence = obj.score.value

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

        # Salvar resultados
        results_data.append({
            "tipo": "video",
            "arquivo": file,
            "frame": frame_number,
            "classe": class_name,
            "confianca": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    out.write(frame)

    frame_number += 1

cap.release()
out.release()