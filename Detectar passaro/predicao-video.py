import cv2
import sys
import numpy as np
from ultralytics import YOLO

relative_path = "Detectar passaro\\"

#specie_file = "Falco sparverius.mp4"
#specie_file = "Euphonia chlorotica.mp4"
specie_file = "Cyanocorax Cyanopogon.mp4"

#specie_file = "Thraupis sayaca.mp4"
#specie_file = "Phacellodomus rufifrons.mp4"
#specie_file = "Cathartes burrovianus.mp4"

file_path = relative_path + "videos\\" + specie_file

img_size = 640
confidence = 0.4
modelorede = "yolo26m.pt"
modelo_bird = "best.pt"
confianca_bird = 0.5

# Load dos modelos
model = YOLO(modelorede)
model_bird = YOLO(modelo_bird)


# Removendo o parâmetro img_size e ajustando o código para lidar automaticamente com o tamanho do frame

def detectar_e_classificar_passaros(frame, model_det, model_clf, confianca_det, confianca_clf):
    """
    1) Roda o modelo de detecção no frame para detectar pássaros.
    2) Para cada pássaro detectado, recorta a bounding box.
    3) Classifica o recorte usando o modelo de classificação.
    4) Desenha a bounding box e a classe no frame original.
    Retorna o frame anotado.
    """
    # Predição: detectar pássaros
    results = model_det(frame, conf=confianca_det)

    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            # Verifica se a label é "bird"
            cls_id = int(box.cls[0])
            label = model_det.names[cls_id]

            if label.lower() == "bird":
                # Coordenadas da bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Garante que as coordenadas estão dentro dos limites
                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Recorta a região do pássaro
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Classifica o recorte usando o modelo de classificação
                results_clf = model_clf(crop, conf=confianca_clf)
                class_label = "Desconhecido"
                class_conf = 0.0

                for clf_result in results_clf:
                    for clf_box in clf_result.boxes:
                        class_id = int(clf_box.cls[0])
                        class_label = model_clf.names[class_id]
                        class_conf = clf_box.conf[0]

                # Desenha a bounding box e a classe no frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{class_label} {class_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame

# Verifica se é imagem ou vídeo
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

if file_path.lower().endswith(image_extensions):
    # Processar imagem
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"Erro ao carregar imagem: {file_path}")
        sys.exit(1)

    annotated_frame = detectar_e_classificar_passaros(
        frame, model, model_bird, confianca_bird, confidence
    )

    # output_image_path = "output_image.jpg"
    # cv2.imwrite(output_image_path, annotated_frame)
    # print(f"Imagem processada salva em: {output_image_path}")

    output_image_path = relative_path + "resultados\\images\\output_image.jpg"
    if cv2.imwrite(output_image_path, annotated_frame):
        print(f"Imagem processada salva em: {output_image_path}")
    else:
        print("Erro ao salvar a imagem")

    cv2.namedWindow("YOLO Inference", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("YOLO Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Processar vídeo
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {file_path}")
        sys.exit(1)

    # Configurar o vídeo de saída
    # output_video_path = "output_video.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Configurar o vídeo de saída para MP4
    output_video_path = relative_path + "resultados\\videos\\" + specie_file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Erro ao inicializar o VideoWriter")
        sys.exit(1)

    cv2.namedWindow("YOLO Inference", cv2.WINDOW_AUTOSIZE)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = detectar_e_classificar_passaros(
            frame, model, model_bird, confianca_bird, confidence
        )

        out.write(annotated_frame)

        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo processado salvo em: {output_video_path}")
