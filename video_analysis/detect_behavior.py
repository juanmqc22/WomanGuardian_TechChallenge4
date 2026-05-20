import cv2
import numpy as np
from ultralytics import YOLO


# YOLOv8 nano — leve e suficiente para detecção de pessoas
model = YOLO("yolov8n.pt")

# Configurações de amostragem
FRAME_INTERVAL = 10       # analisa 1 frame a cada 10
MAX_FRAMES = 50           # máximo de frames analisados


def analyze_video(video_path: str) -> dict:
    """
    Analisa vídeo com YOLOv8 para detectar padrões corporais de risco.
    Indicadores: presença humana, variação de movimento, postura retraída.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Não foi possível abrir o vídeo"}

    frame_count = 0
    analyzed_frames = 0
    detections_per_frame = []
    confidence_scores = []
    movement_deltas = []
    prev_boxes = None

    while analyzed_frames < MAX_FRAMES:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_INTERVAL != 0:
            continue

        # Detecta apenas pessoas (class 0 no COCO)
        results = model(frame, classes=[0], verbose=False)

        boxes = results[0].boxes
        num_persons = len(boxes)
        detections_per_frame.append(num_persons)

        # Confiança média das detecções
        if num_persons > 0:
            confs = boxes.conf.cpu().numpy()
            confidence_scores.extend(confs.tolist())

            # Variação de posição entre frames (proxy de movimento)
            current_centers = _get_centers(boxes)

            if prev_boxes is not None and len(prev_boxes) > 0 and len(current_centers) > 0:
                delta = _movement_delta(prev_boxes, current_centers)
                movement_deltas.append(delta)

            prev_boxes = current_centers

        analyzed_frames += 1

    cap.release()

    if analyzed_frames == 0:
        return {"error": "Nenhum frame analisado"}

    return _classify_risk(
        detections_per_frame,
        confidence_scores,
        movement_deltas,
        analyzed_frames,
        frame_count
    )


def _get_centers(boxes) -> list:
    centers = []
    for box in boxes.xyxy.cpu().numpy():
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centers.append((cx, cy))
    return centers


def _movement_delta(prev: list, curr: list) -> float:
    """Calcula variação média de posição entre frames."""
    deltas = []
    for p, c in zip(prev[:len(curr)], curr[:len(prev)]):
        d = np.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)
        deltas.append(d)
    return float(np.mean(deltas)) if deltas else 0.0


def _classify_risk(
    detections: list,
    confidences: list,
    movements: list,
    analyzed_frames: int,
    total_frames: int
) -> dict:

    avg_persons = np.mean(detections) if detections else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    avg_movement = np.mean(movements) if movements else 0
    frames_with_person = sum(1 for d in detections if d > 0)
    presence_rate = frames_with_person / analyzed_frames if analyzed_frames > 0 else 0

    reasons = []

    # --- REGRAS DE RISCO ---

    # Presença humana muito baixa pode indicar colapso/queda
    if presence_rate < 0.3 and avg_confidence > 0.4:
        reasons.append("Presença humana intermitente (possível queda ou colapso)")

    # Movimento muito baixo pode indicar imobilidade/tensão
    if avg_movement < 15 and presence_rate > 0.5:
        reasons.append("Movimento corporal muito reduzido (postura retraída ou tensão)")

    # Múltiplas pessoas podem indicar situação de confronto
    if avg_persons > 1.5:
        reasons.append("Múltiplas pessoas detectadas em cena")

    # Alta variação de movimento pode indicar agitação
    if avg_movement > 80:
        reasons.append("Alta agitação de movimento detectada")

    # --- NÍVEL DE RISCO ---
    score = len(reasons)

    if score >= 2:
        visual_risk = "HIGH"
    elif score == 1:
        visual_risk = "MEDIUM"
    else:
        visual_risk = "LOW"

    return {
        "visualRisk": visual_risk,
        "confidence": float(round(avg_confidence, 2)),
        "framesAnalyzed": analyzed_frames,
        "totalFrames": total_frames,
        "avgPersonsDetected": float(round(avg_persons, 2)),
        "presenceRate": float(round(presence_rate, 2)),
        "avgMovement": float(round(avg_movement, 2)),
        "reasons": reasons
    }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    result = analyze_video("sample_video.mp4")
    print(result)