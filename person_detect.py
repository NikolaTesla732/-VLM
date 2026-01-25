import cv2
from ultralytics import YOLO

def person_detect(
    video_path: str,
    sample_every_n_frames: int = 3,  # 1 = проверять каждый кадр, 5 = каждый 5-й (быстрее)
    *,
    model_path: str = "yolov8s.pt",
    imgsz: int = 416,
    conf: float = 0.35,
    iou: float = 0.5,
    max_seconds: float | None = None,  # если задано — проверяем только первые N секунд
) -> bool:
    """
    Проверяет, есть ли человек (class 0 = person) на видео.
    Возвращает True, если хотя бы на одном проверенном кадре найден человек, иначе False.

    Параметры:
      - sample_every_n_frames: ускорение за счёт пропуска кадров
      - max_seconds: ограничение по времени проверки (например 5.0 = только первые 5 секунд)
    """

    # Загружаем модель
    model = YOLO(model_path)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    # Если нужно ограничить проверку по времени — считаем лимит кадров
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * fps))

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Ограничение по времени (по количеству кадров)
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Пропуск кадров для ускорения
        if sample_every_n_frames > 1 and (frame_idx % sample_every_n_frames) != 0:
            frame_idx += 1
            continue

        # Детекция (нам не нужен трекинг, просто проверка наличия человека)
        results = model.predict(
            frame,
            classes=[0],   # person
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

        # Если есть хотя бы один bbox — человек найден
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            cap.release()
            return True

        frame_idx += 1

    cap.release()
    return False


if __name__ == "__main__":
    print(person_detect(r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_004.mp4", 3))
