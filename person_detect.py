import cv2
from ultralytics import YOLO


def person_detect(
    video_path: str,
    n_frames: int,
    *,
    sample_every_n_frames: int = 3,  # проверяем каждый N-й кадр (ускорение)
    skip_first_n_frames: int = 2,    # сколько кадров подряд можно "пропустить" (нет человека), не обрывая серию
    model_path: str = "yolov8s.pt",
    imgsz: int = 416,
    conf: float = 0.35,
    iou: float = 0.5,
    max_seconds: float | None = None,
) -> bool:
    """
    True, если человек встречается на протяжении n_frames (по проверяемым кадрам),
    при этом допускаются "провалы" без человека длиной до skip_first_n_frames подряд,
    не обрывая серию.

    Важно:
      "подряд" и "пропуски" считаются по тем кадрам, которые реально проверяются
      (после учёта sample_every_n_frames).
    """

    if n_frames < 1:
        raise ValueError("n_frames должен быть >= 1")
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames должен быть >= 1")
    if skip_first_n_frames < 0:
        raise ValueError("skip_first_n_frames должен быть >= 0")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * fps))

    frame_idx = 0

    run = 0       # сколько "кадров с человеком" накопили в текущей серии
    gap = 0       # сколько подряд "кадров без человека" внутри серии

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # ускорение: обрабатываем только каждый sample_every_n_frames-й кадр
        if (frame_idx % sample_every_n_frames) != 0:
            frame_idx += 1
            continue

        results = model.predict(
            frame,
            classes=[0],   # person
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        boxes = results[0].boxes
        person_now = boxes is not None and len(boxes) > 0

        if person_now:
            run += 1
            gap = 0
        else:
            gap += 1
            if gap > skip_first_n_frames:
                # провал слишком длинный — серия обрывается
                run = 0
                gap = 0

        if run >= n_frames:
            cap.release()
            return True

        frame_idx += 1

    cap.release()
    return False
