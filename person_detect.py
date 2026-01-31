import cv2
from ultralytics import YOLO
from typing import Optional, Tuple


def _clip_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError(f"Некорректный ROI (w/h <= 0): {roi}")

    x = max(0, x)
    y = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    w2 = x2 - x
    h2 = y2 - y
    if w2 <= 0 or h2 <= 0:
        raise ValueError(f"ROI вне кадра после обрезки: {roi} при размере кадра {(frame_w, frame_h)}")

    return x, y, w2, h2


def _apply_roi(frame, roi: Optional[Tuple[int, int, int, int]]):
    if roi is None:
        return frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x, y, rw, rh = _clip_roi(roi, w, h)
    return frame[y:y + rh, x:x + rw]


def person_detect(
    video_path: str,
    n_frames: int,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    sample_every_n_frames: int = 3,
    skip_first_n_frames: int = 2,
    model_path: str = "yolo11s.pt",
    imgsz: int = 416,
    conf: float = 0.35,
    iou: float = 0.5,
    max_seconds: float | None = None,
    device: str | int | None = None,
) -> bool:
    """
    Детект человека (COCO class=0) на выбранных кадрах.

    ROI:
      - если roi задан, детекция выполняется ТОЛЬКО внутри roi=(x,y,w,h).
      - ROI НЕ меняет видео на диске (только анализ).
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
    run = 0
    gap = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            if (frame_idx % sample_every_n_frames) != 0:
                frame_idx += 1
                continue

            frame_roi = _apply_roi(frame, roi)

            results = model.predict(
                frame_roi,
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
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
                    run = 0
                    gap = 0

            if run >= n_frames:
                return True

            frame_idx += 1

        return False
    finally:
        cap.release()
