# video_io.py
from typing import List, Tuple, Optional
import numpy as np

def _try_import_decord():
    try:
        import decord
        from decord import VideoReader
        return decord, VideoReader
    except Exception as e:
        raise ImportError(
            "Не найден decord. Установи: pip install decord\n"
            f"Текущая ошибка: {e}"
        )

def read_video_metadata(video_path: str, fps_hint: Optional[float] = None) -> dict:
    _, VideoReader = _try_import_decord()
    vr = VideoReader(video_path)
    num_frames = len(vr)

    # decord иногда даёт fps через get_avg_fps()
    fps = None
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = fps_hint

    return {"num_frames": num_frames, "fps": fps}

def read_frames(video_path: str, frame_indices: List[int]) -> np.ndarray:
    """
    Returns: frames uint8 array [T, H, W, 3] in RGB
    """
    _, VideoReader = _try_import_decord()
    vr = VideoReader(video_path)

    # clip indices to valid range
    max_idx = len(vr) - 1
    safe_idx = [min(max(i, 0), max_idx) for i in frame_indices]

    frames = vr.get_batch(safe_idx).asnumpy()  # [T, H, W, 3], uint8, RGB
    return frames

def sample_indices_uniform(start: int, end: int, num: int) -> List[int]:
    """
    Uniformly sample `num` indices in [start, end) (end exclusive).
    """
    if end <= start:
        return [start] * num
    if num == 1:
        return [start]
    lin = np.linspace(start, end - 1, num=num)
    return [int(round(x)) for x in lin]

def build_clips(frame_count: int, clip_len: int, stride: int) -> List[Tuple[int, int]]:
    """
    Возвращает список клипов как (start_frame, end_frame_exclusive).
    """
    clips = []
    start = 0
    while start < frame_count:
        end = start + clip_len
        if end > frame_count:
            # baseline: добиваем последний клип до длины clip_len повтором кадров (позже в семплинге)
            end = frame_count
        clips.append((start, end))
        if start + stride >= frame_count:
            break
        start += stride
    return clips
