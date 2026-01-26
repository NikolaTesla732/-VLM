# visualize_results.py
from __future__ import annotations

import time
from typing import Optional

import matplotlib.pyplot as plt

from video_io import read_frames


def _fmt_time(t: Optional[float]) -> str:
    return "?" if t is None else f"{t:.2f}"


def show_top_segments(
    video_path: str,
    results_list: list[dict],
    max_clips: int = 1,
    delay: float = 0.2,
    *,
    max_frames_per_clip: Optional[int] = 200,
) -> None:
    """
    Показывает top-сегменты в ОКНЕ matplotlib (подходит для PyCharm / обычного запуска).

    Args:
        video_path: путь к видео
        results_list: список сегментов (dict) из retrieve_topk_segments
        max_clips: сколько сегментов показывать
        delay: задержка между кадрами, сек
        max_frames_per_clip: ограничение на число кадров в сегменте (None = без лимита)
    """
    num_to_show = min(len(results_list), max_clips)
    if num_to_show == 0:
        print("No segments to show.")
        return

    # интерактивный режим — окно будет обновляться без блокировки
    plt.ion()

    fig, ax = plt.subplots()
    ax.axis("off")

    for r in results_list[:num_to_show]:
        start = int(r.get("start_frame", 0))
        end = int(r.get("end_frame", start))

        score = r.get("score", None)
        score_s = "?" if score is None else f"{float(score):.3f}"

        t0 = r.get("start_time_sec", None)
        t1 = r.get("end_time_sec", None)

        print(
            f"Rank {r.get('rank','?')}, score={score_s}, "
            f"time {_fmt_time(t0)}-{_fmt_time(t1)} sec"
        )

        if end <= start:
            print("  (skip: empty segment)")
            continue

        frame_indices = list(range(start, end))

        # чтобы не читать/рисовать тысячи кадров
        if max_frames_per_clip is not None and len(frame_indices) > max_frames_per_clip:
            step = max(1, len(frame_indices) // max_frames_per_clip)
            frame_indices = frame_indices[::step]

        frames = read_frames(video_path, frame_indices)  # [T,H,W,3] uint8 RGB

        for f in frames:
            ax.clear()
            ax.axis("off")
            ax.imshow(f)

            fig.canvas.draw_idle()
            plt.pause(0.001)  # даём GUI обработать события
            time.sleep(delay)

    # оставляем окно открытым после завершения показа
    plt.ioff()
    plt.show()
