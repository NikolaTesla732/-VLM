# visualize_results.py
from video_io import read_frames
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time


def show_top_segments(video_path: str, results_list: list, max_clips: int = 5, delay: float = 0.2):
    """
    Пошагово отображает лучшие сегменты видео для текстового запроса.

    Args:
        video_path: путь к видео файлу
        results_list: список сегментов из retrieve_topk_segments
        max_clips: сколько топ-сегментов показывать
        delay: задержка между кадрами в секундах
    """
    num_to_show = min(len(results_list), max_clips)

    for r in results_list[:num_to_show]:
        start = r["start_frame"]
        end = r["end_frame"]
        print(f"Rank {r['rank']}, score={r['score']:.3f}, time {r['start_time_sec']:.2f}-{r['end_time_sec']:.2f} sec")

        frames = read_frames(video_path, list(range(start, end)))  # [T,H,W,3]

        for f in frames:
            plt.imshow(f)
            plt.axis('off')
            display(plt.gcf())
            clear_output(wait=True)
            time.sleep(delay)
        clear_output(wait=True)
