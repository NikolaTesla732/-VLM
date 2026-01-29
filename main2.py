import os
import glob
import time

from motion_detect import motion_detect
from split_video_mvp import split_video_mvp
from count_deleteVideo_video import count_video, delete_video
from person_detect import person_detect


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


input_path = r"Videos/Vandalism"   # папка с видео
base_out_path = r"video_split"     # куда складывать нарезку

# какие видео ищем
video_exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")

videos = []
for ext in video_exts:
    videos += glob.glob(os.path.join(input_path, ext))

videos = sorted(videos)

if not videos:
    raise RuntimeError(f"Не найдено видео в папке: {input_path}")

t0_total = time.perf_counter()

all_results = {}

for idx, video_file in enumerate(videos, 1):
    print(f"\n==============================")
    print(f"[{idx}/{len(videos)}] Processing: {video_file}")
    print(f"==============================")

    t0_one = time.perf_counter()

    # отдельная папка под каждый файл
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_path = os.path.join(base_out_path, video_name)

    video = split_video_mvp(
        input_path=video_file,
        out_dir=out_path,
        segment_seconds=5,
        crop=True,
    )

    video_all = video.copy()
    count = count_video(dir_path=out_path, recursive=False)

    # 1) motion filter
    for i in range(len(count) - 1, -1, -1):
        part_id = count[i]
        part_path = rf"{out_path}\part_{'0'*(3-len(str(part_id)))+str(part_id)}.mp4"

        if not motion_detect(
            video_path=part_path,
            min_motion_seconds=3
        ):
            video.pop(i)
            count.pop(i)

    delete_video(video_all, video, folder_path=out_path)

    # 2) person filter
    for i in range(len(count) - 1, -1, -1):
        part_id = count[i]
        part_path = rf"{out_path}\part_{'0'*(3-len(str(part_id)))+str(part_id)}.mp4"

        if not person_detect(
            video_path=part_path,
            n_frames=8
        ):
            video.pop(i)
            count.pop(i)

    delete_video(video_all, video, folder_path=out_path)

    print("\nОставшиеся фрагменты:")
    print(*video, sep="\n")

    t1_one = time.perf_counter()
    elapsed_one = t1_one - t0_one

    print(f"\nTime for this video: {elapsed_one:.2f} sec ({fmt_hms(elapsed_one)})")

    all_results[video_file] = {
        "kept_segments": video,
        "time_sec": elapsed_one,
        "time_hms": fmt_hms(elapsed_one),
        "out_dir": out_path,
    }

t1_total = time.perf_counter()
elapsed_total = t1_total - t0_total

print(f"\n====================================")
print(f"ALL DONE. Total time: {elapsed_total:.2f} sec ({fmt_hms(elapsed_total)})")
print(f"Processed videos: {len(videos)}")
print(f"====================================")
