from motion_detect import motion_detect
from split_video_mvp import split_video_mvp
from count_deleteVideo_video import count_video
from count_deleteVideo_video import delete_video
from person_detect import person_detect

video = split_video_mvp(
    input_path=r"C:\Users\user\Downloads\rrr.mp4",
    out_dir=r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split",
    segment_seconds=5,
    crop=True,
)

video_all = video.copy()
count = count_video(dir_path=r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split", recursive=False)

for i in range(len(count) - 1, -1, -1):
    if not motion_detect(
        video_path=rf"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_00{count[i]}.mp4",
        min_motion_seconds=3
    ):
        video.pop(i)
        count.pop(i)

delete_video(video_all, video, folder_path=r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split")

for i in range(len(count) - 1, -1, -1):
    if not person_detect(
        video_path=rf"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_00{count[i]}.mp4",
        n_frames=8
    ):
        video.pop(i)
        count.pop(i)

delete_video(video_all, video, folder_path=r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split")

print(*video, sep="\n")
