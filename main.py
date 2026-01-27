from motion_detect import motion_detect
from split_video_mvp import split_video_mvp
from count_deleteVideo_video import count_video
from count_deleteVideo_video import delete_video
from person_detect import person_detect


video = split_video_mvp(
    r"C:\Users\user\Downloads\rrr.mp4",
    r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split",
    10,
    True,
)
video_all = video.copy()
count = count_video(r'C:\Users\user\Documents\Prodjeeeect\all_func\video_split', False)
res = []

for i in count:
    if not motion_detect(rf"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_00{i}.mp4", 3):
        video.pop(i)
        count.pop(i)
delete_video(video_all, video, r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split")

for i in count:
    if not person_detect(rf"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_00{i}.mp4", 8):
        video.pop(i)
        count.pop(i)
delete_video(video_all, video, r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split")

print(*video, sep='\n')