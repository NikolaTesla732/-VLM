# main.py
from __future__ import annotations

import os
from dataclasses import asdict

from config import Config
from cache_io import ensure_all_cache_dirs
from xclip_wrapper import load_xclip
from visualize_results import show_top_segments

from run_one_video import process_one_video
import batch_run


def main():
    cfg = Config()
    ensure_all_cache_dirs(cfg=cfg)
    cfg_dict = asdict(cfg)

    args = getattr(Config, "args", None)
    if args is None:
        raise RuntimeError("В config.py нет Config.args (Namespace). Добавь mode/video/query.")

    mode = getattr(args, "mode", "single").lower()

    # batch mode
    if mode == "batch":
        batch_run.main()
        return

    # single mode
    video_path = getattr(args, "video", None)
    query = getattr(args, "query", None)
    show = bool(getattr(args, "show", True))

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Config.args.video не задан или файл не найден: {video_path}")
    if not query:
        raise ValueError("Config.args.query не задан")

    model, processor = load_xclip(cfg=cfg)

    out = process_one_video(
        video_path,
        query=query,
        model=model,
        processor=processor,
        cfg=cfg,
        cfg_dict=cfg_dict,
    )

    # Печать таймингов для 1 видео
    print(f"\n[Single] processing: {out.get('processing_time_hms')} ({out.get('processing_time_sec', 0.0):.2f}s)")
    st = out.get("stage_times", {})
    if st:
        print("Stages:")
        for k in ("cache_check", "index_build_or_load", "retrieve", "save_results", "export", "total"):
            if f"{k}_hms" in st:
                print(f"  - {k:18s}: {st[f'{k}_hms']}")

    if show and out.get("results_list"):
        top_k = int(getattr(cfg, "top_k", 5))
        show_top_segments(video_path, out["results_list"], max_clips=top_k, delay=0.15)


if __name__ == "__main__":
    main()
