# main.py
from __future__ import annotations

import os
from dataclasses import asdict

from config import Config
from xclip_wrapper import load_xclip
from cache_io import ensure_all_cache_dirs
from visualize_results import show_top_segments

from run_one_video import process_one_video  # наш "один файл"
import batch_run  # batch режим


def main():
    cfg = Config()
    ensure_all_cache_dirs(cfg=cfg)
    cfg_dict = asdict(cfg)

    # ВАЖНО: всё управление через config.py (Config.args и/или поля cfg)
    args = getattr(Config, "args", None)
    if args is None:
        raise RuntimeError("В config.py нет Config.args (Namespace). Добавь туда video/query и режим.")

    # режим запуска (добавь это поле в Config.args)
    # args.mode = "single" или "batch"
    mode = getattr(args, "mode", "single").lower()

    # загрузка модели нужна и для single, и для batch (batch_run сам может грузить — но лучше единообразно)
    model, processor = load_xclip(cfg=cfg)

    if mode == "batch":
        # batch_run уже использует process_one_video(...) внутри
        batch_run.main()
        return

    # single mode
    video_path = getattr(args, "video", None)
    query = getattr(args, "query", None)
    show = bool(getattr(args, "show", True))  # показывать в конце или нет

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"Config.args.video не задан или файл не найден: {video_path}")
    if not query:
        raise ValueError("Config.args.query не задан")

    out = process_one_video(
        video_path,
        query=query,
        model=model,
        processor=processor,
        cfg=cfg,
        cfg_dict=cfg_dict,
    )

    # визуализация только для single
    if show and out.get("results_list"):
        top_k = int(getattr(cfg, "top_k", 1))
        show_top_segments(video_path, out["results_list"], max_clips=top_k, delay=0.15)


if __name__ == "__main__":
    main()
