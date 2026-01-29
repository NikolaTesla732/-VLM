# main.py
from __future__ import annotations

import os
from dataclasses import asdict
from config import parse_cli
from cache_io import ensure_all_cache_dirs
from models import create_backend, available_backends
from run_one_video import process_one_video
from visualize_results import show_top_segments
import batch_run


def _apply_torch_tuning(cfg: object) -> None:
    """Небольшие настройки torch для скорости (опционально)."""

    try:
        import torch
    except Exception:
        return

    mc = getattr(cfg, "model", None)
    num_threads = getattr(mc, "torch_num_threads", None) if mc is not None else None
    if isinstance(num_threads, int) and num_threads > 0:
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass


def main() -> None:
    run, cfg = parse_cli()

    ensure_all_cache_dirs(cfg=cfg)
    _apply_torch_tuning(cfg)

    # batch mode — делегируем в batch_run (но CLI единый)
    if run.mode == "batch":
        batch_run.run_batch(videos_dir=run.video, query=run.query, cfg=cfg)
        return

    # single mode
    video_path = run.video
    query = run.query
    show = bool(run.show)

    if not video_path or not os.path.exists(video_path):
        raise ValueError(f"--video не задан или файл не найден: {video_path}")
    if not query:
        raise ValueError("--query не задан")

    backend = create_backend(cfg, load=True)
    cfg_full = asdict(cfg)

    out = process_one_video(
        video_path,
        query=query,
        backend=backend,
        cfg=cfg,
        cfg_full=cfg_full,
    )

    # Печать таймингов
    print(f"\n[Single] processing: {out.get('processing_time_hms')} ({out.get('processing_time_sec', 0.0):.2f}s)")
    st = out.get("stage_times", {})
    if st:
        print("Stages:")
        for k in (
            "video_meta",
            "cache_check",
            "index_build_or_load",
            "retrieve",
            "save_results",
            "export",
            "total",
        ):
            if f"{k}_hms" in st:
                print(f"  - {k:18s}: {st[f'{k}_hms']}")
        # extra index breakdown (sec)
        if "index_decode_hms" in st:
            print(f"  - {'index_decode':18s}: {st['index_decode_hms']} (decode frames)")
        if "index_encode_hms" in st:
            print(f"  - {'index_encode':18s}: {st['index_encode_hms']} (model forward)")

    if show and out.get("results_list"):
        top_k = int(getattr(cfg, "top_k", 5))
        show_top_segments(video_path, out["results_list"], max_clips=top_k, delay=0.15)


if __name__ == "__main__":
    main()
