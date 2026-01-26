# batch_run.py
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict
from typing import List, Dict, Any

from config import Config
from xclip_wrapper import load_xclip
from cache_io import ensure_all_cache_dirs
from run_one_video import process_one_video, fmt_hms


def get_videos_from_folder(folder: str) -> List[str]:
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
    vids: List[str] = []
    for e in exts:
        vids += glob.glob(os.path.join(folder, e))
    return sorted(vids)


def progress_bar(done: int, total: int, *, width: int = 34) -> str:
    if total <= 0:
        return "[?]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main():
    args = getattr(Config, "args", None)
    if args is None:
        raise RuntimeError("В config.py нет Config.args (Namespace). Добавьте mode/video/query.")

    cfg = Config()
    ensure_all_cache_dirs(cfg=cfg)
    cfg_dict = asdict(cfg)

    model, processor = load_xclip(cfg=cfg)

    videos_dir = getattr(args, "video", "./Videos")
    out_dir = getattr(cfg, "batch_out_dir", "./batch_out")
    clips_subdir = getattr(cfg, "clips_subdir", "clips")
    clips_root = os.path.join(out_dir, clips_subdir)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(clips_root, exist_ok=True)

    # query: сначала cfg.query (если ты добавлял), иначе fallback на Config.args.query
    query = getattr(args, "query", None)

    if not query:
        raise ValueError("Не задан query: задай cfg.query или Config.args.query в config.py")

    videos = get_videos_from_folder(videos_dir)
    if not videos:
        raise RuntimeError(f"No videos found in: {videos_dir}")

    summary_path = os.path.join(out_dir, "summary.json")
    report_path = os.path.join(out_dir, "report.json")

    summary: List[Dict[str, Any]] = []
    batch_t0 = time.perf_counter()

    total_video_duration = 0.0
    total_processing_time_ok = 0.0
    ok_count = 0

    # ETA smoothing
    avg_sec = 0.0
    alpha = 0.25  # 0..1

    total = len(videos)
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{total}] {vp}")
        try:
            res = process_one_video(
                vp,
                query=query,
                model=model,
                processor=processor,
                cfg=cfg,
                cfg_dict=cfg_dict,
                clips_root=clips_root,
            )
            res["ok"] = True
        except Exception as e:
            res = {"video": vp, "ok": False, "error": repr(e)}
            print("  ERROR:", e)

        summary.append(res)

        # stats
        if res.get("ok"):
            ok_count += 1
            total_processing_time_ok += float(res.get("processing_time_sec", 0.0))
            total_video_duration += float(res.get("video_duration_sec", 0.0))

            cur_sec = float(res.get("processing_time_sec", 0.0))
            if cur_sec > 0:
                if avg_sec == 0.0:
                    avg_sec = cur_sec
                else:
                    avg_sec = (1 - alpha) * avg_sec + alpha * cur_sec

            cached = "cached" if res.get("cached_results") else "fresh"
            clips_n = len(res.get("saved_clips", []))
            print(
                f"  -> {cached} | proc {res.get('processing_time_hms')} | "
                f"video {res.get('video_duration_hms')} | clips {clips_n}"
            )

        # progress + ETA
        elapsed_sec = time.perf_counter() - batch_t0
        remaining = total - i
        eta_sec = avg_sec * remaining if avg_sec > 0 else 0.0
        bar = progress_bar(i, total)
        pct = 100.0 * i / total if total else 0.0
        print(f"{bar} {pct:6.2f}% | elapsed {fmt_hms(elapsed_sec)} | ETA {fmt_hms(eta_sec)}")

        # incremental save
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    batch_elapsed = time.perf_counter() - batch_t0

    report = {
        "videos_dir": videos_dir,
        "num_videos_total": len(videos),
        "num_videos_ok": ok_count,
        "query": query,
        "model_name": cfg.model_name,
        "total_video_duration_sec": total_video_duration,
        "total_video_duration_hms": fmt_hms(total_video_duration),
        "total_processing_time_sec_ok_only": total_processing_time_ok,
        "total_processing_time_hms_ok_only": fmt_hms(total_processing_time_ok),
        "batch_wall_time_sec": batch_elapsed,
        "batch_wall_time_hms": fmt_hms(batch_elapsed),
        "out_dir": out_dir,
        "clips_root": clips_root,
        "summary_path": summary_path,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Clips:   {clips_root}")
    print(f"Total video duration: {report['total_video_duration_hms']}")
    print(f"Total processing (ok): {report['total_processing_time_hms_ok_only']}")
    print(f"Batch wall time:       {report['batch_wall_time_hms']}")


if __name__ == "__main__":
    main()
