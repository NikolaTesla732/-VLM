# batch_run.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import parse_cli
from cache_io import ensure_all_cache_dirs
from models import create_backend
from run_one_video import process_one_video, fmt_hms


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def iter_videos(folder: str, *, recursive: bool = True) -> List[str]:
    root = Path(folder)
    if not root.exists():
        return []

    if root.is_file():
        return [str(root)]

    if recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in root.glob("*") if p.suffix.lower() in VIDEO_EXTS]

    return [str(p) for p in sorted(files)]


def progress_bar(done: int, total: int, *, width: int = 34) -> str:
    if total <= 0:
        return "[?]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def run_batch(
    *,
    videos_dir: str,
    query: str,
    cfg: object,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Прогон по папке.

    Оптимизации:
    - модель загружается один раз
    - text_emb считается один раз и переиспользуется для всех видео
    """

    ensure_all_cache_dirs(cfg=cfg)

    backend = create_backend(cfg, load=True)

    # куда сохранять репорты/клипы
    out_dir = out_dir or getattr(cfg, "batch_out_dir", "./batch_out")
    clips_subdir = getattr(cfg, "clips_subdir", "clips")
    clips_root = str(Path(out_dir) / clips_subdir)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(clips_root).mkdir(parents=True, exist_ok=True)

    videos = iter_videos(videos_dir, recursive=True)
    if not videos:
        raise RuntimeError(f"No videos found in: {videos_dir}")

    summary_path = str(Path(out_dir) / "summary.json")
    report_path = str(Path(out_dir) / "report.json")

    # считаем text_emb один раз (ускоряет batch)
    normalize_embeddings = bool(getattr(cfg, "normalize_embeddings", True))
    text_emb = backend.encode_text([query], normalize=normalize_embeddings)

    summary: List[Dict[str, Any]] = []
    batch_t0 = time.perf_counter()

    total_video_duration = 0.0
    total_processing_time_ok = 0.0
    ok_count = 0

    avg_sec = 0.0
    alpha = 0.25

    total = len(videos)
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{total}] {vp}")
        try:
            res = process_one_video(
                vp,
                query=query,
                backend=backend,
                cfg=cfg,
                clips_root=clips_root,
                text_emb=text_emb,
            )
            res["ok"] = True
        except Exception as e:
            res = {"video": vp, "ok": False, "error": repr(e)}
            print("  ERROR:", e)

        summary.append(res)

        if res.get("ok"):
            ok_count += 1
            total_processing_time_ok += float(res.get("processing_time_sec", 0.0))
            total_video_duration += float(res.get("video_duration_sec", 0.0))

            cur_sec = float(res.get("processing_time_sec", 0.0))
            if cur_sec > 0:
                avg_sec = cur_sec if avg_sec == 0.0 else (1 - alpha) * avg_sec + alpha * cur_sec

            cached = "cached" if res.get("cached_results") else ("index_cached" if res.get("cached_index") else "fresh")
            clips_n = len(res.get("saved_clips", []))
            print(
                f"  -> {cached} | proc {res.get('processing_time_hms')} | "
                f"video {res.get('video_duration_hms')} | clips {clips_n}"
            )

        elapsed_sec = time.perf_counter() - batch_t0
        remaining = total - i
        eta_sec = avg_sec * remaining if avg_sec > 0 else 0.0
        bar = progress_bar(i, total)
        pct = 100.0 * i / total if total else 0.0
        print(f"{bar} {pct:6.2f}% | elapsed {fmt_hms(elapsed_sec)} | ETA {fmt_hms(eta_sec)}")

        # incremental save summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    batch_elapsed = time.perf_counter() - batch_t0

    report = {
        "videos_dir": videos_dir,
        "num_videos_total": len(videos),
        "num_videos_ok": ok_count,
        "query": query,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
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
    return report


def main() -> None:
    run, cfg = parse_cli()
    if run.mode != "batch":
        raise SystemExit("batch_run.py запускается только с --mode batch")

    run_batch(videos_dir=run.video, query=run.query, cfg=cfg)


if __name__ == "__main__":
    main()
