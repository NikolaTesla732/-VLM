# run_one_video.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from cache_io import (
    ensure_all_cache_dirs,
    make_index_path,
    make_results_path,
    save_index,
    load_index,
    index_matches_cfg,
    save_results_json,
    load_results_json,
)
from index_video import index_video_segments
from retrieve import retrieve_topk_segments


# ----------------------------
# utils
# ----------------------------
def safe_name(s: str, max_len: int = 90) -> str:
    s = re.sub(r"[^\w\-\.\(\) ]+", "_", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if len(s) > max_len else s


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def extract_top_times(results_list: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    n = max(0, int(n))
    out: List[Dict[str, Any]] = []
    for i, r in enumerate((results_list or [])[:n], 1):
        out.append({
            "rank": r.get("rank", i),
            "score": r.get("score"),
            "start_time_sec": r.get("start_time_sec"),
            "end_time_sec": r.get("end_time_sec"),
            "start_frame": r.get("start_frame"),
            "end_frame": r.get("end_frame"),
        })
    return out


# ----------------------------
# ffmpeg tools
# ----------------------------
def ensure_ffmpeg_tools() -> None:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def video_duration_sec_ffprobe(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0


def export_segment_ffmpeg(
    video_path: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
    *,
    reencode: bool = True,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    start_sec = float(start_sec)
    end_sec = float(end_sec)
    dur = max(0.0, end_sec - start_sec)
    if dur <= 0:
        raise ValueError(f"Non-positive duration: {start_sec}..{end_sec}")

    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{dur:.3f}",
            "-c", "copy",
            out_path,
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def export_topk_clips(
    video_path: str,
    query: str,
    results_list: List[Dict[str, Any]],
    out_dir: str,
    *,
    top_k: int = 5,
    pad_sec: float = 2.0,
    reencode: bool = True,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    q = safe_name(query)

    saved: List[str] = []
    for i, r in enumerate((results_list or [])[:top_k], 1):
        t0 = r.get("start_time_sec", None)
        t1 = r.get("end_time_sec", None)
        if t0 is None or t1 is None:
            continue

        start = max(0.0, float(t0) - pad_sec)
        end = max(start, float(t1) + pad_sec)

        score = r.get("score", None)
        score_s = "na" if score is None else f"{float(score):.4f}"

        out_name = f"rank{i}_score{score_s}_{q}.mp4"
        out_path = os.path.join(out_dir, out_name)

        export_segment_ffmpeg(video_path, out_path, start, end, reencode=reencode)
        saved.append(out_path)

    return saved


# =========================================================
# MAIN: process one video
# =========================================================
def process_one_video(
    video_path: str,
    *,
    query: str,
    model,
    processor,
    cfg,
    cfg_dict: Optional[dict] = None,
    clips_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Один файл:
      - кэш (index/results)
      - indexing/retrieve если нужно
      - сохраняет results.json (+ top_times)
      - опционально экспортирует top-k клипы
      - возвращает payload с таймингами + stage_times
    """
    ensure_all_cache_dirs(cfg=cfg)

    if cfg_dict is None:
        cfg_dict = asdict(cfg)

    # stage timing helper
    stage: Dict[str, Any] = {}

    def _mark(name: str, t_start: float, t_end: float):
        stage[f"{name}_sec"] = t_end - t_start
        stage[f"{name}_hms"] = fmt_hms(stage[f"{name}_sec"])

    # куда сохранять клипы
    if clips_root is None:
        out_dir = getattr(cfg, "batch_out_dir", "./batch_out")
        clips_subdir = getattr(cfg, "clips_subdir", "clips")
        clips_root = os.path.join(out_dir, clips_subdir)

    # knobs
    top_k = int(getattr(cfg, "top_k", 5))
    save_top_n_times = int(getattr(cfg, "save_top_n_times", top_k))

    cache_index = bool(getattr(cfg, "cache_index", True))
    cache_results = bool(getattr(cfg, "cache_results", True))
    strict_cache_match = bool(getattr(cfg, "strict_cache_match", True))
    force_reindex = bool(getattr(cfg, "force_reindex", False))

    export_clips = bool(getattr(cfg, "export_clips", True))
    export_top_k = int(getattr(cfg, "export_top_k", top_k))
    pad_sec = float(getattr(cfg, "pad_sec", 2.0))
    reencode = bool(getattr(cfg, "reencode", True))

    index_path = make_index_path(video_path=video_path, model_name=cfg.model_name, cfg=cfg)
    results_path = make_results_path(video_path=video_path, model_name=cfg.model_name, query=query, cfg=cfg)

    # total timing
    t0 = time.perf_counter()

    # video duration (content)
    dur_sec = 0.0
    try:
        ensure_ffmpeg_tools()
        dur_sec = video_duration_sec_ffprobe(video_path)
    except Exception:
        dur_sec = 0.0

    # ---- cache check stage
    t_cache0 = time.perf_counter()

    # 1) results cache
    if cache_results and os.path.exists(results_path):
        payload = load_results_json(results_path)
        results_list = payload["results_list"] if isinstance(payload, dict) and "results_list" in payload else payload
        top_times = payload.get("top_times") if isinstance(payload, dict) else None
        if top_times is None:
            top_times = extract_top_times(results_list, save_top_n_times)

        t_cache1 = time.perf_counter()
        _mark("cache_check", t_cache0, t_cache1)

        elapsed = time.perf_counter() - t0
        stage["total_sec"] = elapsed
        stage["total_hms"] = fmt_hms(elapsed)

        return {
            "video": video_path,
            "query": query,
            "cached_results": True,
            "cached_index": None,
            "index_path": index_path,
            "results_path": results_path,
            "results_list": results_list,
            "top_times": top_times,
            "saved_clips": [],
            "clips_dir": "",
            "video_duration_sec": dur_sec,
            "video_duration_hms": fmt_hms(dur_sec),
            "processing_time_sec": elapsed,
            "processing_time_hms": fmt_hms(elapsed),
            "stage_times": stage,
        }

    t_cache1 = time.perf_counter()
    _mark("cache_check", t_cache0, t_cache1)

    # ---- index stage
    t_index0 = time.perf_counter()

    index = None
    cached_index = False
    if cache_index and os.path.exists(index_path) and not force_reindex:
        idx_payload = load_index(index_path)
        if (not strict_cache_match) or index_matches_cfg(idx_payload, cfg_dict):
            index = idx_payload
            cached_index = True

    if index is None:
        index = index_video_segments(video_path, model, processor, cfg=cfg)
        if cache_index:
            save_index(index_path, index=index, cfg_dict=cfg_dict)

    t_index1 = time.perf_counter()
    _mark("index_build_or_load", t_index0, t_index1)

    # ---- retrieve stage
    t_ret0 = time.perf_counter()
    results_list = retrieve_topk_segments(index, model, processor, query, cfg=cfg)
    t_ret1 = time.perf_counter()
    _mark("retrieve", t_ret0, t_ret1)

    top_times = extract_top_times(results_list, save_top_n_times)

    # ---- save results (very small, but still track)
    t_save0 = time.perf_counter()
    if cache_results:
        save_results_json(results_path, {
            "video": video_path,
            "query": query,
            "model_name": cfg.model_name,
            "cfg": cfg_dict,
            "results_list": results_list,
            "top_times": top_times,
        })
    t_save1 = time.perf_counter()
    _mark("save_results", t_save0, t_save1)

    # ---- export stage
    t_exp0 = time.perf_counter()
    saved_clips: List[str] = []
    clips_dir = ""
    if export_clips:
        try:
            ensure_ffmpeg_tools()
            video_base = safe_name(os.path.splitext(os.path.basename(video_path))[0])
            clips_dir = os.path.join(clips_root, video_base, safe_name(query))
            saved_clips = export_topk_clips(
                video_path=video_path,
                query=query,
                results_list=results_list,
                out_dir=clips_dir,
                top_k=min(export_top_k, top_k),
                pad_sec=pad_sec,
                reencode=reencode,
            )
        except Exception:
            saved_clips = []
            clips_dir = ""

    t_exp1 = time.perf_counter()
    _mark("export", t_exp0, t_exp1)

    elapsed = time.perf_counter() - t0
    stage["total_sec"] = elapsed
    stage["total_hms"] = fmt_hms(elapsed)

    return {
        "video": video_path,
        "query": query,
        "cached_results": False,
        "cached_index": cached_index,
        "index_path": index_path,
        "results_path": results_path,
        "results_list": results_list,
        "top_times": top_times,
        "saved_clips": saved_clips,
        "clips_dir": clips_dir,
        "top1": results_list[0] if results_list else None,
        "video_duration_sec": dur_sec,
        "video_duration_hms": fmt_hms(dur_sec),
        "processing_time_sec": elapsed,
        "processing_time_hms": fmt_hms(elapsed),
        "stage_times": stage,
    }
