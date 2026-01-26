# config.py
from __future__ import annotations

import argparse
from dataclasses import dataclass

@dataclass
class Config:
    args = argparse.Namespace(
        mode="batch",           # single или "batch"
        video="Videos/",
        query="Vandalism",
        show=True,
    )
    # -------- Model / device --------
    model_name: str = "microsoft/xclip-base-patch16"
    device: str = "cpu"          # "cuda:0" если есть
    dtype: str = "fp32"          # "fp16" для cuda обычно
    use_torch_compile: bool = False

    # -------- Clip segmentation / indexing --------
    clip_len_frames: int = 8
    clip_stride_frames: int = 32
    sample_strategy: str = "uniform"   # "uniform" | "head"
    batch_size_clips: int = 32          # сколько клипов прогонять за раз (если у тебя есть batching)

    # -------- Retrieval --------
    top_k: int = 5                     # сколько результатов искать/показывать

    # -------- Batch run inputs/outputs --------
    videos_dir: str = "./Videos"
    batch_out_dir: str = "./batch_out"
    clips_subdir: str = "clips"        # внутри batch_out_dir

    # -------- Query --------
    query: str = "Vandalism"

    # -------- Clip export (ffmpeg) --------
    export_clips: bool = True
    export_top_k: int = 5              # сколько сохранять клипов как видео
    pad_sec: float = 2.0               # запас по краям (делает клипы длиннее)
    reencode: bool = True              # True точнее, False быстрее

    # -------- Cache --------
    cache_results: bool = True
    cache_index: bool = True
    strict_cache_match: bool = True
    force_reindex: bool = False

    # -------- NEW: save top-N times --------
    save_top_n_times: int = 5          # сохранять start/end для top-N (0 = отключить)
