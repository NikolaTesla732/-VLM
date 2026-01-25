# config.py
from dataclasses import dataclass, field
from typing import Optional
import argparse
@dataclass
class CachePaths:
    # Корневая директория для всех артефактов пайплайна (кэш/логи/результаты).
    root_dir: str = "./cache"

    # Где хранится ИНДЕКС видео (эмбеддинги клипов + ranges + мета + cfg_hash).
    # Обычно самый “тяжёлый” артефакт, пересчитывать дорого.
    index_dir: str = "./cache/index"

    # Где хранится РЕЗУЛЬТАТ конкретного запроса (top-k клипы + таймкоды + score) в JSON.
    # Это самый быстрый кэш: если есть results — индекс/модель можно даже не трогать.
    results_dir: str = "./cache/results"

    # Где хранится служебная информация/логи эксперимента (опционально, но путь задаём явно).
    # Можно писать сюда summary, параметры запуска, time profiling, ошибки.
    run_meta_dir: str = "./cache/run_meta"

    # Расширение файла индекса (torch.save).
    index_ext: str = ".pt"

    # Расширение файла результатов (json).
    results_ext: str = ".json"

    # Шаблон имени индекса. {key} — стабильный хэш (video_path + model_name).
    index_name_tpl: str = "index_{key}"

    # Шаблон имени результатов. {key} — стабильный хэш (video_path + model_name + query).
    results_name_tpl: str = "results_{key}"

    # Файл с “последним запуском”/служебной инфой (в run_meta_dir).
    # Можно хранить последний video/query/cfg для удобства дебага.
    last_run_name: str = "last_run.json"


@dataclass
class Config:
    # ---- Model ----
    model_name: str = "microsoft/xclip-base-patch16"
    device: str = "cpu"         # "cuda" or "cpu"
    dtype: str = "fp16"          # "fp16" | "fp32"
    use_torch_compile: bool = False
    args = argparse.Namespace(
        video="./Videos/video.mp4",
        query="The man left the frame",
    )
    # ---- Video segmentation ----
    clip_len_frames: int = 8
    clip_stride_frames: int = 16
    fps_hint: Optional[float] = None

    # ---- Sampling ----
    sample_strategy: str = "uniform"  # "uniform" | "head"

    # ---- Retrieval ----
    top_k: int = 1
    batch_size_clips: int = 8
    normalize_embeddings: bool = True

    # ---- Cache policy ----
    force_reindex: bool = False
    strict_cache_match: bool = True

    # ---- Paths (все пути заданы явно и используются всеми функциями) ----
    paths: CachePaths = field(default_factory=CachePaths)

    # ---- IO ----
    verbose: bool = True
