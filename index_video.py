# index_video.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import time

from defaults import pick
from video_io import open_video, sample_indices_uniform, build_clips
from models.base import BaseVideoTextBackend


def clip_to_frame_indices(
    clip_start: int,
    clip_end: int,
    clip_len_frames: int,
    strategy: str = "uniform",
) -> List[int]:
    """Какие кадры брать из [clip_start, clip_end) для модели."""

    if strategy == "head":
        idx = list(range(clip_start, min(clip_start + clip_len_frames, clip_end)))
        if len(idx) < clip_len_frames:
            idx += [idx[-1] if idx else clip_start] * (clip_len_frames - len(idx))
        return idx

    # uniform
    return sample_indices_uniform(clip_start, clip_end, clip_len_frames)


def batchify(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.no_grad()
def index_video_segments(
    video_path: str,
    backend: BaseVideoTextBackend,
    # ---- explicit params (can be None -> take from cfg/defaults)
    clip_len_frames: Optional[int] = None,
    clip_stride_frames: Optional[int] = None,
    batch_size_clips: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    sample_strategy: Optional[str] = None,
    fps_hint: Optional[float] = None,
    verbose: Optional[bool] = None,
    # ---- cfg (optional)
    cfg: Optional[object] = None,
) -> Dict[str, Any]:
    """Строит индекс: клипы -> эмбеддинги + диапазоны кадров.

    Важно для скорости:
    - открываем VideoReader один раз (VideoSource)
    - читаем кадры пачками через get_batch
    """

    clip_len_frames = int(pick(clip_len_frames, cfg, "clip_len_frames", 8))
    clip_stride_frames = int(pick(clip_stride_frames, cfg, "clip_stride_frames", 64))
    batch_size_clips = int(pick(batch_size_clips, cfg, "batch_size_clips", 32))
    normalize_embeddings = bool(pick(normalize_embeddings, cfg, "normalize_embeddings", True))
    sample_strategy = str(pick(sample_strategy, cfg, "sample_strategy", "uniform"))
    fps_hint = pick(fps_hint, cfg, "fps_hint", None)
    verbose = bool(pick(verbose, cfg, "verbose", True))

    vs = open_video(video_path, fps_hint=fps_hint)
    num_frames = vs.num_frames
    fps = vs.fps

    clips = build_clips(num_frames, clip_len_frames, clip_stride_frames)
    if verbose:
        print(f"[index] video={video_path} frames={num_frames} fps={fps} clips={len(clips)}")

    all_embs: List[torch.Tensor] = []
    all_ranges: List[Tuple[int, int]] = []

    # profiling
    decode_sec = 0.0
    encode_sec = 0.0
    batches = 0

    for batch_clips in batchify(clips, batch_size_clips):
        batches += 1

        # индексы кадров: [B,T]
        idx_matrix: List[List[int]] = []
        for (s, e) in batch_clips:
            idx = clip_to_frame_indices(s, e, clip_len_frames, strategy=sample_strategy)
            idx_matrix.append(idx)
            all_ranges.append((s, e))

        B = len(idx_matrix)
        T = clip_len_frames

        flat_idx = [i for row in idx_matrix for i in row]

        # decode frames once per batch
        t_dec0 = time.perf_counter()
        frames_flat = vs.get_frames(flat_idx)  # [B*T, H, W, 3]
        t_dec1 = time.perf_counter()
        decode_sec += (t_dec1 - t_dec0)

        # reshape
        if frames_flat.ndim != 4 or frames_flat.shape[-1] != 3:
            raise RuntimeError(f"Unexpected frames shape from decoder: {frames_flat.shape}")

        # [B*T,H,W,3] -> [B,T,H,W,3]
        frames_bt = frames_flat.reshape(B, T, *frames_flat.shape[1:])

        # encode
        t_enc0 = time.perf_counter()
        embs = backend.encode_video_clips(frames_bt, normalize=normalize_embeddings)  # [B,D] on backend.device
        t_enc1 = time.perf_counter()
        encode_sec += (t_enc1 - t_enc0)

        # сохраняем на CPU float32 (это важнее для воспроизводимости/кэша)
        all_embs.append(embs.detach().float().cpu())

    video_embs = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 0), dtype=torch.float32)

    return {
        "video_path": video_path,
        "backend": backend.backend_name,
        "model_name": backend.model_name,
        "fps": fps,
        "num_frames": num_frames,
        "clip_len_frames": clip_len_frames,
        "clip_stride_frames": clip_stride_frames,
        "ranges": all_ranges,
        "embeddings": video_embs,  # CPU float32
        "timing": {
            "decode_sec": float(decode_sec),
            "encode_sec": float(encode_sec),
            "batches": int(batches),
        },
    }
