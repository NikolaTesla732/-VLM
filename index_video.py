# index_video.py
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch

from defaults import pick
from video_io import read_video_metadata, read_frames, sample_indices_uniform, build_clips
from xclip_wrapper import encode_videos

def clip_to_frame_indices(
    clip_start: int,
    clip_end: int,
    clip_len_frames: int,
    strategy: str = "uniform",
) -> List[int]:
    if strategy == "head":
        idx = list(range(clip_start, min(clip_start + clip_len_frames, clip_end)))
        if len(idx) < clip_len_frames:
            idx += [idx[-1] if idx else clip_start] * (clip_len_frames - len(idx))
        return idx
    # uniform
    return sample_indices_uniform(clip_start, clip_end, clip_len_frames)

def batchify(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

@torch.no_grad()
def index_video_segments(
    video_path: str,
    model,
    processor,
    # ---- explicit params (can be None -> take from cfg/defaults)
    device: Optional[str] = None,
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
    device = pick(device, cfg, "device", "cpu")
    clip_len_frames = pick(clip_len_frames, cfg, "clip_len_frames", 16)
    clip_stride_frames = pick(clip_stride_frames, cfg, "clip_stride_frames", 8)
    batch_size_clips = pick(batch_size_clips, cfg, "batch_size_clips", 16)
    normalize_embeddings = pick(normalize_embeddings, cfg, "normalize_embeddings", True)
    sample_strategy = pick(sample_strategy, cfg, "sample_strategy", "uniform")
    fps_hint = pick(fps_hint, cfg, "fps_hint", None)
    verbose = pick(verbose, cfg, "verbose", True)

    meta = read_video_metadata(video_path, fps_hint=fps_hint)
    num_frames = meta["num_frames"]
    fps = meta["fps"]

    clips = build_clips(num_frames, clip_len_frames, clip_stride_frames)
    if verbose:
        print(f"[index] video={video_path} frames={num_frames} fps={fps} clips={len(clips)}")

    all_embs = []
    all_ranges: List[Tuple[int, int]] = []

    for batch_clips in batchify(clips, batch_size_clips):
        batch_frames = []
        for (s, e) in batch_clips:
            frame_idx = clip_to_frame_indices(s, e, clip_len_frames, strategy=sample_strategy)
            frames = read_frames(video_path, frame_idx)  # [T,H,W,3]
            batch_frames.append(frames)
            all_ranges.append((s, e))

        videos_np = np.stack(batch_frames, axis=0)  # [B,T,H,W,3]
        embs = encode_videos(
            model=model,
            processor=processor,
            videos_rgb_uint8=videos_np,
            device=device,
            normalize=normalize_embeddings,
            cfg=cfg,  # не обязательно, но пусть будет
        )
        all_embs.append(embs.detach().cpu())

    video_embs = torch.cat(all_embs, dim=0)

    return {
        "video_path": video_path,
        "fps": fps,
        "num_frames": num_frames,
        "clip_len_frames": clip_len_frames,
        "clip_stride_frames": clip_stride_frames,
        "ranges": all_ranges,
        "embeddings": video_embs,  # CPU
    }
