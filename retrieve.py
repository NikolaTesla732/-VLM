# retrieve.py
from typing import List, Dict, Any, Optional
import torch

from defaults import pick
from xclip_wrapper import encode_texts

def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.T

def frames_to_time(frame_idx: int, fps: Optional[float]) -> Optional[float]:
    if fps is None or fps <= 0:
        return None
    return float(frame_idx) / float(fps)

def retrieve_topk_segments(
    index: Dict[str, Any],
    model,
    processor,
    query_text: str,
    # explicit (optional)
    device: Optional[str] = None,
    top_k: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    # cfg optional
    cfg: Optional[object] = None,
) -> List[Dict[str, Any]]:
    device = pick(device, cfg, "device", "cpu")
    top_k = pick(top_k, cfg, "top_k", 5)
    normalize_embeddings = pick(normalize_embeddings, cfg, "normalize_embeddings", True)

    text_emb = encode_texts(
        model=model,
        processor=processor,
        texts=[query_text],
        device=device,
        normalize=normalize_embeddings,
        cfg=cfg,
    )

    video_embs = index["embeddings"].to(text_emb.device)  # [C,D]
    sims = cosine_sim_matrix(text_emb, video_embs)[0]     # [C]
    vals, idxs = torch.topk(sims, k=min(top_k, sims.numel()))

    fps = index.get("fps", None)
    results = []
    for score, clip_i in zip(vals.tolist(), idxs.tolist()):
        s, e = index["ranges"][clip_i]
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time_sec": frames_to_time(s, fps),
            "end_time_sec": frames_to_time(e, fps),
        })
    return results
