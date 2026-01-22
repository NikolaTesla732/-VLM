# xclip_wrapper.py
from typing import List, Optional, Tuple
import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor
from defaults import pick
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if dtype_str.lower() == "fp16" else torch.float32

def load_xclip(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    dtype_str: Optional[str] = None,
    use_torch_compile: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> Tuple[XCLIPModel, XCLIPProcessor]:
    model_name = pick(model_name, cfg, "model_name", "microsoft/xclip-base-patch16")
    device = pick(device, cfg, "device", "cpu")
    dtype_str = pick(dtype_str, cfg, "dtype", "fp32")
    use_torch_compile = pick(use_torch_compile, cfg, "use_torch_compile", False)

    dtype = get_torch_dtype(dtype_str)

    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    model.to(device)
    if device.startswith("cuda"):
        model = model.to(dtype=dtype)

    model.eval()

    if use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model, processor

@torch.no_grad()
def encode_texts(
    model: XCLIPModel,
    processor: XCLIPProcessor,
    texts: List[str],
    device: Optional[str] = None,
    normalize: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> torch.Tensor:
    device = pick(device, cfg, "device", "cpu")
    normalize = pick(normalize, cfg, "normalize_embeddings", True)

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_feats = model.get_text_features(**inputs)

    if normalize:
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
    return text_feats

@torch.no_grad()
def encode_videos(
    model: XCLIPModel,
    processor: XCLIPProcessor,
    videos_rgb_uint8: np.ndarray,   # [B,T,H,W,3]
    device: Optional[str] = None,
    normalize: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> torch.Tensor:
    device = pick(device, cfg, "device", "cpu")
    normalize = pick(normalize, cfg, "normalize_embeddings", True)
    video_list = [
    [videos_rgb_uint8[i, t] for t in range(videos_rgb_uint8.shape[1])]
    for i in range(videos_rgb_uint8.shape[0])
]

    inputs = processor(videos=video_list, return_tensors="pt",padding=True)
    # video_list = [videos_rgb_uint8[i] for i in range(videos_rgb_uint8.shape[0])]
    # inputs = processor(videos=video_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    video_feats = model.get_video_features(**inputs)

    if normalize:
        video_feats = torch.nn.functional.normalize(video_feats, dim=-1)
    return video_feats
