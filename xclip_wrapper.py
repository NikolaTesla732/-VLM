# xclip_wrapper.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor

from defaults import pick


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    return torch.float16 if str(dtype_str).lower() == "fp16" else torch.float32


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

    # ВАЖНО: чтобы выходы были ModelOutput, а не tuple
    if hasattr(model, "config"):
        model.config.return_dict = True

    model.to(device)
    if str(device).startswith("cuda"):
        model = model.to(dtype=dtype)

    model.eval()

    # compile на Windows часто даёт сюрпризы — включай только если уверен
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


def _squeeze_singletons(t: torch.Tensor) -> torch.Tensor:
    # Убираем лишние оси размера 1, если processor вдруг их добавил
    # (напр. [T,1,3,H,W] или [1,T,3,H,W])
    while t.dim() > 4 and 1 in t.shape:
        for d in range(t.dim()):
            if t.shape[d] == 1:
                t = t.squeeze(d)
                break
    return t


@torch.no_grad()
def encode_videos(
    model: XCLIPModel,
    processor: XCLIPProcessor,
    videos_rgb_uint8: np.ndarray,  # [B,T,H,W,3] uint8 RGB
    device: Optional[str] = None,
    normalize: Optional[bool] = None,
    cfg: Optional[object] = None,
) -> torch.Tensor:
    """
    Надёжный путь для твоей ситуации:
    - НЕ используем processor(videos=...) (у тебя он возвращал пусто)
    - НЕ используем model.get_video_features() (падал tuple.pooler_output)
    - Берём кадры как images -> pixel_values 4D -> model.vision_model -> mean по T
    """
    from PIL import Image

    device = pick(device, cfg, "device", "cpu")
    normalize = pick(normalize, cfg, "normalize_embeddings", True)

    if videos_rgb_uint8.dtype != np.uint8:
        # processor ожидает uint8 или PIL.Image; приведём мягко
        videos_rgb_uint8 = videos_rgb_uint8.astype(np.uint8, copy=False)

    if videos_rgb_uint8.ndim != 5:
        raise ValueError(f"videos_rgb_uint8 must be 5D [B,T,H,W,3], got {videos_rgb_uint8.shape}")

    B, T, H, W, C = videos_rgb_uint8.shape
    if C != 3:
        raise ValueError(f"Expected RGB frames with 3 channels, got shape {videos_rgb_uint8.shape}")

    # Соберём все кадры всех видео в один список картинок (B*T)
    frames_pil: List[Image.Image] = []
    for i in range(B):
        for t in range(T):
            frames_pil.append(Image.fromarray(videos_rgb_uint8[i, t]))

    out = processor(images=frames_pil, return_tensors="pt")
    if "pixel_values" not in out:
        raise RuntimeError(f"processor(images=...) did not return pixel_values. Keys: {list(out.keys())}")

    pixel_values = out["pixel_values"]  # обычно [B*T, 3, H', W']
    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.as_tensor(pixel_values)

    pixel_values = _squeeze_singletons(pixel_values)

    # гарантируем 4D: [N,3,H,W]
    if pixel_values.dim() != 4:
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)} (expected 4D [N,3,H,W])")

    pixel_values = pixel_values.to(device)

    # Прогоняем через vision tower (она НЕ требует input_ids)
    # Прогоняем через vision tower
    vision_out = model.vision_model(pixel_values=pixel_values, return_dict=True)

    frame_feats = vision_out.pooler_output
    if frame_feats is None:
        frame_feats = vision_out.last_hidden_state[:, 0]  # CLS  -> [B*T, hidden_size=768]

    # >>> ВАЖНО: проектируем hidden_size -> projection_dim (обычно 512), чтобы совпало с text
    if hasattr(model, "visual_projection") and model.visual_projection is not None:
        frame_feats = model.visual_projection(frame_feats)  # [B*T, 512]
    else:
        raise RuntimeError("У модели нет visual_projection — не могу привести video feats к размерности текста.")

    # [B*T, D] -> [B, T, D] -> mean over T
    video_feats = frame_feats.view(B, T, -1).mean(dim=1)

    if normalize:
        video_feats = torch.nn.functional.normalize(video_feats, dim=-1)
    return video_feats
