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
def encode_videos(model, processor, videos_rgb_uint8, device=None, normalize=None, cfg=None):
    from PIL import Image
    import torch
    import numpy as np

    device = pick(device, cfg, "device", "cpu")
    normalize = pick(normalize, cfg, "normalize_embeddings", True)

    if videos_rgb_uint8.dtype != np.uint8:
        videos_rgb_uint8 = videos_rgb_uint8.astype(np.uint8, copy=False)

    B, T, H, W, C = videos_rgb_uint8.shape
    assert C == 3

    # 1) preprocess frames as images (stable on твоей версии transformers)
    frames = [Image.fromarray(videos_rgb_uint8[i, t]) for i in range(B) for t in range(T)]
    out = processor(images=frames, return_tensors="pt")
    pixel_values = out["pixel_values"]  # [B*T, 3, H', W'] обычно

    # иногда processor добавляет лишние оси 1 — убираем
    while pixel_values.dim() > 4 and 1 in pixel_values.shape:
        for d in range(pixel_values.dim()):
            if pixel_values.shape[d] == 1:
                pixel_values = pixel_values.squeeze(d)
                break

    if pixel_values.dim() != 4:
        raise RuntimeError(f"pixel_values must be 4D [N,3,H,W], got {tuple(pixel_values.shape)}")

    pixel_values = pixel_values.to(device)

    # 2) vision tower (кадровые фичи)
    # return_dict=False специально: тогда даже если HF чудит, мы стабильно читаем tuple
    vision_out = model.vision_model(pixel_values=pixel_values, return_dict=False)

    # Обычно: (last_hidden_state, pooler_output, ...)
    if isinstance(vision_out, (tuple, list)):
        last_hidden = vision_out[0]
        pooled = vision_out[1] if len(vision_out) > 1 and vision_out[1] is not None else last_hidden[:, 0]
    else:
        last_hidden = vision_out.last_hidden_state
        pooled = vision_out.pooler_output if vision_out.pooler_output is not None else last_hidden[:, 0]

    # pooled: [B*T, hidden] -> [B, T, hidden]
    # pooled: [B*T, 768] (или другой hidden), но mit у тебя ждёт 512
    # 1) project -> 512
    if not hasattr(model, "visual_projection") or model.visual_projection is None:
        raise RuntimeError("model.visual_projection missing; can't match MIT dim")

    frame_feats_512 = model.visual_projection(pooled)  # [B*T, 512]

    # 2) [B*T, 512] -> [B, T, 512]
    frame_feats_512 = frame_feats_512.view(B, T, -1)

    # 3) MIT (без kwargs, твоя версия не принимает inputs_embeds)
    mit_out = model.mit(frame_feats_512)

    # 4) аккуратно достаём pooled/CLS (tuple vs ModelOutput)
    if isinstance(mit_out, (tuple, list)):
        mit_last = mit_out[0]
        mit_pooled = mit_out[1] if len(mit_out) > 1 and mit_out[1] is not None else mit_last[:, 0]
    else:
        mit_last = mit_out.last_hidden_state
        mit_pooled = mit_out.pooler_output if getattr(mit_out, "pooler_output", None) is not None else mit_last[:, 0]

    video_feats = mit_pooled  # уже [B, 512]

    if normalize:
        video_feats = torch.nn.functional.normalize(video_feats, dim=-1)

    return video_feats
