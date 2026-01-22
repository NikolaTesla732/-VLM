# cache_io.py
import os
import json
import hashlib
from typing import Dict, Any, Optional
from defaults import pick
import torch

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def stable_hash(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def ensure_all_cache_dirs(
    cfg: Optional[object] = None,
    root_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    run_meta_dir: Optional[str] = None,
) -> None:
    root_dir = pick(root_dir, cfg, "paths.root_dir", "./cache")
    index_dir = pick(index_dir, cfg, "paths.index_dir", os.path.join(root_dir, "index"))
    results_dir = pick(results_dir, cfg, "paths.results_dir", os.path.join(root_dir, "results"))
    run_meta_dir = pick(run_meta_dir, cfg, "paths.run_meta_dir", os.path.join(root_dir, "run_meta"))

    ensure_dir(root_dir)
    ensure_dir(index_dir)
    ensure_dir(results_dir)
    ensure_dir(run_meta_dir)

def make_index_path(
    video_path: str,
    model_name: str,
    cfg: Optional[object] = None,
    index_dir: Optional[str] = None,
    index_name_tpl: Optional[str] = None,
    index_ext: Optional[str] = None,
) -> str:
    # если cfg есть — можно не передавать index_dir/tpl/ext
    index_dir = pick(index_dir, cfg, "paths.index_dir", "./cache/index")
    index_name_tpl = pick(index_name_tpl, cfg, "paths.index_name_tpl", "index_{key}")
    index_ext = pick(index_ext, cfg, "paths.index_ext", ".pt")

    key = stable_hash({"video_path": video_path, "model_name": model_name})
    fname = index_name_tpl.format(key=key) + index_ext
    return os.path.join(index_dir, fname)

def make_results_path(
    video_path: str,
    model_name: str,
    query: str,
    cfg: Optional[object] = None,
    results_dir: Optional[str] = None,
    results_name_tpl: Optional[str] = None,
    results_ext: Optional[str] = None,
) -> str:
    results_dir = pick(results_dir, cfg, "paths.results_dir", "./cache/results")
    results_name_tpl = pick(results_name_tpl, cfg, "paths.results_name_tpl", "results_{key}")
    results_ext = pick(results_ext, cfg, "paths.results_ext", ".json")

    key = stable_hash({"video_path": video_path, "model_name": model_name, "query": query})
    fname = results_name_tpl.format(key=key) + results_ext
    return os.path.join(results_dir, fname)

def make_last_run_path(
    cfg: Optional[object] = None,
    run_meta_dir: Optional[str] = None,
    last_run_name: Optional[str] = None,
) -> str:
    run_meta_dir = pick(run_meta_dir, cfg, "paths.run_meta_dir", "./cache/run_meta")
    last_run_name = pick(last_run_name, cfg, "paths.last_run_name", "last_run.json")
    return os.path.join(run_meta_dir, last_run_name)

def save_index(path: str, index: Dict[str, Any], cfg_dict: Dict[str, Any]) -> None:
    payload = dict(index)
    payload["cfg"] = cfg_dict
    payload["cfg_hash"] = stable_hash(cfg_dict)
    torch.save(payload, path)

def load_index(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")

def index_matches_cfg(index_payload: Dict[str, Any], cfg_dict: Dict[str, Any]) -> bool:
    saved = index_payload.get("cfg_hash", None)
    current = stable_hash(cfg_dict)
    return (saved is not None) and (saved == current)

def save_results_json(path: str, results: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_last_run(
    payload: Dict[str, Any],
    cfg: Optional[object] = None,
    path: Optional[str] = None,
) -> str:
    path = pick(path, cfg, "paths.last_run_path", None)  # обычно нет, но оставим хук
    if path is None:
        path = make_last_run_path(cfg=cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
