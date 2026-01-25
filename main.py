# main.py
from dataclasses import asdict
import os

from visualize_results import show_top_segments

from config import Config
from xclip_wrapper import load_xclip
from index_video import index_video_segments
from retrieve import retrieve_topk_segments
from cache_io import (
    ensure_all_cache_dirs,
    make_index_path,
    make_results_path,
    save_index,
    load_index,
    index_matches_cfg,
    save_results_json,
    load_results_json,
    save_last_run,
)

cfg = Config()
ensure_all_cache_dirs(cfg=cfg)
cfg_dict = asdict(cfg)

args = Config.args  # пока так; если вернёшь argparse — просто подставь args оттуда

model, processor = load_xclip(cfg=cfg)

index_path = make_index_path(video_path=args.video, model_name=cfg.model_name, cfg=cfg)
results_path = make_results_path(video_path=args.video, model_name=cfg.model_name, query=args.query, cfg=cfg)

# 1) Если уже есть готовые results — не считаем вообще ничего
if os.path.exists(results_path):
    print(f"[cache] Using cached results: {results_path}")
    payload = load_results_json(results_path)
    results_list = payload["results_list"] if isinstance(payload, dict) and "results_list" in payload else payload
else:
    # 2) Индекс: пытаемся загрузить подходящий
    index = None
    if os.path.exists(index_path) and not cfg.force_reindex:
        print(f"[cache] Found index: {index_path}")
        index_payload = load_index(index_path)

        if (not cfg.strict_cache_match) or index_matches_cfg(index_payload, cfg_dict):
            print("[cache] Index matches config -> using cached index")
            index = index_payload
        else:
            print("[cache] Index does NOT match config -> reindexing")

    # 3) Если индекса нет/не подошёл — строим заново и сохраняем
    if index is None:
        print("[run] Indexing video...")
        index = index_video_segments(args.video, model, processor, cfg=cfg)

        print(f"[cache] Saving index: {index_path}")
        save_index(index_path, index=index, cfg_dict=cfg_dict)

    # 4) Retrieval
    print("[run] Retrieving top-k segments...")
    results_list = retrieve_topk_segments(index, model, processor, args.query, cfg=cfg)

    # 5) Сохраняем results
    results_payload = {
        "video": args.video,
        "query": args.query,
        "model_name": cfg.model_name,
        "cfg": cfg_dict,
        "results_list": results_list,
    }
    print(f"[cache] Saving results: {results_path}")
    save_results_json(results_path, results_payload)

# 6) last_run (не обязательно, но удобно)
save_last_run(
    {
        "video": args.video,
        "query": args.query,
        "model_name": cfg.model_name,
        "index_path": index_path,
        "results_path": results_path,
    },
    cfg=cfg,
)

# 7) Визуализация
show_top_segments(args.video, results_list, max_clips=cfg.top_k, delay=0.15)
