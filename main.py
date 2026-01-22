# main.py (фрагменты)
from dataclasses import asdict
import os
import argparse
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

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video", type=str, required=True)
#     parser.add_argument("--query", type=str, required=True)
#     return parser.parse_args()

# args = parse_args()
ensure_all_cache_dirs(cfg=cfg)
cfg_dict = asdict(cfg)
args = Config.args
model, processor = load_xclip(cfg=cfg)

index_path = make_index_path(video_path=args.video, model_name=cfg.model_name, cfg=cfg)
results_path = make_results_path(video_path=args.video, model_name=cfg.model_name, query=args.query, cfg=cfg)

# индексация (можно вообще не передавать параметры — они подтянутся из cfg)
index = index_video_segments(args.video, model, processor, cfg=cfg)

# retrieval
results_list = retrieve_topk_segments(index, model, processor, args.query, cfg=cfg)
# Показываем топ-5 сегментов
show_top_segments(args.video, results_list, max_clips=5, delay=0.15)
