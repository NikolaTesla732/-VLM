# -VLM: Text → Video Segment Retrieval (XCLIP)

Мини-проект для **поиска релевантных отрезков видео по текстовому запросу** с помощью модели **XCLIP** из `transformers`.

Идея простая:
1) видео режется на клипы по кадрам (sliding window),
2) для каждого клипа считаются видео-эмбеддинги,
3) текст кодируется в текстовый эмбеддинг,
4) считаем сходство `text_emb @ video_embs.T`,
5) берём `top_k` клипов и визуализируем.

---

## Быстрый старт

### Зависимости
Проект использует:
- `torch`
- `transformers` (XCLIPModel, XCLIPProcessor)
- `decord` (чтение видео)
- `numpy`
- `matplotlib`
- `ipython` (для `display/clear_output` в визуализации)

> Если нет `decord`, чтение видео упадёт с подсказкой по установке (см. `video_io.py`).  

### Запуск
Сейчас запуск берёт параметры из `Config.args` (внутри `config.py`), а `argparse` в `main.py` закомментирован.

По умолчанию:
- `video="./Videos/video.mp4"`
- `query="The man left the frame"`

---

## Структура репозитория

Файлы (по факту кода):

- `main.py` — точка входа, склеивает пайплайн
- `config.py` — конфигурация и пути кэша
- `defaults.py` — утилита `pick()` для параметров с приоритетом (явный аргумент → cfg → default)
- `xclip_wrapper.py` — загрузка XCLIP + кодирование текста/видео
- `video_io.py` — чтение видео/кадров через decord + нарезка клипов
- `index_video.py` — построение индекса видео (ranges + embeddings)
- `retrieve.py` — retrieval top-k клипов по запросу
- `cache_io.py` — кэш (пути, save/load индекса, save/load результатов)
- `visualize_results.py` — показ top-сегментов кадр за кадром

---

## Подробно по каждому файлу и функциям

### `main.py`
Главный скрипт: создаёт `cfg`, загружает модель, индексирует видео, делает retrieval, показывает сегменты.

Ключевые действия:
- создаёт `cfg = Config()`  
- создаёт директории кэша: `ensure_all_cache_dirs(cfg=cfg)`  
- берёт параметры запуска из `args = Config.args`  
- загружает модель: `model, processor = load_xclip(cfg=cfg)`  
- строит пути кэша:
  - `index_path = make_index_path(video_path=..., model_name=..., cfg=cfg)`
  - `results_path = make_results_path(video_path=..., model_name=..., query=..., cfg=cfg)`
- **сейчас всегда** пересчитывает индекс (не читает/не пишет кэш):
  - `index = index_video_segments(args.video, model, processor, cfg=cfg)`
- retrieval:
  - `results_list = retrieve_topk_segments(index, model, processor, args.query, cfg=cfg)`
- визуализация:
  - `show_top_segments(..., max_clips=5, delay=0.15)`

> Важно: хотя `cache_io` импортирован (save/load и проверки), в `main.py` это пока не задействовано — индекс всегда пересчитывается.

---

### `config.py`
Содержит dataclass-конфиг.

#### `CachePaths`
Пути и шаблоны имён артефактов:
- `root_dir`, `index_dir`, `results_dir`, `run_meta_dir`
- `index_ext`, `results_ext`
- `index_name_tpl`, `results_name_tpl`
- `last_run_name`

#### `Config`
Основные параметры пайплайна:
- **Model**
  - `model_name` (по умолчанию `"microsoft/xclip-base-patch16"`)
  - `device` (`"cpu"` или `"cuda"`)
  - `dtype` (`"fp16"`/`"fp32"`)
  - `use_torch_compile`
- **Псевдо-аргументы запуска**
  - `args = argparse.Namespace(video=..., query=...)`
- **Video segmentation**
  - `clip_len_frames`
  - `clip_stride_frames`
  - `fps_hint`
- **Sampling**
  - `sample_strategy` (`"uniform"` или `"head"`)
- **Retrieval**
  - `top_k`
  - `batch_size_clips`
  - `normalize_embeddings`
- **Cache policy**
  - `force_reindex`
  - `strict_cache_match`
- **Paths**
  - `paths: CachePaths`
- **IO**
  - `verbose`

---

### `defaults.py`
#### `pick(value, cfg, attr_path, default)`
Утилита выбора значения:
- если `value` не `None` → вернуть его
- иначе если `cfg` есть и в нём есть атрибут по пути `attr_path` (например `"paths.index_dir"`) → вернуть его
- иначе → `default`

Используется почти везде, чтобы параметры можно было задавать явно или через `cfg`.

---

### `xclip_wrapper.py`

#### `get_torch_dtype(dtype_str: str) -> torch.dtype`
- `"fp16"` → `torch.float16`, иначе `torch.float32`

#### `load_xclip(..., cfg=None) -> (XCLIPModel, XCLIPProcessor)`
Параметры можно передать явно или взять из `cfg` через `pick()`:
- `model_name`, `device`, `dtype_str`, `use_torch_compile`

Делает:
- `processor = XCLIPProcessor.from_pretrained(model_name)`
- `model = XCLIPModel.from_pretrained(model_name)`
- `model.to(device)` (+ dtype на cuda)
- `model.eval()`
- опционально `torch.compile(model)`

#### `encode_texts(model, processor, texts, device=None, normalize=None, cfg=None) -> torch.Tensor`
- кодирует список текстов
- `model.get_text_features(**inputs)`
- опционально `torch.nn.functional.normalize(..., dim=-1)`

#### `encode_videos(model, processor, videos_rgb_uint8, device=None, normalize=None, cfg=None) -> torch.Tensor`
- вход: `np.ndarray` формы `[B, T, H, W, 3]` (uint8 RGB)
- превращает в `video_list` формата, который ждёт `XCLIPProcessor`
- `model.get_video_features(**inputs)`
- опционально нормализует

---

### `video_io.py`
Работа с видео через `decord.VideoReader`.

#### `_try_import_decord()`
- пытается импортировать `decord`
- если не удалось — бросает `ImportError` с подсказкой `pip install decord`

#### `read_video_metadata(video_path, fps_hint=None) -> dict`
- открывает видео
- `num_frames = len(vr)`
- пытается взять `fps = vr.get_avg_fps()`, если не получилось — берёт `fps_hint`
- возвращает `{"num_frames": ..., "fps": ...}`

#### `read_frames(video_path, frame_indices) -> np.ndarray`
- клипает индексы в допустимый диапазон `[0, len(vr)-1]`
- читает батч кадров `vr.get_batch(...)`
- возвращает `[T, H, W, 3]` uint8 RGB

#### `sample_indices_uniform(start, end, num) -> List[int]`
- равномерно семплирует `num` индексов в `[start, end)` (end exclusive)
- использует `np.linspace(..., end-1, num=num)` и округление

#### `build_clips(frame_count, clip_len, stride) -> List[(start, end)]`
- генерирует список клипов как `(start_frame, end_frame_exclusive)`
- последний клип может быть короче — добивка делается на этапе семплинга кадров

---

### `index_video.py`
Построение индекса видео: ranges + embeddings.

#### `clip_to_frame_indices(clip_start, clip_end, clip_len_frames, strategy="uniform") -> List[int]`
- `"head"`: берёт первые `clip_len_frames` кадров (и дополняет последним кадром, если не хватает)
- `"uniform"`: вызывает `sample_indices_uniform(...)`

#### `batchify(items, batch_size)`
- генератор батчей

#### `index_video_segments(video_path, model, processor, ..., cfg=None) -> Dict[str, Any]`
Параметры (все можно брать из cfg через `pick()`):
- `device`
- `clip_len_frames`
- `clip_stride_frames`
- `batch_size_clips`
- `normalize_embeddings`
- `sample_strategy`
- `fps_hint`
- `verbose`

Что делает:
1) `meta = read_video_metadata(...)` → `num_frames`, `fps`
2) `clips = build_clips(num_frames, clip_len_frames, clip_stride_frames)`
3) для каждого батча клипов:
   - для каждого клипа:
     - `frame_idx = clip_to_frame_indices(...)`
     - `frames = read_frames(video_path, frame_idx)` → `[T,H,W,3]`
     - копит `all_ranges.append((s,e))`
   - `videos_np = np.stack(batch_frames, axis=0)` → `[B,T,H,W,3]`
   - `embs = encode_videos(...)`
4) `video_embs = torch.cat(all_embs, dim=0)` (CPU)

Возвращает dict:
- `"video_path"`
- `"fps"`
- `"num_frames"`
- `"clip_len_frames"`
- `"clip_stride_frames"`
- `"ranges"`: список `(start_frame, end_frame)`
- `"embeddings"`: `torch.Tensor` `[num_clips, dim]` (на CPU)

---

### `retrieve.py`

#### `cosine_sim_matrix(a, b) -> torch.Tensor`
- возвращает `a @ b.T` (работает как cosine similarity только если эмбеддинги нормализованы)

#### `frames_to_time(frame_idx, fps) -> Optional[float]`
- `frame_idx / fps` если `fps` валидный, иначе `None`

#### `retrieve_topk_segments(index, model, processor, query_text, ..., cfg=None) -> List[dict]`
Параметры:
- `device`
- `top_k`
- `normalize_embeddings`

Шаги:
1) `text_emb = encode_texts(..., texts=[query_text])`
2) `video_embs = index["embeddings"].to(text_emb.device)`
3) `sims = (text_emb @ video_embs.T)[0]`
4) `topk` по `sims`
5) собирает результаты:
   - `"rank"`
   - `"score"`
   - `"start_frame"`, `"end_frame"`
   - `"start_time_sec"`, `"end_time_sec"` (через fps)

---

### `cache_io.py`
Функции для кэша (пути + сохранение/загрузка).  
Внимание: **инфраструктура кэша есть**, но `main.py` пока её не использует для skip вычислений.

#### `ensure_dir(path)`
`os.makedirs(..., exist_ok=True)`

#### `stable_hash(obj: dict) -> str`
`sha256(json.dumps(sort_keys=True))`

#### `ensure_all_cache_dirs(cfg=None, ...)`
Создаёт:
- `root_dir`
- `index_dir`
- `results_dir`
- `run_meta_dir`

#### `make_index_path(video_path, model_name, cfg=None, ...) -> str`
- hash от `{video_path, model_name}`
- путь вида `index_dir/index_{key}.pt`

#### `make_results_path(video_path, model_name, query, cfg=None, ...) -> str`
- hash от `{video_path, model_name, query}`
- путь вида `results_dir/results_{key}.json`

#### `make_last_run_path(cfg=None, ...) -> str`
- путь в `run_meta_dir` к `last_run.json`

#### `save_index(path, index, cfg_dict)`
- сохраняет `payload = index + cfg + cfg_hash`
- `torch.save(payload, path)`

#### `load_index(path) -> dict`
- `torch.load(..., map_location="cpu")`

#### `index_matches_cfg(index_payload, cfg_dict) -> bool`
- сравнивает `cfg_hash` сохранённого индекса с текущим

#### `save_results_json(path, results)`
#### `load_results_json(path) -> dict`
- сериализация/десериализация JSON

#### `save_last_run(payload, cfg=None, path=None) -> str`
- пишет JSON в `run_meta_dir/last_run.json` (или явный path)

---

### `visualize_results.py`

#### `show_top_segments(video_path, results_list, max_clips=5, delay=0.2)`
Пошагово показывает кадры для каждого сегмента:
- печатает `rank`, `score`, `time start-end`
- читает кадры `read_frames(video_path, range(start, end))`
- показывает каждый кадр через `matplotlib + IPython.display`
- делает `time.sleep(delay)` между кадрами

> Это рассчитано на запуск в Jupyter/Notebook (используется `display/clear_output`).

---

## Что в текущем виде важно знать

- Retrieval использует `a @ b.T`. Поэтому **нормализация эмбеддингов критична**, и она включена по умолчанию (`normalize_embeddings=True`) в `Config` и в `encode_*`.  
- Кэш реализован (индекс и результаты), но `main.py` пока **не делает** `load_index/save_index/load_results_json/save_results_json`.

---

## Идеи улучшений (по делу, под текущий код)

- Включить кэш в `main.py` (если `results_path` существует — сразу показывать, иначе пробовать `index_path`).
- Вернуть CLI через `argparse` (сейчас закомментирован).
- Ограничить количество кадров для визуализации (сейчас `show_top_segments` может читать много кадров, если stride большой).

