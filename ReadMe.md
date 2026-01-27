# Сыщик норм — Video Retrieval (XCLIP)

Проект: индексация видео на клипы + поиск по текстовому запросу (text→video retrieval) на базе **XCLIP**.

Идея пайплайна:
1) Разбиваем видео на клипы (по кадрам).
2) Считаем эмбеддинги клипов моделью.
3) Считаем эмбеддинг текста.
4) Ищем top-k клипов по cosine similarity.
5) (Опционально) экспортируем найденные клипы в отдельные .mp4 через ffmpeg.

---

## Структура репозитория
- `main.py` — входная точка (single/batch)
- `config.py` — настройки пайплайна + CLI
- `run_one_video.py` — `process_one_video(...)` (кэш → индекс → retrieval → сохранение → (опц.) экспорт)
- `batch_run.py` — прогон папки с видео
- `xclip_wrapper.py` — загрузка модели + энкодинг (текст / видео)
- `index_video.py` — нарезка видео на клипы и индексация
- `retrieve.py` — similarity + top-k
- `video_io.py` — чтение кадров через `decord`
- `cache_io.py` — кэш индекса/результатов
- `visualize_results.py` — показ найденных сегментов (matplotlib)

---

## Зависимости
### Python пакеты
Минимально нужны:
- `torch`
- `transformers`
- `numpy`
- `Pillow`
- `decord`
- `matplotlib`

См. `requirements.txt`.

### Внешние зависимости
- **ffmpeg + ffprobe** — нужны для:
  - определения длительности видео (ffprobe)
  - экспорта клипов (ffmpeg)

Если ffmpeg не установлен — пайплайн всё равно отдаст `results_list`, но:
- `video_duration_sec` будет 0
- экспорт клипов будет пропущен (в payload появится `export_error`)

---

## Быстрый старт
### 1) Установка
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
