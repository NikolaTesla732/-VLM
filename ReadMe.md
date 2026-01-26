# Сыщик норм — Video Retrieval (XCLIP)

Проект: индексация видео на клипы + поиск по текстовому запросу (text→video retrieval) на базе XCLIP.

## Структура
- `main.py` — простой запуск single/batch
- `run_one_video.py` — функция `process_one_video(...)` (один файл: кэш → индекс → retrieval → сохранение → (опц.) экспорт клипов)
- `batch_run.py` — прогон по папке `Videos/` (использует `process_one_video(...)`)
- `xclip_wrapper.py` — загрузка модели + энкодинг
- `index_video.py` — нарезка видео на клипы и индексация
- `retrieve.py` — расчёт сходства и top-k
- `cache_io.py` — кэш индекса/результатов

## Кэш
Пути к кэшу задаются в `config.py` (`CachePaths`):
- `./cache/index` — индекс видео (torch save)
- `./cache/results` — результаты запроса (json)
- `./cache/run_meta` — мета/last_run и т.п.

## Запуск: один файл (single)
### Вариант 1: аргументами
```bash
python main.py --video "./Videos/video.mp4" --query "The man left the frame" --show
