# 🕵️ Сыщик норм — Video Retrieval (XCLIP)

Проект для **поиска “интересных” моментов в видео** и **ранжирования фрагментов по текстовому запросу** (*text → video retrieval*).

## 🔥 Что делает проект

Основной пайплайн:

1) Берём исходные видео  
2) Нарезаем на сегменты фиксированной длины  
3) Фильтруем “неинтересные” сегменты (движение / человек в кадре)  
4) Считаем эмбеддинги сегментов (XCLIP / CLIP backend)  
5) По текстовому запросу сортируем сегменты по similarity

---

## 🚀 Возможности

- ✅ Нарезка видео на сегменты (`segment_seconds`)
- ✅ Motion filter (движение в кадре)
- ✅ Person filter (человек в кадре через YOLO, COCO class=0)
- ✅ Индексация сегментов (эмбеддинги)
- ✅ Поиск по тексту: *“Vandalism”, “Person running”, “Car crash”…*
- ✅ Кеширование промежуточных результатов (ускоряет повторные запуски)
- ✅ Поддержка ROI (область интереса)

---

## 📦 Структура проекта

```
.
├── main.py                    # основной вход
├── batch_run.py               # обработка папки с видео (batch)
├── run_one_video.py           # обработка одного видео
├── index_video.py             # индексация сегментов
├── retrieve.py                # ранжирование по текстовому запросу
├── motion_detect.py           # фильтр движения
├── person_detect.py           # фильтр человека (YOLO)
├── visualize_results.py       # визуализация результатов
├── cache_io.py                # кеш директории / чтение/запись кеша
├── config.py                  # CLI + конфиги
├── defaults.py                # утилиты подстановки значений
└── models/
    ├── base.py                # интерфейс backend'ов
    ├── clip_frame_backend.py  # frame-based CLIP backend
    ├── xclip_backend.py       # video-text XCLIP backend
    ├── dummy_backend.py       # заглушка
    └── registry.py            # регистрация backend'ов
```

---

## ⚙️ Установка

### 1) Python окружение
Рекомендуется Python **3.9+**.

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2) Зависимости
По импорту в проекте используются:

- `opencv-python` (`cv2`)
- `numpy`
- `torch`
- `transformers`
- `ultralytics` (YOLO, для `person_detect`)
- `tqdm`

Установка (примерно):

```bash
pip install opencv-python numpy torch transformers tqdm ultralytics
```

⚠️ Если у тебя CUDA — ставь PyTorch под свою версию CUDA.

---

## ▶️ Запуск

### Batch (папка с видео/сегментами)
```bash
python main.py --mode batch --video video_split --query "Vandalism"
```

Где:
- `--mode batch` — обработка набора видео
- `--video video_split` — папка с видео (или сегментами)
- `--query "..."` — текстовый запрос

---

## 🧩 Как работает пайплайн (подробнее)

### 1) Нарезка видео
Каждое видео делится на сегменты по `segment_seconds`.

### 2) Фильтр движения (motion_detect)
Сегмент считается “интересным”, если в нём есть движение, которое длится не меньше `min_motion_seconds`.

### 3) Фильтр человека (person_detect)
Используется детектор (YOLO).  
Если человек найден (COCO class=0) → сегмент остаётся.

### 4) Индексация сегментов
Для каждого сегмента считается embedding (вектор признаков).  
Это делается через backend из `models/`.

### 5) Поиск по тексту
Текстовый запрос переводится в embedding, затем считается similarity между текстом и каждым сегментом.  
Результаты сортируются по убыванию similarity.

---

## 🧠 Backends (модели)

### XCLIP backend (рекомендуется для качества)
Файл: `models/xclip_backend.py`

Плюсы:
- лучше понимает динамику

Минусы:
- тяжелее по ресурсам, особенно на CPU

### Frame-based CLIP backend
Файл: `models/clip_frame_backend.py`

Плюсы:
- быстрее

Минусы:
- хуже работает с динамикой (движение, действия)

---

## 🗂️ Кеширование

Проект использует кеш для:
- сохранения нарезанных сегментов
- сохранения эмбеддингов
- ускорения повторного поиска по новым запросам

Если ты меняешь параметры preprocess — лучше чистить кеш.

---

## 🧪 Примеры запросов

```bash
python main.py --mode batch --video video_split --query "Person running"
python main.py --mode batch --video video_split --query "Fight"
python main.py --mode batch --video video_split --query "Car accident"
python main.py --mode batch --video video_split --query "Vandalism"
```

---

## ❗ Частые проблемы

### “No module named cv2”
```bash
pip install opencv-python
```

### YOLO не работает / не найдены веса
Проверь:
- установлена ли библиотека (`ultralytics`)
- скачаны ли веса
- доступна ли видеокарта (если рассчитываешь на GPU)

### Очень медленно на CPU
Решения:
- отключить `person_detect`
- увеличить `segment_seconds`
- перейти на CLIP frame backend
- использовать CUDA

---

## 📌 Минимальный старт

```bash
python main.py --mode batch --video video_split --query "Vandalism"
```
