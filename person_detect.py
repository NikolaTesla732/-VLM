import cv2
from ultralytics import YOLO


def person_detect(
    video_path: str,
    n_frames: int,
    *,
    sample_every_n_frames: int = 3,  # обрабатываем каждый N-й кадр (ускорение)
    skip_first_n_frames: int = 2,    # допустимое число подряд "пустых" проверенных кадров внутри серии
    model_path: str = "yolov8s.pt",
    imgsz: int = 416,                # размер входа модели (меньше = быстрее, но может хуже детектить)
    conf: float = 0.35,              # порог уверенности (confidence threshold)
    iou: float = 0.5,                # порог IoU для NMS (слияние/удаление дублей боксов)
    max_seconds: float | None = None # ограничение по времени анализа видео
) -> bool:
    """
    Возвращает True, если человек появляется на протяжении n_frames "проверенных" кадров,
    при этом разрешены "провалы" (кадры без человека) длиной до skip_first_n_frames подряд.

    Важно:
    - "подряд" и "пропуски" считаются ТОЛЬКО по кадрам, которые реально прогоняются через модель
      (т.е. после фильтра sample_every_n_frames).
    """

    # --- Валидация входных параметров (защита от некорректных значений) ---
    if n_frames < 1:
        raise ValueError("n_frames должен быть >= 1")
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames должен быть >= 1")
    if skip_first_n_frames < 0:
        raise ValueError("skip_first_n_frames должен быть >= 0")

    # --- Загрузка YOLO модели (весов) ---
    # Важно: сейчас модель загружается при каждом вызове функции.
    # Если функция вызывается часто, выгоднее загрузить модель один раз снаружи и передавать в функцию.
    model = YOLO(model_path)

    # --- Открытие видеофайла/потока ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # --- Получение FPS для расчёта лимита по времени (max_seconds) ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # запасной вариант, если FPS неизвестен/не читается
        fps = 30.0

    # --- Если задан лимит по секундам, переводим его в лимит по кадрам ---
    max_frames = None
    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(max_seconds * fps))

    frame_idx = 0  # индекс текущего кадра (всех кадров, включая пропущенные)

    # --- Переменные логики "серии" ---
    run = 0  # сколько "кадров с человеком" накопили в текущей серии
    gap = 0  # сколько подряд "кадров без человека" внутри текущей серии

    # --- Основной цикл чтения видео ---
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # видео закончилось или кадр не прочитан
            break

        # --- Ограничение по длительности анализа (в кадрах) ---
        if max_frames is not None and frame_idx >= max_frames:
            break

        # --- Ускорение: прогоняем через модель только каждый N-й кадр ---
        # Остальные кадры просто пропускаем.
        if (frame_idx % sample_every_n_frames) != 0:
            frame_idx += 1
            continue

        # --- Запуск детекции YOLO на выбранном кадре ---
        # classes=[0] => детектим только класс "person" (в COCO это класс 0)
        results = model.predict(
            frame,
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        # --- Извлекаем боксы ---
        # results[0] потому что predict возвращает список результатов (на случай батча)
        boxes = results[0].boxes

        # --- Флаг: найден ли человек на текущем кадре ---
        # Если боксов (класса person) больше 0 — значит человек есть.
        person_now = boxes is not None and len(boxes) > 0

        # --- Логика "серии" присутствия человека ---
        if person_now:
            # кадр с человеком: продолжаем/наращиваем серию
            run += 1
            gap = 0
        else:
            # кадр без человека: увеличиваем "дыру" внутри серии
            gap += 1

            # если "дыр" стало больше допустимого — серия обрывается и начинаем заново
            if gap > skip_first_n_frames:
                run = 0
                gap = 0

        # --- Условие успеха: серия достигла нужной длины ---
        if run >= n_frames:
            cap.release()
            return True

        frame_idx += 1

    # --- Если дошли до конца видео/лимита и условие не выполнено ---
    cap.release()
    return False
