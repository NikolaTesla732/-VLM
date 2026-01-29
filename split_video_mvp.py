import os
import cv2


def split_video_mvp(
    input_path: str,
    out_dir: str,
    segment_seconds: float,
    crop: bool,
    *,
    roi: tuple[int, int, int, int] | None = None,  # (x, y, w, h). Если None и crop=True — выберешь мышкой
) -> list[tuple[str, float]]:
    """
    Разбивает видео на части фиксированной длительности.

    Возвращает:
        list[(out_path, start_time_seconds)]
        где start_time_seconds — время начала сегмента в исходном видео (в секундах).
    """

    # Создаём выходную папку, если её ещё нет
    os.makedirs(out_dir, exist_ok=True)

    # Открываем видеофайл
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    # Получаем FPS (кадров/сек). Если FPS некорректен — подставляем дефолт
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    # Сколько кадров должно быть в одном сегменте (округляем до ближайшего целого)
    frames_per_seg = int(round(fps * segment_seconds))
    if frames_per_seg < 1:
        cap.release()
        raise ValueError("segment_seconds слишком маленький (получилось < 1 кадра)")

    # Читаем первый кадр:
    # 1) проверяем что видео реально читается
    # 2) узнаём размеры кадра
    # 3) при crop=True можем выбрать ROI (область обрезки)
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return []

    # Исходные размеры кадров (высота, ширина)
    src_h, src_w = first_frame.shape[:2]

    # -------------------- ROI (обрезка) --------------------
    # Если crop=True, то будем сохранять только прямоугольную область (ROI).
    if crop:
        if roi is None:
            # Если ROI не задан — даём выбрать прямоугольник мышкой.
            # Управление: выделить прямоугольник -> Enter/Space подтвердить, Esc отменить
            r = cv2.selectROI(
                "Select ROI (Enter/Space to confirm, Esc to cancel)",
                first_frame,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyWindow("Select ROI (Enter/Space to confirm, Esc to cancel)")
            x, y, w, h = map(int, r)
        else:
            # ROI задан параметром
            x, y, w, h = roi

        # Валидируем ROI: размеры должны быть положительными
        if w <= 0 or h <= 0:
            cap.release()
            raise ValueError("ROI не выбран или выбран некорректно (w/h должны быть > 0)")

        # ROI должен полностью лежать внутри кадра
        if x < 0 or y < 0 or x + w > src_w or y + h > src_h:
            cap.release()
            raise ValueError("ROI выходит за границы кадра")

        # Размеры выходного видео равны размерам ROI
        out_w, out_h = w, h
    else:
        # Без обрезки: ROI = весь кадр
        x = y = 0
        out_w, out_h = src_w, src_h

    # Выбираем кодек для записи mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Список результата: (путь к сегменту, время старта сегмента в исходном видео)
    segments: list[tuple[str, float]] = []

    # Текущий writer (куда пишем кадры), индексы кадров/сегментов
    writer = None
    frame_idx = 0  # индекс текущего кадра в исходном видео (0-based)
    seg_idx = 0    # индекс текущего сегмента (0-based)

    def process_frame(frame):
        """
        Приводит кадр к ожидаемому размеру (если внезапно отличается),
        затем обрезает по ROI, если crop=True.
        """
        if frame is None:
            return None

        # Если размер кадра отличается от размера первого кадра — ресайзим
        if frame.shape[1] != src_w or frame.shape[0] != src_h:
            frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_AREA)

        # Обрезаем ROI, если включено
        if crop:
            frame = frame[y : y + out_h, x : x + out_w]

        return frame

    def open_new_segment(start_frame_idx: int):
        """
        Закрывает предыдущий файл (если был), открывает новый файл сегмента.
        Также добавляет кортеж (out_path, start_time) в итоговый список:
            start_time = start_frame_idx / fps
        """
        nonlocal writer, seg_idx

        # Закрываем предыдущий writer (если уже писали в файл)
        if writer is not None:
            writer.release()

        # Имя файла сегмента
        out_path = os.path.join(out_dir, f"part_{seg_idx:03d}.mp4")

        # Открываем writer для нового сегмента
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Не удалось создать файл: {out_path}")

        # Время старта сегмента (секунды от начала видео)
        start_time_sec = round(start_frame_idx / fps, 1)

        # Сохраняем в список (путь, время старта)
        segments.append((out_path, start_time_sec))

        # Увеличиваем индекс сегмента для следующего файла
        seg_idx += 1

    # -------------------- Запись кадра №0 --------------------
    # Мы уже прочитали first_frame — обработаем его как кадр 0
    frame0 = process_frame(first_frame)
    if frame0 is None:
        cap.release()
        return []

    # Открываем первый сегмент, он стартует с кадра 0
    open_new_segment(start_frame_idx=0)

    # Пишем кадр 0 в файл
    writer.write(frame0)

    # Следующий кадр будет иметь индекс 1
    frame_idx = 1

    # -------------------- Основной цикл по кадрам --------------------
    while True:
        # Читаем очередной кадр
        ret, frame = cap.read()
        if not ret or frame is None:
            # Видео закончилось или ошибка чтения
            break

        # Если текущий кадр — первый кадр нового сегмента, открываем новый файл
        if frame_idx % frames_per_seg == 0:
            open_new_segment(start_frame_idx=frame_idx)

        # Приводим кадр к нужному размеру и обрезаем ROI (если надо)
        frame_out = process_frame(frame)
        if frame_out is None:
            break

        # Записываем кадр в текущий сегмент
        writer.write(frame_out)

        # Переходим к следующему кадру
        frame_idx += 1

    # -------------------- Освобождение ресурсов --------------------
    if writer is not None:
        writer.release()
    cap.release()

    # Возвращаем список (путь сегмента, время старта)
    return segments


if __name__ == "__main__":
    print(
        split_video_mvp(
            r"C:\Users\user\Downloads\rrr.mp4",
            r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split",
            10,
            False,
        )
    )
