import cv2
import math


def motion_detect(
    video_path: str,
    min_motion_seconds: float,
    *,
    min_area: int = 30,          # минимальная площадь "пятна движения", чтобы отсечь шум
    threshold_value: int = 13,   # порог отличия (0..255)
    blur_ksize: int = 21,        # размер размытия (должен быть нечётным и >= 3)
    max_gap_frames: int = 20     # сколько кадров подряд можно "потерять" движение, не обрывая серию
) -> bool:
    """
    Ищет СЕРИЮ длительностью >= min_motion_seconds, где:
      - серия начинается с кадра, на котором обнаружено движение
      - затем серия может продолжаться даже если движение пропадает,
        но не более чем на max_gap_frames подряд
      - ВАЖНО: в длительность серии входят и "провальные" кадры (в пределах допуска).
    """

    # --- Валидация параметров ---
    if min_motion_seconds <= 0:
        raise ValueError("min_motion_seconds должен быть > 0")
    if min_area < 1:
        raise ValueError("min_area должен быть >= 1")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames должен быть >= 0")
    if not (0 <= threshold_value <= 255):
        raise ValueError("threshold_value должен быть в диапазоне 0..255")
    if blur_ksize < 3:
        raise ValueError("blur_ksize должен быть >= 3")
    if blur_ksize % 2 == 0:
        blur_ksize += 1  # делаем нечётным

    # --- Открываем видео ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # --- FPS ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not math.isfinite(float(fps)):
        fps = 30.0

    # --- Сколько кадров должна длиться серия ---
    need_frames = int(round(fps * min_motion_seconds))
    if need_frames < 1:
        cap.release()
        raise ValueError("min_motion_seconds слишком маленький (получилось < 1 кадра)")

    # --- Устойчивое преобразование в серый ---
    def to_gray(frame):
        if frame is None:
            return None
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            ch = frame.shape[2]
            if ch == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ch == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Первый кадр ---
    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        return False

    prev_gray = to_gray(prev)
    if prev_gray is None:
        cap.release()
        return False

    prev_gray = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)

    # --- Счётчики серии ---
    in_series = False     # находимся ли сейчас внутри серии
    series_len = 0        # длина текущей серии в кадрах (включая допустимые провалы)
    gap_run = 0           # сколько кадров подряд нет движения внутри серии

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = to_gray(frame)
        if gray is None:
            break
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # --- Дельта между кадрами ---
        diff = cv2.absdiff(prev_gray, gray)

        # --- Порог и "раздувание" пятен ---
        _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # --- Контуры (совместимо с OpenCV 3/4) ---
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        # --- Есть ли движение на текущем кадре ---
        motion_now = any(cv2.contourArea(c) >= min_area for c in contours)

        if not in_series:
            # Серия ещё не началась — стартуем только при первом обнаружении движения
            if motion_now:
                in_series = True
                series_len = 1   # серия начинается с текущего кадра
                gap_run = 0
        else:
            # Мы внутри серии: длину серии увеличиваем на КАЖДОМ кадре,
            # пока провал не превысит max_gap_frames
            series_len += 1

            if motion_now:
                # движение вернулось — сбрасываем провал
                gap_run = 0
            else:
                # движения нет — наращиваем провал
                gap_run += 1
                if gap_run > max_gap_frames:
                    # провал слишком длинный — серия обрывается и сбрасывается
                    in_series = False
                    series_len = 0
                    gap_run = 0

        # Если серия идёт и достигла нужной длины — успех
        if in_series and series_len >= need_frames:
            cap.release()
            return True

        # Обновляем "предыдущий" кадр
        prev_gray = gray

    cap.release()
    return False


if __name__ == "__main__":
    print(motion_detect(r"C:\Users\user\Documents\Prodjeeeect\all_func\video_split\part_003.mp4", 4))
