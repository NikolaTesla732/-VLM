import os
import sys
import time
import uuid
import socket
import threading
import webbrowser
import tempfile
import atexit
import signal
import mimetypes

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

from split_video import split_video_web


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def runtime_dir() -> str:
    return os.path.dirname(sys.executable) if is_frozen() else os.path.dirname(os.path.abspath(__file__))


def resource_dir(name: str) -> str:
    local = os.path.join(runtime_dir(), name)
    if os.path.isdir(local):
        return local
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


app = Flask(
    __name__,
    template_folder=resource_dir("templates"),
    static_folder=resource_dir("static"),
)

# job_id -> dict
JOBS: dict[str, dict] = {}

# temp files cleanup
_TEMP_FILES: set[str] = set()


def cleanup_temp_files() -> None:
    for p in list(_TEMP_FILES):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


atexit.register(cleanup_temp_files)


def _handle_exit(signum, frame):
    cleanup_temp_files()
    raise SystemExit


signal.signal(signal.SIGINT, _handle_exit)
try:
    signal.signal(signal.SIGTERM, _handle_exit)
except Exception:
    pass


# -------------------- PROCESSING --------------------

def process_job(job_id: str) -> None:
    """
    Тут запускается обработка уже ПОСЛЕ того как пользователь выбрал ROI.
    """
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        temp_path = job["temp_path"]
        roi = job["roi"]            # (x,y,w,h) в пикселях
        query_text = job["query_text"]

        # ==== ПРИМЕР: тут вставь твою реальную логику ====
        # out_dir = os.path.join(runtime_dir(), "video_split_out", job_id)
        # segments = split_video_web(temp_path, out_dir, 10.0, roi=roi)
        # job["segments"] = segments

        # Заглушка (проверка интерфейса):
        time.sleep(2)
        split_video_web(temp_path, 'split_video', roi = roi)
        job["timestamps"] = [12.5, 38.0, 76.2]  # секунды (потом заменишь на свои таймкоды)
        job["result"] = f"Готово. Запрос: {query_text or '(пусто)'}"

        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# -------------------- ROUTES --------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run():
    """
    1) сохраняем видео в TEMP
    2) создаём job со статусом await_roi
    3) редиректим на страницу /roi/<job_id>
    """
    file = request.files.get("video_file")
    query_text = request.form.get("query_text", "").strip()

    if not file or not file.filename:
        return "Файл не выбран", 400

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    fd, temp_path = tempfile.mkstemp(prefix="detective_", suffix=ext)
    os.close(fd)
    file.save(temp_path)
    _TEMP_FILES.add(temp_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "await_roi",  # ждём ROI
        "result": "",
        "error": "",
        "temp_path": temp_path,
        "timestamps": [],
        "roi": None,            # (x,y,w,h) пиксели
        "query_text": query_text,
    }

    return redirect(url_for("roi", job_id=job_id))


@app.get("/roi/<job_id>")
def roi(job_id: str):
    """
    Страница выбора области (ROI) поверх видео.
    """
    job = JOBS.get(job_id)
    if not job:
        return "Задача не найдена", 404

    return render_template("roi.html", job_id=job_id)


@app.post("/roi/<job_id>")
def roi_submit(job_id: str):
    """
    Получаем ROI (x,y,w,h) и запускаем обработку.
    """
    job = JOBS.get(job_id)
    if not job:
        return "Задача не найдена", 404

    try:
        x = int(float(request.form.get("x", "0")))
        y = int(float(request.form.get("y", "0")))
        w = int(float(request.form.get("w", "0")))
        h = int(float(request.form.get("h", "0")))
    except ValueError:
        return "ROI некорректен", 400

    if w <= 0 or h <= 0:
        return "ROI некорректен", 400

    job["roi"] = (x, y, w, h)
    job["status"] = "processing"

    threading.Thread(target=process_job, args=(job_id,), daemon=True).start()
    return redirect(url_for("processing", job_id=job_id))


@app.get("/processing/<job_id>")
def processing(job_id: str):
    if job_id not in JOBS:
        return "Задача не найдена", 404
    return render_template("processing.html", job_id=job_id)


@app.get("/api/status/<job_id>")
def api_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": job["status"]})


@app.get("/result/<job_id>")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return "Задача не найдена", 404
    return render_template("result.html", job=job, job_id=job_id)


@app.get("/video/<job_id>")
def video(job_id: str):
    """
    Отдаём видео браузеру для просмотра на result/roi страницах.
    """
    job = JOBS.get(job_id)
    if not job:
        return "Видео не найдено", 404

    path = job.get("temp_path")
    if not path or not os.path.isfile(path):
        return "Видео не найдено", 404

    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "video/mp4", conditional=True)


# -------------------- RUN --------------------

def open_browser_later(url: str) -> None:
    time.sleep(0.7)
    webbrowser.open(url, new=1)


def main():
    host = "127.0.0.1"
    port = find_free_port(host)
    url = f"http://{host}:{port}/"

    threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
