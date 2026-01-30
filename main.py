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

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_file,
)


# -------------------- helpers --------------------

def is_frozen() -> bool:
    """True, если запущено как PyInstaller exe."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def runtime_dir() -> str:
    """
    Где хранить данные/ресурсы:
    - .py: рядом с main.py
    - .exe: рядом с main.exe
    """
    return os.path.dirname(sys.executable) if is_frozen() else os.path.dirname(os.path.abspath(__file__))


def resource_dir(name: str) -> str:
    """
    Где искать папки templates/static:
    1) рядом с exe/скриптом (можно менять без пересборки)
    2) внутри PyInstaller onefile (_MEIPASS)
    """
    local = os.path.join(runtime_dir(), name)
    if os.path.isdir(local):
        return local
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def find_free_port(host: str = "127.0.0.1") -> int:
    """Берём свободный порт, чтобы не падать, если 5000 занят."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


# -------------------- app --------------------

app = Flask(
    __name__,
    template_folder=resource_dir("templates"),
    static_folder=resource_dir("static"),
)

# job_id -> {
#   "status": "processing|done|error",
#   "result": str,
#   "error": str,
#   "temp_path": str,
#   "timestamps": list[float],  # секунды
# }
JOBS: dict[str, dict] = {}

# TEMP: сохраняем загруженные видео во временную папку ОС и чистим при закрытии
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


# -------------------- video processing --------------------

def process_job(job_id: str, temp_path: str, query_text: str) -> None:
    """
    Фоновая обработка видео.

    ВАЖНО:
    - Сюда вставь свою реальную обработку.
    - Заполняй JOBS[job_id]["timestamps"] списком секунд [12.5, 38, 76.2]
    """
    try:
        # ====== ТУТ ТВОЯ ОБРАБОТКА ======
        # Пример:
        # timestamps = process_video(temp_path, query_text=query_text)
        # JOBS[job_id]["timestamps"] = timestamps
        # JOBS[job_id]["result"] = "Найдены моменты нарушения"

        # Заглушка для проверки интерфейса:
        time.sleep(4)
        JOBS[job_id]["timestamps"] = [12.5, 38.0, 76.2]
        JOBS[job_id]["result"] = f"Готово. Запрос: {query_text or '(пусто)'}"

        JOBS[job_id]["status"] = "done"

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


# -------------------- routes --------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run():
    file = request.files.get("video_file")
    query_text = request.form.get("query_text", "").strip()

    if not file or not file.filename:
        return "Файл не выбран", 400

    # расширение, чтобы opencv/ffmpeg нормально определяли формат
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"

    # создаём временный файл и получаем путь
    fd, temp_path = tempfile.mkstemp(prefix="detective_", suffix=ext)
    os.close(fd)  # важно закрыть дескриптор на Windows

    # сохраняем загруженное видео в temp
    file.save(temp_path)
    _TEMP_FILES.add(temp_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "processing",
        "result": "",
        "error": "",
        "temp_path": temp_path,
        "timestamps": [],
    }

    threading.Thread(
        target=process_job,
        args=(job_id, temp_path, query_text),
        daemon=True
    ).start()

    # редирект на страницу ожидания
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
    Отдаём видео браузеру для просмотра на result.html
    """
    job = JOBS.get(job_id)
    if not job:
        return "Видео не найдено", 404

    path = job.get("temp_path")
    if not path or not os.path.isfile(path):
        return "Видео не найдено", 404

    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "video/mp4", conditional=True)


# -------------------- run server --------------------

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
