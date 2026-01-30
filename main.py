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

from flask import Flask, render_template, request, redirect, url_for, jsonify


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

# --- TEMP files cleanup on exit ---
_TEMP_FILES = set()


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
# --------------------------------

# job_id -> {"status": "processing|done|error", "result": str, "error": str}
JOBS = {}


def process_job(job_id: str, temp_path: str, query_text: str):
    try:
        # ====== ТВОЯ ОБРАБОТКА ВИДЕО ======
        # result = process_video(temp_path, query_text=query_text)

        # пример (заглушка)
        time.sleep(5)
        result = f"Готово! Файл: {os.path.basename(temp_path)} | Запрос: {query_text or '(пусто)'}"

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run():
    file = request.files.get("video_file")
    query_text = request.form.get("query_text", "").strip()

    if not file or not file.filename:
        return "Файл не выбран", 400

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"

    fd, temp_path = tempfile.mkstemp(prefix="detective_", suffix=ext)
    os.close(fd)  # важно для Windows
    file.save(temp_path)
    _TEMP_FILES.add(temp_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "processing", "result": "", "error": ""}

    threading.Thread(target=process_job, args=(job_id, temp_path, query_text), daemon=True).start()

    return redirect(url_for("processing", job_id=job_id))


@app.get("/processing/<job_id>")
def processing(job_id):
    if job_id not in JOBS:
        return "Задача не найдена", 404
    return render_template("processing.html", job_id=job_id)


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": job["status"]})


@app.get("/result/<job_id>")
def result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return "Задача не найдена", 404
    return render_template("result.html", job=job)


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
