import os
import sys
import time
import socket
import threading
import webbrowser
import tempfile
import atexit
import signal
from flask import Flask, render_template, request


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
    Где искать templates/static:
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


app = Flask(
    __name__,
    template_folder=resource_dir("templates"),
    static_folder=resource_dir("static"),
)

# --- TEMP: сохраняем загруженные видео во временную папку ОС и чистим при закрытии ---
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
# -------------------------------------------------------------------------------


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run():
    file = request.files.get("video_file")
    query_text = request.form.get("query_text", "").strip()

    if not file or not file.filename:
        return "Файл не выбран", 400

    # расширение, чтобы обработчики (opencv/ffmpeg) нормально определяли формат
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"

    # создаём временный файл и получаем путь
    fd, temp_path = tempfile.mkstemp(prefix="detective_", suffix=ext)
    os.close(fd)  # важно закрыть дескриптор на Windows

    # сохраняем загруженное видео в temp
    file.save(temp_path)
    _TEMP_FILES.add(temp_path)

    # --- ТУТ ТВОЯ ОБРАБОТКА ---
    # process_video(temp_path, query_text=query_text)

    return f"Ок! Временный файл сохранён: {os.path.basename(temp_path)}. Запрос: {query_text or '(пусто)'}"


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
