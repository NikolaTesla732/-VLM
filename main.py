import os
import sys
import time
import uuid
import socket
import threading
import webbrowser
import atexit
import signal
from flask import Flask, render_template, request


def is_frozen() -> bool:
    """True, если запущено как PyInstaller exe."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def runtime_dir() -> str:
    """
    Где хранить логи/загрузки:
    - .py: рядом с main.py
    - .exe: рядом с main.exe
    """
    return os.path.dirname(sys.executable) if is_frozen() else os.path.dirname(os.path.abspath(__file__))


def resource_dir(name: str) -> str:
    """
    Где искать папки templates/static:
    1) рядом с exe/скриптом (можно менять без пересборки, если файлы лежат рядом)
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

UPLOAD_DIR = os.path.join(runtime_dir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Очистка uploads при закрытии приложения ---
_CREATED_UPLOADS = set()


def cleanup_uploads() -> None:
    for path in list(_CREATED_UPLOADS):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


atexit.register(cleanup_uploads)


def _handle_exit(signum, frame):
    cleanup_uploads()
    raise SystemExit


signal.signal(signal.SIGINT, _handle_exit)
try:
    signal.signal(signal.SIGTERM, _handle_exit)
except Exception:
    pass
# ---------------------------------------------


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
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)
    _CREATED_UPLOADS.add(save_path)

    # ВСТАВЬ СВОЙ КОД ОБРАБОТКИ ВОТ ЗДЕСЬ
    # process_video(save_path, query_text=query_text)

    return f"Ок! Сохранено: {safe_name}. Текст: {query_text or '(пусто)'}"


def open_browser_later(url: str) -> None:
    """Небольшая задержка, чтобы сервер успел подняться, и открываем браузер."""
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
