from flask import Flask, render_template
import threading
import webbrowser

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/", new=1)

if __name__ == "__main__":
    # откроем браузер чуть позже, чтобы сервер успел стартовать
    threading.Timer(0.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
