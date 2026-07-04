# window.py
import webview
import time
import requests

def wait_for_server(url, timeout=10):
    for _ in range(timeout):
        try:
            requests.get(url)
            return True
        except:
            time.sleep(1)
    return False

if wait_for_server("http://127.0.0.1:8501"):
    window = webview.create_window(
        "Lucrum Private AI",
        "http://127.0.0.1:8501",
        width=1080,
        height=720,
        text_select=True
    )
    webview.start()
else:
    print("Server not responding.")
