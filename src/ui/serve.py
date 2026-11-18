"""Serve the PathSnap web UI (Person 4). Runs on Laptop 2.

The UI is a static page (src/ui/web/index.html) that calls the FastAPI backend on Laptop 1 from the
browser. This tiny server only hands out those files, plus /config.json telling the page where the API is.
It needs nothing beyond Python and PyYAML: no dataset, no model.

API address, in order of priority:
    1. PATHSNAP_API_URL environment variable
    2. config.yaml -> ui.api_base_url
(The page also accepts ?api=http://<ip>:8000 in its URL, and can be changed in the page itself.)

Run from the repo root:
    python -m src.ui.serve                 # http://localhost:8501
    python -m src.ui.serve --port 9000
"""
import argparse
import json
import os
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"


def api_config() -> dict:
    env = os.environ.get("PATHSNAP_API_URL")
    if env:
        return {"api_base_url": env, "source": "env"}
    try:
        with open(ROOT / "config.yaml") as f:
            return {"api_base_url": yaml.safe_load(f)["ui"]["api_base_url"], "source": "config"}
    except Exception:
        return {"api_base_url": "http://127.0.0.1:8000", "source": "default"}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/config.json":
            body = json.dumps(api_config()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):      # keep Laptop 2's terminal quiet; the API logs live on Laptop 1
        pass


def main():
    ap = argparse.ArgumentParser(description="Serve the PathSnap web UI")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(WEB_DIR)))
    url = f"http://localhost:{args.port}"
    print(f"PathSnap UI on {url}  (API: {api_config()['api_base_url']})  Ctrl+C to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nUI server stopped")


if __name__ == "__main__":
    main()
