from __future__ import annotations

import argparse
import atexit
import errno
import json
import mimetypes
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .models import CONTEXT, DEFAULT_URL, MAX_HEADER, download, inspect_url, normalize_url, parse_header, validate

ROOT = Path(__file__).resolve().parent.parent


class Bridge:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.lines = queue.Queue()

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
            self.process = None

    def start(self):
        self.stop()
        binary = ROOT / "build/bin/kavin-engine"
        if not binary.exists():
            raise ValueError("Inference engine is not built. Run python3 scripts/setup.py first.")
        self.lines = queue.Queue()
        log = (ROOT / "engine.log").open("a")
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=log, text=True, bufsize=1)
        log.close()
        process, lines = self.process, self.lines

        def read():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        threading.Thread(target=read, daemon=True).start()

    def call(self, command):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                if command["op"] != "load":
                    raise ValueError("Inference engine stopped. Reload the model.")
                self.start()
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
            try:
                line = self.lines.get(timeout=180)
            except queue.Empty as error:
                self.stop()
                raise ValueError("Inference timed out after three minutes. Try a smaller model.") from error
            if line is None:
                self.stop()
                raise ValueError("Inference engine exited. See engine.log; try a smaller supported model.")
            result = json.loads(line)
            if not result.get("ok"):
                raise ValueError(result.get("error", "Inference failed"))
            return result


class App:
    def __init__(self):
        self.bridge = Bridge()
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {"phase": "empty", "message": "Choose a pretrained GGUF", "progress": 0,
                      "model": None, "default_url": DEFAULT_URL, "engine_ready": (ROOT / "build/bin/kavin-engine").exists()}
        self.inspected = {}
        self.trace = None
        self.prompt = None
        self.source = None
        atexit.register(self.bridge.stop)

    def update(self, **fields):
        with self.state_lock:
            self.state.update(fields)

    def status(self):
        with self.state_lock:
            return dict(self.state)

    def load(self, url=None, local=None):
        if not self.lock.acquire(blocking=False):
            raise ValueError("An inference or loading operation is already running.")
        self.update(phase="validating", message="Inspecting GGUF metadata", progress=0, model=None, error=None)
        self.trace = None
        self.prompt = None

        def work():
            try:
                if local:
                    path = Path(local).resolve()
                    with path.open("rb") as file:
                        validate(parse_header(file.read(MAX_HEADER)), path.stat().st_size)
                    self.source = {"file": path.name, "url": None}
                else:
                    info = inspect_url(url)
                    self.update(phase="downloading", message="Downloading GGUF", progress=0)
                    path = download(info, ROOT / "models", lambda p, m: self.update(progress=p, message=m))
                    self.source = {"file": info["filename"], "url": info["url"]}
                self.update(phase="loading", message="Loading model and testing real tensor capture", progress=1)
                result = self.bridge.call({"op": "load", "path": str(path), "context": CONTEXT})
                self.update(phase="ready", message="Model ready · inference check passed", model=result["model"], source=self.source)
            except Exception as error:
                self.bridge.stop()
                self.update(phase="error", message=str(error), error=str(error), model=None)
            finally:
                self.lock.release()
        threading.Thread(target=work, daemon=True).start()

    def infer(self, request):
        if not self.lock.acquire(blocking=False):
            raise ValueError("An inference or loading operation is already running.")
        try:
            if self.status()["phase"] != "ready":
                raise ValueError("Load a model first.")
            if request.get("op") not in ("prefill", "step", "view"):
                raise ValueError("Unknown inference operation.")
            result = self.bridge.call(request)
            if request["op"] == "prefill":
                self.prompt = request["prompt"]
            result["prompt"] = self.prompt
            result["source"] = self.source
            result["format"] = "kavin-trace-v1"
            result["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.trace = result
            return result
        except Exception:
            self.trace = None
            raise
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type="application/json"):
        raw = json.dumps(data).encode() if content_type == "application/json" else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    @property
    def app(self):
        return self.server.app

    def valid_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {"error": "Local connections only"})
        path = urlsplit(self.path).path
        if path == "/api/status":
            return self.reply(200, self.app.status())
        if path == "/api/trace":
            return self.reply(200 if self.app.trace else 404, self.app.trace or {"error": "No trace yet"})
        files = {"/": "index.html", "/app.js": "app.js", "/matrix_proxy.js": "matrix_proxy.js", "/style.css": "style.css", "/favicon.svg": "favicon.svg", "/fonts/ArchitectsDaughter-Regular.ttf": "fonts/ArchitectsDaughter-Regular.ttf"}
        if path not in files:
            return self.reply(404, {"error": "Not found"})
        file = ROOT / "web" / files[path]
        return self.reply(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")

    def do_POST(self):
        try:
            if not self.valid_host():
                return self.reply(403, {"error": "Local connections only"})
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                return self.reply(403, {"error": "Cross-origin requests are not allowed"})
            if self.headers.get_content_type() != "application/json":
                return self.reply(415, {"error": "Use application/json"})
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 32768:
                return self.reply(413, {"error": "Request is too large or empty"})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/inspect":
                return self.reply(200, inspect_url(data["url"]))
            if self.path == "/api/load":
                self.app.load(url=normalize_url(data["url"]))
                return self.reply(202, {"ok": True})
            if self.path == "/api/session":
                return self.reply(200, self.app.infer(data))
            return self.reply(404, {"error": "Not found"})
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self.reply(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description="KeVin - Key–Value Cache Inspection Tool")
    parser.add_argument("--port", type=int, default=8765, help="Local port; use 0 to select a free port automatically")
    parser.add_argument("--model", type=Path, help="Load a local GGUF on startup (same compatibility checks apply)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        parser.exit(2, f"Port {args.port} is already in use. KeVin may already be running.\n"
                       f"Open http://127.0.0.1:{args.port}\n"
                       "For a separate instance on a free port, run: python3 -m kavin --port 0\n")
    server.daemon_threads = True
    server.app = App()
    if args.model:
        server.app.load(local=args.model)
    print(f"KeVin is running at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.app.bridge.stop(); server.server_close()


if __name__ == "__main__":
    main()
