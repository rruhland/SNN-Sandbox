from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from .model import ArithmeticSNNConfig, ArithmeticSTDPSNN


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
STATE_DIR = ROOT / "states"


class TrainingService:
    def __init__(self, device: str, allow_cpu: bool) -> None:
        self.model = ArithmeticSTDPSNN(ArithmeticSNNConfig(), device=device, allow_cpu=allow_cpu)
        self.running = False
        self.episodes_per_tick = 8
        self.seq = 0
        self.snapshot: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._loop, name="snn-training-loop", daemon=True)
        self.worker.start()
        self.refresh_snapshot()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            should_run = self.running
            if should_run:
                with self.lock:
                    self.model.train_episodes(self.episodes_per_tick)
                    self.refresh_snapshot_locked()
                time.sleep(0.001)
            else:
                time.sleep(0.05)

    def refresh_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.refresh_snapshot_locked()

    def refresh_snapshot_locked(self) -> dict[str, Any]:
        self.seq += 1
        self.snapshot = self.model.snapshot(self.running, self.episodes_per_tick, self.seq)
        return self.snapshot

    def latest(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.snapshot)

    def command(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if name == "start":
                self.running = True
            elif name == "stop":
                self.running = False
            elif name == "reset":
                self.running = False
                seed = payload.get("seed")
                self.model.reset(seed=int(seed) if seed is not None else None)
            elif name == "step":
                self.model.train_episodes(int(payload.get("episodes", 1)))
            elif name == "speed":
                self.episodes_per_tick = max(1, min(256, int(payload.get("episodesPerTick", self.episodes_per_tick))))
            elif name == "save":
                name_value = str(payload.get("name") or f"snn-state-{int(time.time())}")
                safe_name = "".join(ch for ch in name_value if ch.isalnum() or ch in ("-", "_")).strip("-_")
                path = STATE_DIR / f"{safe_name or 'snn-state'}.pt"
                self.model.save(path)
            elif name == "load":
                file_name = str(payload["name"])
                path = (STATE_DIR / file_name).resolve()
                if STATE_DIR.resolve() not in path.parents:
                    raise ValueError("State path must stay inside states/")
                self.running = False
                self.model.load(path)
            else:
                raise ValueError(f"Unknown command: {name}")
            return self.refresh_snapshot_locked()

    def list_states(self) -> list[dict[str, Any]]:
        STATE_DIR.mkdir(exist_ok=True)
        return [
            {"name": path.name, "modified": path.stat().st_mtime, "bytes": path.stat().st_size}
            for path in sorted(STATE_DIR.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]


SERVICE: TrainingService | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SNNSandbox/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path in {"/app.js", "/styles.css"}:
            content_type = "application/javascript; charset=utf-8" if parsed.path.endswith(".js") else "text/css; charset=utf-8"
            self._send_file(WEB_DIR / parsed.path.lstrip("/"), content_type)
        elif parsed.path == "/api/snapshot":
            self._send_json(self.service.latest())
        elif parsed.path == "/api/states":
            self._send_json({"states": self.service.list_states()})
        elif parsed.path == "/api/stream":
            self._stream()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        command = parsed.path.removeprefix("/api/")
        payload = self._read_json()
        try:
            snapshot = self.service.command(command, payload)
        except Exception as exc:  # noqa: BLE001 - API should return JSON errors.
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(snapshot)

    @property
    def service(self) -> TrainingService:
        if SERVICE is None:
            raise RuntimeError("Training service is not initialized")
        return SERVICE

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        interval = max(0.02, min(1.0, float(query.get("interval", ["0.08"])[0])))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_seq = -1
        while True:
            snapshot = self.service.latest()
            seq = int(snapshot.get("seq", 0))
            if seq != last_seq:
                message = f"data: {json.dumps(snapshot)}\n\n".encode("utf-8")
                try:
                    self.wfile.write(message)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_seq = seq
            time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the desktop GPU SNN sandbox server.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to expose on LAN/Tailscale.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for smoke tests.")
    return parser.parse_args()


def main() -> None:
    global SERVICE
    args = parse_args()
    SERVICE = TrainingService(device=args.device, allow_cpu=args.allow_cpu)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SNN Sandbox listening on http://{args.host}:{args.port}")
    print(f"Model device: {SERVICE.model.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping SNN Sandbox...")
    finally:
        SERVICE.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
