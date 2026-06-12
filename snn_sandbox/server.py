from __future__ import annotations

import argparse
from collections import deque
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .model import ArithmeticSNNConfig, ArithmeticSTDPSNN


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
STATE_DIR = ROOT / "states"


class TrainerRuntime:
    """Owns the CUDA SNN and publishes only cheap latest live metrics."""

    def __init__(self, device: str, allow_cpu: bool) -> None:
        self.model = ArithmeticSTDPSNN(ArithmeticSNNConfig(), device=device, allow_cpu=allow_cpu)
        self.running = False
        self.episodes_per_tick = 8
        self.metrics_seq = 0
        self.live_metrics: dict[str, Any] = {}
        self.live_metrics_json = "{}"
        self.metrics_history: deque[dict[str, float]] = deque(maxlen=240)
        self.model_lock = threading.RLock()
        self.control_lock = threading.RLock()
        self.metrics_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.last_rate_time = time.monotonic()
        self.last_rate_trials = 0
        self.trials_per_second = 0.0
        self.worker = threading.Thread(target=self._training_loop, name="snn-trainer", daemon=True)
        self.publisher = threading.Thread(target=self._metrics_loop, name="snn-live-metrics", daemon=True)
        self._publish_metrics()
        self.worker.start()
        self.publisher.start()

    def _training_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.control_lock:
                running = self.running
                episodes_remaining = self.episodes_per_tick
            while running and episodes_remaining > 0 and not self.stop_event.is_set():
                chunk = 1
                with self.control_lock:
                    running = self.running
                    if not running:
                        break
                with self.model_lock:
                    self.model.train_episodes(chunk)
                episodes_remaining -= chunk
            time.sleep(0 if running else 0.04)

    def _metrics_loop(self) -> None:
        interval = 1.0 / 20.0
        while not self.stop_event.is_set():
            started = time.monotonic()
            self._publish_metrics()
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

    def _publish_metrics(self) -> dict[str, Any]:
        self.metrics_seq += 1
        with self.control_lock:
            running = self.running
            episodes_per_tick = self.episodes_per_tick
        with self.model_lock:
            trials = int(self.model.trials)
            correct = int(self.model.correct)
            last_result = dict(self.model.last_result)
            device = str(self.model.device)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            architecture = {
                "inputs": self.model.config.input_neurons,
                "hidden": self.model.config.hidden_neurons,
                "outputs": self.model.config.output_neurons,
                "episodeSteps": self.model.config.episode_steps,
                "learning": "input->hidden local STDP; hidden->output reward-modulated STDP; no autograd/backprop",
            }
        accuracy = correct / trials if trials else 0.0
        now = time.monotonic()
        elapsed = now - self.last_rate_time
        if elapsed >= 0.5:
            self.trials_per_second = (trials - self.last_rate_trials) / elapsed
            self.last_rate_trials = trials
            self.last_rate_time = now
        self.metrics_history.append({"trial": trials, "accuracy": round(accuracy, 4), "time": time.time() * 1000.0})
        packet = {
            "seq": self.metrics_seq,
            "running": running,
            "device": device,
            "gpuName": gpu_name,
            "episodesPerTick": episodes_per_tick,
            "trials": trials,
            "trialsPerSecond": round(self.trials_per_second, 1),
            "accuracy": round(accuracy, 4),
            "lastResult": last_result,
            "architecture": architecture,
            "history": list(self.metrics_history),
            "publishedAt": time.time() * 1000.0,
        }
        packet_json = json.dumps(packet, separators=(",", ":"))
        with self.metrics_lock:
            self.live_metrics = packet
            self.live_metrics_json = packet_json
        return packet

    def latest_metrics(self) -> dict[str, Any]:
        with self.metrics_lock:
            return dict(self.live_metrics)

    def latest_metrics_event(self) -> tuple[int, str]:
        with self.metrics_lock:
            return int(self.live_metrics.get("seq", 0)), self.live_metrics_json

    def command(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.control_lock:
            if name == "start":
                self.running = True
            elif name == "stop":
                self.running = False
            elif name == "speed":
                self.episodes_per_tick = max(1, min(256, int(payload.get("episodesPerTick", self.episodes_per_tick))))
            elif name == "reset":
                self.running = False
                with self.model_lock:
                    seed = payload.get("seed")
                    self.model.reset(seed=int(seed) if seed is not None else None)
                    self.metrics_history.clear()
            elif name == "save":
                with self.model_lock:
                    name_value = str(payload.get("name") or f"snn-state-{int(time.time())}")
                    safe_name = "".join(ch for ch in name_value if ch.isalnum() or ch in ("-", "_")).strip("-_")
                    self.model.save(STATE_DIR / f"{safe_name or 'snn-state'}.pt")
            elif name == "load":
                file_name = str(payload["name"])
                path = (STATE_DIR / file_name).resolve()
                if STATE_DIR.resolve() not in path.parents:
                    raise ValueError("State path must stay inside states/")
                self.running = False
                with self.model_lock:
                    self.model.load(path)
                    self.metrics_history.clear()
            else:
                raise ValueError(f"Unknown command: {name}")
        return self._publish_metrics()

    def pull_current(self) -> dict[str, Any]:
        started = time.perf_counter()
        with self.model_lock:
            config = self.model.config
            recent = torch.roll(
                self.model.recent_spikes,
                shifts=-int(self.model.spike_cursor % self.model.recent_spikes.shape[0]),
                dims=0,
            )
            input_hidden = self.model.input_hidden.detach().cpu().round(decimals=4).tolist()
            hidden_output = self.model.hidden_output.detach().cpu().round(decimals=4).tolist()
            snapshot = {
                "pulledAt": time.time() * 1000.0,
                "trial": int(self.model.trials),
                "correct": int(self.model.correct),
                "accuracy": round(self.model.correct / self.model.trials, 4) if self.model.trials else 0.0,
                "lastResult": dict(self.model.last_result),
                "config": {
                    "inputNeurons": config.input_neurons,
                    "hiddenNeurons": config.hidden_neurons,
                    "outputNeurons": config.output_neurons,
                    "episodeSteps": config.episode_steps,
                    "hiddenThreshold": config.hidden_threshold,
                    "hiddenWinners": config.hidden_winners,
                    "outputThreshold": config.output_threshold,
                    "inputRate": config.input_rate,
                    "backgroundRate": config.background_rate,
                },
                "architecture": {
                    "inputs": config.input_neurons,
                    "hidden": config.hidden_neurons,
                    "outputs": config.output_neurons,
                },
                "weights": {
                    "inputHidden": input_hidden,
                    "hiddenOutput": hidden_output,
                },
                "recentSpikes": recent[:, : min(48, config.hidden_neurons)].detach().cpu().int().tolist(),
                "confusion": self.model.confusion.detach().cpu().int().tolist(),
                "patternCounts": self.model.pattern_counts.detach().cpu().int().tolist(),
                "patternCorrect": self.model.pattern_correct.detach().cpu().int().tolist(),
            }
        snapshot["exportMs"] = round((time.perf_counter() - started) * 1000.0, 3)
        self._publish_metrics()
        snapshot["trainerMetrics"] = self.latest_metrics()
        return snapshot

    def list_states(self) -> list[dict[str, Any]]:
        STATE_DIR.mkdir(exist_ok=True)
        return [
            {"name": path.name, "modified": path.stat().st_mtime, "bytes": path.stat().st_size}
            for path in sorted(STATE_DIR.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]


TRAINER: TrainerRuntime | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SNNSandbox/0.3"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path in {"/app.js", "/styles.css"}:
            content_type = "application/javascript; charset=utf-8" if parsed.path.endswith(".js") else "text/css; charset=utf-8"
            self._send_file(WEB_DIR / parsed.path.lstrip("/"), content_type)
        elif parsed.path == "/api/snapshot":
            self._send_json(self.trainer.latest_metrics())
        elif parsed.path == "/api/states":
            self._send_json({"states": self.trainer.list_states()})
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
            if command == "pull_current":
                response = self.trainer.pull_current()
            else:
                response = self.trainer.command(command, payload)
        except Exception as exc:  # noqa: BLE001 - API should return JSON errors.
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(response)

    @property
    def trainer(self) -> TrainerRuntime:
        if TRAINER is None:
            raise RuntimeError("Trainer runtime is not initialized")
        return TRAINER

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

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
        interval = max(0.02, min(1.0, float(query.get("interval", ["0.05"])[0])))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_seq = -1
        while True:
            seq, packet_json = self.trainer.latest_metrics_event()
            if seq != last_seq:
                try:
                    self.wfile.write(f"data: {packet_json}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
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
    global TRAINER
    args = parse_args()
    TRAINER = TrainerRuntime(device=args.device, allow_cpu=args.allow_cpu)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SNN Sandbox listening on http://{args.host}:{args.port}")
    print(f"Trainer device: {TRAINER.model.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping SNN Sandbox...")
    finally:
        TRAINER.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
