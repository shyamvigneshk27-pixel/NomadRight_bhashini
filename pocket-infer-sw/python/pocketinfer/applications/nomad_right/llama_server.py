"""
On-demand llama-server for Qwen3-VL (the llama.cpp server Ollama bundles), owned
by the kiosk process.

Why not Ollama: measured on this Jetson on 2026-09-12 with the identical GGUF and
vision projector, Ollama fixes the vision encoder at ~1,215 image tokens per photo
(prompt evaluation 2.0 s, a fresh photo answered in 5.6 s) and holds a 2.3 GB
runner; the same binary started directly with --image-max-tokens 512 and a q8 KV
cache answers a fresh photo in 2.6 s, text in 1.7 s instead of 2.9 s, and needs
about 1.2 GB less. See benchmarks/qwen_benchmark.py and docs/reports/.

Why the app owns the process: llama-server has no keep-alive, and the 8 GB device
cannot afford the model resident while nobody uses the camera. The manager starts
the server on first use (or on a Camera press - app._prewarm_qwen), keeps it for
constants.LLAMA_SERVER_IDLE_STOP_S after the last request, then stops it. No root
needed; the server dies with the kiosk.

Pitfall carried over from the benchmark: ggml finds its CUDA backend only through
GGML_BACKEND_PATH (the .so file, not the directory); with LD_LIBRARY_PATH alone it
silently runs on the CPU at 10 tok/s.
"""

import atexit
import json
import logging
import os
import subprocess
import threading
import time
from typing import List, Optional

import requests

from pocketinfer.applications.nomad_right import constants

logger = logging.getLogger(__name__)

_OLLAMA_MODELS = "/usr/share/ollama/.ollama/models"


def resolve_model_files() -> (str, str):
    """GGUF and vision projector paths: from Ollama's manifest for the configured
    model (content-addressed blobs, so a re-pull changes the hashes), falling back
    to the paths recorded in constants."""
    try:
        name, tag = constants.LLM_FALLBACK_MODEL.rsplit(":", 1)
        with open(os.path.join(_OLLAMA_MODELS, "manifests", name, tag)) as f:
            layers = json.load(f)["layers"]
        by_type = {l["mediaType"].rsplit(".", 1)[-1]: l["digest"].replace(":", "-") for l in layers}
        gguf = os.path.join(_OLLAMA_MODELS, "blobs", by_type["model"])
        mmproj = os.path.join(_OLLAMA_MODELS, "blobs", by_type["projector"])
        if os.path.exists(gguf) and os.path.exists(mmproj):
            return gguf, mmproj
    except Exception as exc:
        logger.debug(f"manifest lookup failed ({exc}); using the recorded blob paths")
    return constants.LLAMA_SERVER_GGUF, constants.LLAMA_SERVER_MMPROJ


class LlamaServerManager:
    """One llama-server per kiosk process, started lazily, stopped after idling."""

    _lock = threading.Lock()
    _shared: Optional["LlamaServerManager"] = None

    @classmethod
    def shared(cls) -> "LlamaServerManager":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def __init__(self):
        self.port = constants.LLAMA_SERVER_PORT
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None
        self.log_file = None
        self.last_use = 0.0
        self.started_at = 0.0
        self._start_lock = threading.Lock()
        self._watch = None
        atexit.register(self.stop)

    # ── lifecycle ────────────────────────────────────────────────────────

    def command(self) -> List[str]:
        gguf, mmproj = resolve_model_files()
        args = [constants.LLAMA_SERVER_BIN, "-m", gguf, "--mmproj", mmproj,
                "-ngl", "99", "-c", str(constants.LLM_NUM_CTX), "-t", str(constants.LLM_NUM_THREAD),
                "--port", str(self.port), "--host", "127.0.0.1", "-np", "1", "--no-webui", "-fa", "auto",
                "--image-min-tokens", str(constants.LLAMA_SERVER_IMAGE_MIN_TOKENS),
                "--image-max-tokens", str(constants.LLAMA_SERVER_IMAGE_MAX_TOKENS)]
        if constants.LLAMA_SERVER_KV_TYPE:
            args += ["-ctk", constants.LLAMA_SERVER_KV_TYPE, "-ctv", constants.LLAMA_SERVER_KV_TYPE]
        return args

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=1.0).status_code == 200
        except Exception:
            return False

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_running(self) -> float:
        """Start the server if it is not up; block until it answers /health.
        Returns the seconds spent starting (0 when it was already up)."""
        with self._start_lock:
            if self.is_running() and self.healthy():
                self.touch()
                return 0.0
            if self.proc is not None and self.proc.poll() is not None:
                logger.warning(f"llama-server exited with {self.proc.returncode}; restarting")
                self._close()
            t0 = time.monotonic()
            os.makedirs(os.path.dirname(constants.LLAMA_SERVER_LOG), exist_ok=True)
            self.log_file = open(constants.LLAMA_SERVER_LOG, "ab")
            env = {**os.environ, "LD_LIBRARY_PATH": constants.LLAMA_SERVER_LIB,
                   "GGML_BACKEND_PATH": constants.LLAMA_SERVER_BACKEND}
            self.proc = subprocess.Popen(self.command(), env=env, stdout=self.log_file, stderr=subprocess.STDOUT)
            self.started_at = time.time()
            while time.monotonic() - t0 < constants.LLAMA_SERVER_START_TIMEOUT_S:
                if self.healthy():
                    elapsed = time.monotonic() - t0
                    logger.info(f"llama-server up in {elapsed:.1f}s (pid {self.proc.pid}, port {self.port})")
                    self.touch()
                    self._ensure_watch()
                    return elapsed
                if self.proc.poll() is not None:
                    self._close()
                    raise RuntimeError(f"llama-server exited with {self.proc.returncode if self.proc else '?'} - see {constants.LLAMA_SERVER_LOG}")
                time.sleep(0.25)
            self.stop()
            raise RuntimeError(f"llama-server did not become healthy in {constants.LLAMA_SERVER_START_TIMEOUT_S}s")

    def touch(self) -> None:
        self.last_use = time.monotonic()

    def stop(self) -> None:
        with self._start_lock:
            self._close()

    def _close(self) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
            logger.info("llama-server stopped")
        self.proc = None
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

    def _ensure_watch(self) -> None:
        if self._watch is None or not self._watch.is_alive():
            self._watch = threading.Thread(target=self._idle_watch, name="llama-idle-watch", daemon=True)
            self._watch.start()

    def _idle_watch(self) -> None:
        """Stop the server LLAMA_SERVER_IDLE_STOP_S after the last request - the
        memory comes back for bhashini and the kiosk; the next Camera press starts it again."""
        while True:
            time.sleep(15)
            if not self.is_running():
                return
            idle = time.monotonic() - self.last_use
            if idle >= constants.LLAMA_SERVER_IDLE_STOP_S:
                logger.info(f"llama-server idle {idle:.0f}s - releasing its memory")
                self.stop()
                return

    # ── requests ─────────────────────────────────────────────────────────

    def chat(self, prompt: str, images_b64: Optional[List[str]], max_tokens: int, temperature: float,
             timeout: float) -> dict:
        """OpenAI-style chat completion on the running server. Returns the parsed
        response; raises on transport errors (callers map that to 'no answer')."""
        self.ensure_running()
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64 or []:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                   "temperature": temperature, "stream": False}
        try:
            resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        finally:
            self.touch()
