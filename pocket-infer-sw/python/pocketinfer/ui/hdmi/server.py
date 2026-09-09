"""
Local-only (127.0.0.1) HTTP+WebSocket bridge server that replaces the
ILI9341 multiprocess UI (ui/handheld.py) as PocketInferHDMIBoard's display
layer (boards/hdmi.py).

Serves the built React UI (/home/stark-x/UI, `npm run build`, copied/
symlinked to webui_dist/ next to this repo - see webui_dist/README.md) as
static files, and keeps every connected browser tab's view of UIState
(state.py) in sync over /ws/ui. A second socket, /ws/terminal, exposes a
real PTY shell for the Settings > Admin tab, gated behind an admin PIN
check that happens entirely server-side (the old ILI9341 prototype's PIN
check was client-only, which was fine for a fake terminal but is not
acceptable now that this gates a real one).

Runs uvicorn on a background thread with its own asyncio event loop, so
PocketInferHDMIBoard's plain synchronous Board methods (called from GPIO
callback threads, the app's main loop thread, etc. - see boards/base.py)
can schedule broadcasts onto it with asyncio.run_coroutine_threadsafe
without themselves needing to become async.

Deliberately bound to 127.0.0.1 only, never 0.0.0.0: the kiosk browser and
this server run on the same device, and /ws/terminal is a real shell -
exposing it on the network would be a genuine security hole. See
boards/hdmi.py's DEFAULT_ADMIN_PIN warning.
"""

import asyncio
import fcntl
import json
import logging
import os
import pty
import signal
import struct
import termios
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pocketinfer.ui.hdmi import protocol
from pocketinfer.ui.hdmi.state import UIState

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765

_MISSING_UI_HTML = """<!doctype html><html><body style="background:#0f172a;
color:#e2e8f0;font-family:monospace;padding:2rem">
<h2>PocketInfer HDMI UI not built</h2>
<p>Build the UI (<code>cd /home/stark-x/UI &amp;&amp; npm run build</code>)
and copy/symlink its <code>dist/</code> directory to
<code>webui_dist/</code> next to pocket-infer-sw, or set the
<code>POCKETINFER_UI_DIST</code> environment variable to the build
output directory.</p>
<p>The backend API (WebSocket at <code>/ws/ui</code>) is running
normally.</p></body></html>"""


def _resolve_static_dir() -> Optional[Path]:
    # Require an actual built index.html, not just an existing directory -
    # webui_dist/ is checked into the repo (holding only a README, see
    # webui_dist/README.md) so it always exists even before the UI has
    # ever been built; treating a merely-existing-but-empty directory as
    # "found" would 404 on every request instead of showing the friendly
    # placeholder page below.
    env = os.environ.get("POCKETINFER_UI_DIST")
    if env:
        p = Path(env).expanduser()
        if (p / "index.html").is_file():
            return p
        logger.warning("POCKETINFER_UI_DIST=%s has no index.html - ignoring", env)
    # Conventional deployment location: build the UI repo, then copy/
    # symlink its dist/ here. pocket-infer-sw/python/pocketinfer/ui/hdmi/
    # -> pocket-infer-sw (4 parents up).
    repo_root = Path(__file__).resolve().parents[4]
    default = repo_root / "webui_dist"
    if (default / "index.html").is_file():
        return default
    return None


class HDMIBridgeServer:
    def __init__(self, state: UIState, port: int = DEFAULT_PORT,
                 admin_pin: str = "1234", terminal_cwd: Optional[str] = None):
        self.state = state
        self.port = port
        self.admin_pin = admin_pin
        self.terminal_cwd = terminal_cwd or os.getcwd()
        self.logger = logging.getLogger(__name__)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ui_clients: Set[WebSocket] = set()
        self._command_handlers: Dict[str, Callable[[dict], None]] = {}
        self._thread: Optional[threading.Thread] = None
        self.app = self._build_app()

    # ---- registration (called from PocketInferHDMIBoard) ------------------

    def on_command(self, msg_type: str, handler: Callable[[dict], None]) -> None:
        """Register a callback for a client->server /ws/ui command (see
        protocol.py's CMD_* constants). The handler runs on the server's
        asyncio thread and must not block - handlers that need to do real
        work (camera capture, GPIO, ...) should hand off to their own
        thread, as boards/hdmi.py's _dispatch_ui_cb does."""
        self._command_handlers[msg_type] = handler

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:
        ready = threading.Event()
        error_holder: list = []

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port,
                                     log_level="warning", loop="asyncio")
            server = uvicorn.Server(config)

            async def _serve() -> None:
                ready.set()
                await server.serve()

            try:
                loop.run_until_complete(_serve())
            except Exception as exc:  # pragma: no cover - startup failure path
                error_holder.append(exc)
                ready.set()

        self._thread = threading.Thread(target=_run, daemon=True, name="hdmi-bridge")
        self._thread.start()
        ready.wait(timeout=5.0)
        if error_holder:
            raise RuntimeError(f"HDMI bridge server failed to start: {error_holder[0]}")

    # ---- broadcasting (safe to call from any thread) -----------------

    def _broadcast(self, message: dict) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), self._loop)

    async def _broadcast_async(self, message: dict) -> None:
        text = json.dumps(message)
        dead = []
        for ws in list(self._ui_clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ui_clients.discard(ws)

    def broadcast_state_patch(self, patch: dict) -> None:
        if patch:
            self._broadcast({"type": protocol.MSG_STATE_PATCH, "patch": patch})

    def broadcast_log_line(self, text: str) -> None:
        self._broadcast({"type": protocol.MSG_LOG_LINE, "text": text})

    def broadcast_log_clear(self) -> None:
        self._broadcast({"type": protocol.MSG_LOG_CLEAR})

    def broadcast_radio_select(self, prefix: str, name: str) -> None:
        self._broadcast({"type": protocol.MSG_RADIO_SELECT, "prefix": prefix, "name": name})

    def broadcast_system_status(self, status: dict) -> None:
        self._broadcast({"type": protocol.MSG_SYSTEM_STATUS, "status": status})

    def broadcast_camera_frame(self, jpeg_b64: str) -> None:
        self._broadcast({"type": protocol.MSG_CAMERA_FRAME, "jpeg_b64": jpeg_b64})

    def broadcast_diagnostics_result(self, results: list) -> None:
        self._broadcast({"type": protocol.MSG_DIAGNOSTICS_RESULT, "results": results})

    def broadcast_error(self, message: str) -> None:
        self._broadcast({"type": protocol.MSG_ERROR, "message": message})

    # ---- FastAPI app ---------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.websocket("/ws/ui")
        async def ws_ui(websocket: WebSocket) -> None:
            await websocket.accept()
            self._ui_clients.add(websocket)
            self.logger.info("HDMI bridge: UI client connected (%d total)", len(self._ui_clients))
            try:
                # Full snapshot first - a reconnecting/late-joining tab
                # must never be stuck showing stale or blank state.
                await websocket.send_text(json.dumps({
                    "type": protocol.MSG_STATE_FULL,
                    "state": self.state.to_dict(),
                }))
                while True:
                    raw = await websocket.receive_text()
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(msg, dict):
                        continue
                    handler = self._command_handlers.get(msg.get("type"))
                    if handler is None:
                        continue
                    try:
                        handler(msg)
                    except Exception:
                        self.logger.exception("HDMI bridge: command handler failed for %r", msg.get("type"))
            except WebSocketDisconnect:
                pass
            except Exception:
                self.logger.exception("HDMI bridge: /ws/ui session error")
            finally:
                self._ui_clients.discard(websocket)
                self.logger.info("HDMI bridge: UI client disconnected (%d remaining)", len(self._ui_clients))

        @app.websocket("/ws/terminal")
        async def ws_terminal(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._run_terminal_session(websocket)

        static_dir = _resolve_static_dir()
        if static_dir is not None:
            app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
            self.logger.info("HDMI bridge: serving UI from %s", static_dir)
        else:
            @app.get("/")
            async def _missing_ui() -> HTMLResponse:
                return HTMLResponse(_MISSING_UI_HTML)
            self.logger.warning("HDMI bridge: no built UI found (set POCKETINFER_UI_DIST "
                                 "or populate webui_dist/) - serving a placeholder page")
        return app

    # ---- /ws/terminal: real PTY shell, PIN-gated -----------------------

    async def _run_terminal_session(self, websocket: WebSocket) -> None:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            msg = json.loads(raw)
        except Exception:
            await self._safe_close(websocket)
            return

        if not isinstance(msg, dict) or msg.get("type") != protocol.TERM_PIN or \
                str(msg.get("pin", "")) != self.admin_pin:
            await self._safe_send(websocket, {"type": protocol.TERM_AUTH_FAIL})
            await self._safe_close(websocket)
            return
        await self._safe_send(websocket, {"type": protocol.TERM_AUTH_OK})

        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        try:
            pid, fd = pty.fork()
        except OSError:
            self.logger.exception("HDMI bridge: pty.fork() failed")
            await self._safe_send(websocket, {"type": protocol.TERM_CLOSED})
            await self._safe_close(websocket)
            return

        if pid == 0:  # pragma: no cover - child process, never returns
            try:
                os.chdir(self.terminal_cwd)
            except OSError:
                pass
            os.execvp(shell, [shell])
            os._exit(1)

        loop = asyncio.get_event_loop()
        closed = asyncio.Event()

        def _on_readable() -> None:
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if not data:
                loop.call_soon_threadsafe(closed.set)
                return
            asyncio.run_coroutine_threadsafe(
                self._safe_send(websocket, {"type": protocol.TERM_DATA,
                                             "data": data.decode("utf-8", "replace")}),
                loop,
            )

        loop.add_reader(fd, _on_readable)
        try:
            while not closed.is_set():
                recv_task = asyncio.ensure_future(websocket.receive_text())
                closed_task = asyncio.ensure_future(closed.wait())
                done, pending = await asyncio.wait(
                    {recv_task, closed_task}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if closed_task in done:
                    break
                try:
                    raw = recv_task.result()
                except WebSocketDisconnect:
                    break
                except Exception:
                    continue
                try:
                    cmsg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(cmsg, dict):
                    continue
                if cmsg.get("type") == protocol.TERM_INPUT:
                    try:
                        os.write(fd, str(cmsg.get("data", "")).encode("utf-8"))
                    except OSError:
                        break
                elif cmsg.get("type") == protocol.TERM_RESIZE:
                    try:
                        cols = int(cmsg.get("cols", 80))
                        rows = int(cmsg.get("rows", 24))
                        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                    except (OSError, ValueError, TypeError):
                        pass
        finally:
            try:
                loop.remove_reader(fd)
            except Exception:
                pass
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            await self._safe_send(websocket, {"type": protocol.TERM_CLOSED})
            await self._safe_close(websocket)

    @staticmethod
    async def _safe_send(websocket: WebSocket, message: dict) -> None:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            pass

    @staticmethod
    async def _safe_close(websocket: WebSocket) -> None:
        try:
            await websocket.close()
        except Exception:
            pass
