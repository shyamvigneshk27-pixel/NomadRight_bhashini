"""
HDMI board: the Jetson Orin Nano outputs over HDMI to a touch-capable
1024x600 display running the React UI (/home/stark-x/UI, built and served
by pocketinfer.ui.hdmi.server.HDMIBridgeServer) instead of the legacy
physical 2.4" ILI9341 touchscreen (see boards/jetson.py's
PocketInferDevboardUI - LEGACY, kept only as a hardware fallback if that
display is ever reattached).

Implements the exact same Board UI method surface
(statusbar/top_text/bottom_text/mode_text/memory_text/update_screen/
log_line/clear_log/select_radio/button_led) as the ILI9341 board, but
pushes JSON events over WebSocket to a browser tab instead of drawing to
an SPI framebuffer - see boards/base.py for the full contract every
UI-driving board must satisfy. app.py, workflow.py, etc. call these exact
same methods and neither know nor care which board is behind them.

Physical GPIO trigger button, camera, and audio are all inherited
unchanged from PocketInferDevboard (boards/jetson.py) - this class only
swaps the display/input layer, and *adds to* the input side (touch can
now drive the same trigger_button_down/_up Events the GPIO interrupt
does - see virtual_trigger_down()/_up()), it never removes the physical
button.
"""

import base64
import glob
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

from pocketinfer import audio
from pocketinfer.boards.jetson import PocketInferDevboard
from pocketinfer.ui.hdmi import protocol
from pocketinfer.ui.hdmi.server import HDMIBridgeServer
from pocketinfer.ui.hdmi.state import UIState

logger = logging.getLogger(__name__)

# CHANGE THIS before deploying anywhere the device might be network-
# reachable: this PIN now gates a REAL shell over /ws/terminal (see
# server.py), not the old touchscreen prototype's cosmetic client-side
# check. Override with the POCKETINFER_ADMIN_PIN environment variable.
DEFAULT_ADMIN_PIN = "1234"

# Camera preview stream rate for the Document Scanner screen - plenty for
# a "line the document up" live view without competing for camera/CPU
# bandwidth with an actual capture.
_PREVIEW_INTERVAL_S = 0.2  # ~5 fps

# Kiosk browser crash watchdog tuning - see _kiosk_watchdog_loop().
_WATCHDOG_POLL_INTERVAL_S = 3.0
_WATCHDOG_MAX_RESTARTS = 5
_WATCHDOG_RESTART_WINDOW_S = 300.0  # 5 minutes


class PocketInferHDMIBoard(PocketInferDevboard):
    def __init__(self, args, port: Optional[int] = None, kiosk: bool = True,
                 admin_pin: Optional[str] = None):
        super().__init__(args)

        self.state = UIState()
        self.state.set_volume(100)
        self.state.set_brightness(100)

        port = port or int(os.environ.get("POCKETINFER_UI_PORT", "8765"))
        pin = admin_pin or os.environ.get("POCKETINFER_ADMIN_PIN", DEFAULT_ADMIN_PIN)
        if pin == DEFAULT_ADMIN_PIN:
            self.logger.warning(
                "PocketInferHDMIBoard: admin terminal PIN is still the default "
                "'%s' - this now gates a REAL shell (see ui/hdmi/server.py). "
                "Set the POCKETINFER_ADMIN_PIN environment variable before "
                "deploying this device anywhere it could be network-reachable.",
                DEFAULT_ADMIN_PIN)

        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        self.bridge = HDMIBridgeServer(self.state, port=port, admin_pin=pin, terminal_cwd=repo_root)

        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop = threading.Event()

        self._wire_commands()
        self.bridge.start()

        self.url = f"http://127.0.0.1:{port}"
        self._kiosk_proc = None
        self._kiosk_intentional_exit = False
        self._kiosk_restart_times: list = []
        self._kiosk_watchdog_enabled = kiosk
        if kiosk:
            self._launch_kiosk()
            threading.Thread(target=self._kiosk_watchdog_loop, daemon=True).start()

    # ---- Board interface: board state -> browser ------------------------

    def clear_screen(self):
        self.state.clear_log()
        patch = self.state.update_screen(mode="HOME", top="", bottom="", status="")
        self.bridge.broadcast_state_patch(patch)
        self.bridge.broadcast_log_clear()

    def statusbar(self, text) -> bool:
        self.bridge.broadcast_state_patch(self.state.update_screen(status=text))
        return True

    def top_text(self, text) -> bool:
        self.bridge.broadcast_state_patch(self.state.update_screen(top=text))
        return True

    def bottom_text(self, text) -> bool:
        self.bridge.broadcast_state_patch(self.state.update_screen(bottom=text))
        return True

    def mode_text(self, text) -> bool:
        self.bridge.broadcast_state_patch(self.state.update_screen(mode=text))
        return True

    def memory_text(self, text) -> bool:
        # No dedicated memory-% readout in the HDMI UI (the old ILI9341
        # topbar had one) - real system health is surfaced instead via
        # Settings > Diagnostics (run_diagnostics()) and Device Status.
        # Swallow rather than raise so service.py's _update_stats() thread
        # (which calls this every 2s regardless of board type) needs no
        # special case for this board.
        return True

    def update_screen(self, mode=None, top=None, bottom=None, status=None) -> bool:
        self.bridge.broadcast_state_patch(
            self.state.update_screen(mode=mode, top=top, bottom=bottom, status=status))
        return True

    def log_line(self, text) -> bool:
        self.state.add_log_line(text)
        self.bridge.broadcast_log_line(text)
        return True

    def clear_log(self) -> bool:
        self.state.clear_log()
        self.bridge.broadcast_log_clear()
        return True

    def select_radio(self, prefix, name) -> bool:
        self.state.select_radio(prefix, name)
        self.bridge.broadcast_radio_select(prefix, name)
        return True

    def button_led(self, value) -> bool:
        on = bool(value)
        self.state.set_button_led(on)
        self.bridge.broadcast_state_patch({"button_led_on": on})
        return True

    # ---- Touch/virtual trigger button (browser -> board) -----------------

    def virtual_trigger_down(self) -> None:
        """
        Sets exactly the same trigger_button flag/Event the real GPIO
        interrupt (PocketInferDevboard.trig_cb, unchanged) sets, so
        app.py's wait_for_trigger_button_down()/_up() (boards/base.py)
        genuinely cannot tell a touchscreen "tap to speak" from a
        physical button press. This is how touch and the physical trigger
        button both drive the exact same pipeline with zero changes to
        app.py's loop.
        """
        self.trigger_button = True
        self.trigger_button_down.set()

    def virtual_trigger_up(self) -> None:
        self.trigger_button = False
        self.trigger_button_up.set()

    # ---- Live volume / brightness (neither had a live control before) ----

    def set_volume_live(self, pct) -> bool:
        pct = max(0, min(100, int(pct)))
        ok = bool(self.alsa_playback_card is not None and audio.set_volume(self.alsa_playback_card, pct))
        self.state.set_volume(pct)
        self.bridge.broadcast_state_patch({"volume_pct": pct})
        return ok

    def set_brightness(self, pct) -> bool:
        pct = max(15, min(100, int(pct)))
        hw_ok = self._try_hw_brightness(pct)
        self.state.set_brightness(pct)
        # Always broadcast regardless of hw_ok - the UI's own
        # BrightnessOverlay (client-side CSS dim, App.tsx) is the
        # guaranteed fallback so brightness visibly does something even
        # with zero controllable backlight hardware.
        self.bridge.broadcast_state_patch({"brightness_pct": pct})
        return hw_ok

    def _try_hw_brightness(self, pct: int) -> bool:
        """Best-effort real hardware brightness control: try a sysfs
        backlight device first (common on panels with a real eDP/DDC
        backlight), then xrandr software gamma as a fallback for a
        generic HDMI monitor. No physical PWM backlight control exists on
        this device today (the old ILI9341's pwm_pin was only ever driven
        fully on/off - see boards/jetson.py) so both are best-effort and
        failures here are expected/harmless."""
        try:
            max_paths = glob.glob("/sys/class/backlight/*/max_brightness")
            if max_paths:
                max_path = max_paths[0]
                set_path = max_path.replace("max_brightness", "brightness")
                with open(max_path) as f:
                    max_val = int(f.read().strip())
                with open(set_path, "w") as f:
                    f.write(str(int(max_val * pct / 100)))
                return True
        except (OSError, ValueError):
            self.logger.debug("HDMI board: sysfs backlight control unavailable", exc_info=True)

        try:
            display_env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}
            probe = subprocess.run(["xrandr", "--query"], env=display_env,
                                    capture_output=True, text=True, timeout=2.0)
            output_name = next(
                (line.split()[0] for line in probe.stdout.splitlines() if " connected" in line),
                None,
            )
            if output_name:
                level = max(0.15, min(1.0, pct / 100.0))
                subprocess.run(["xrandr", "--output", output_name, "--brightness", f"{level:.2f}"],
                                env=display_env, timeout=2.0)
                return True
        except (OSError, subprocess.SubprocessError):
            self.logger.debug("HDMI board: xrandr brightness control unavailable", exc_info=True)
        return False

    # ---- Diagnostics (real hardware/service self-test, on demand) --------

    def run_diagnostics(self) -> list:
        """The same checks NomadRightApplication._startup_selfcheck()
        already performs at launch, exposed on demand for
        Settings > Diagnostics instead of mockDiagnosticsService's canned
        pass/fail results."""
        import requests

        results = []

        def _check(id_, name, category, fn):
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, str(exc)
            results.append({"id": id_, "name": name, "category": category,
                             "status": "PASSED" if ok else "FAILED", "details": detail})

        _check("bhashini", "BHASHINI Models", "AI Engine", lambda: (
            requests.get("http://localhost:11400/health", timeout=1.5).status_code == 200,
            "localhost:11400/health"))
        _check("ollama", "Ollama / Qwen-VL", "AI Engine", lambda: (
            requests.get("http://localhost:11434/api/tags", timeout=1.5).status_code == 200,
            "localhost:11434"))
        _check("mic", "Microphone", "Audio", lambda: (
            self.alsa_capture_card is not None,
            f"card {self.alsa_capture_card}" if self.alsa_capture_card is not None else "not found"))
        _check("speaker", "Speaker", "Audio", lambda: (
            self.alsa_playback_card is not None,
            f"card {self.alsa_playback_card}" if self.alsa_playback_card is not None else "not found"))
        _check("camera", "Camera", "Camera", lambda: (
            self.camera_frame_jpg() is not None, "USB webcam probe"))

        self.state.set_system_status({r["id"]: r["status"] for r in results})
        self.bridge.broadcast_diagnostics_result(results)
        self.bridge.broadcast_system_status(self.state.system_status)
        return results

    # ---- Document Scanner live camera preview -----------------------------

    def _set_camera_preview(self, enabled: bool) -> None:
        if enabled:
            if self._preview_thread is not None and self._preview_thread.is_alive():
                return
            self._preview_stop.clear()
            self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self._preview_thread.start()
        else:
            self._preview_stop.set()

    def _preview_loop(self) -> None:
        while not self._preview_stop.is_set():
            try:
                jpg = self.camera_frame_jpg()
                if jpg:
                    self.bridge.broadcast_camera_frame(base64.b64encode(bytes(jpg)).decode("ascii"))
            except Exception:
                self.logger.debug("HDMI board: camera preview frame failed", exc_info=True)
            self._preview_stop.wait(timeout=_PREVIEW_INTERVAL_S)

    # ---- Command wiring (browser -> board) --------------------------------

    def _dispatch_ui_cb(self, msg: str) -> None:
        """Fan `msg` out to every board.ui_cbs subscriber (app.py's
        ui_cb(), subscribed via board.subscribe_to_ui() - boards/base.py),
        off the server's asyncio thread since a subscriber may do real
        work (camera capture, network calls)."""
        def _run() -> None:
            for cb in list(self.ui_cbs):
                try:
                    cb(msg)
                except Exception:
                    self.logger.exception("HDMI board: UI callback failed for %r", msg)
        threading.Thread(target=_run, daemon=True).start()

    def _wire_commands(self) -> None:
        b = self.bridge
        b.on_command(protocol.CMD_TRIGGER_DOWN, lambda msg: self.virtual_trigger_down())
        b.on_command(protocol.CMD_TRIGGER_UP, lambda msg: self.virtual_trigger_up())
        b.on_command(protocol.CMD_CAMERA_PRESS, lambda msg: self._dispatch_ui_cb("Camera"))
        b.on_command(protocol.CMD_HOME_PRESS, lambda msg: self._dispatch_ui_cb("Home"))
        b.on_command(protocol.CMD_SELECT_ASR_LANG,
                     lambda msg: self._dispatch_ui_cb(f"ASR {msg.get('code', '')}"))
        b.on_command(protocol.CMD_SELECT_BRIDGE_LANG,
                     lambda msg: self._dispatch_ui_cb(f"Bridge {msg.get('code', '')}"))
        b.on_command(protocol.CMD_SET_VOLUME, lambda msg: self.set_volume_live(msg.get("pct", 100)))
        b.on_command(protocol.CMD_SET_BRIGHTNESS, lambda msg: self.set_brightness(msg.get("pct", 100)))
        b.on_command(protocol.CMD_RUN_DIAGNOSTICS,
                     lambda msg: threading.Thread(target=self.run_diagnostics, daemon=True).start())
        b.on_command(protocol.CMD_CAMERA_PREVIEW, lambda msg: self._set_camera_preview(bool(msg.get("enabled"))))
        b.on_command(protocol.CMD_ASK_DOCUMENT_TEXT, lambda msg: self._dispatch_ui_cb(f"DocText {msg.get('text', '')}"))
        b.on_command(protocol.CMD_EXIT_APP, lambda msg: self.exit_app())

    # ---- Exit portal (Settings > Admin > Exit Application) ----------------

    def exit_app(self) -> None:
        """
        The only way out of a fullscreen kiosk browser with no window
        chrome and (typically) no keyboard attached - PIN-gated in the UI
        the same way the Admin terminal is (see server.py's /ws/terminal
        PIN check; this command travels over the already-PIN-protected
        Admin tab). Closes the kiosk browser, then hard-exits the whole
        process shortly after.

        Uses os._exit() rather than a graceful shutdown: app.py's
        NomadRightApplication.stop() joins its run() thread, which can be
        blocked on a live network call (ASR/NMT/TTS/Qwen) with no
        cooperative cancellation point - see app.py's own module
        docstring on why a live Qwen/RAG call can't be safely aborted
        mid-request. A hard exit is what makes "Exit Application" reliably
        instant instead of occasionally hanging until an in-flight request
        times out. To run again: `pocketinfer-service` from a terminal.
        """
        self.logger.info("HDMI board: Exit Application requested from Admin panel")
        # Set before terminate() so the watchdog (which polls
        # self._kiosk_proc, running concurrently on its own thread) sees
        # this was deliberate and does not relaunch the browser in the
        # ~3s window before the whole process itself exits below.
        self._kiosk_intentional_exit = True
        if self._kiosk_proc is not None:
            try:
                self._kiosk_proc.terminate()
            except Exception:
                self.logger.debug("HDMI board: kiosk browser terminate failed", exc_info=True)

        def _hard_exit() -> None:
            time.sleep(0.8)  # let the terminate() above land, and the
            # TERM_CLOSED/response reach the browser before the process
            # (and its WebSocket) disappears out from under it.
            os._exit(0)

        threading.Thread(target=_hard_exit, daemon=True).start()

    # ---- Kiosk browser launch ---------------------------------------------

    def _launch_kiosk(self) -> None:
        """Makes `pocketinfer-service` alone put the UI on the attached
        HDMI screen with no separate manual step. Needs an active X/
        Wayland session to draw into - uses $DISPLAY if already set
        (normal when launched from a desktop session), else
        $POCKETINFER_DISPLAY, else falls back to ":0"; set
        POCKETINFER_DISPLAY explicitly (e.g. ":1") when running as a
        systemd service outside the interactive desktop session - see
        python/pocketinfer.service's [Service] block. Tries Chromium/
        Chrome first, then Firefox's `-kiosk` mode as a fallback for
        devices that only have Firefox installed. Any failure here (no
        supported browser, no display session yet) is logged and
        swallowed: the backend/API still work headless even if the kiosk
        window itself can't come up.

        Always launches into a fresh, throwaway profile directory (a new
        one every process start, under ~/.cache/pocketinfer/kiosk-
        profiles/) rather than reusing a persistent default profile.
        Three real bugs were traced back to profile handling, on-device:

          1. A hard-killed browser (the exit portal's os._exit(), or any
             external `kill -9`) leaves a stale profile lock, which makes
             the *next* launch against that same profile refuse to open
             at all - silently exits near-instantly, PID goes defunct,
             HDMI shows nothing but the bare desktop.
          2. An unclean shutdown flags a profile as crashed, and the next
             launch against it can show a blank/interstitial "restore
             session" page in kiosk mode instead of the app.
          3. The throwaway profile fixing (1) and (2) was originally
             created via tempfile.mkdtemp() with no `dir=`, which
             defaults to /tmp - invisible to this Firefox build's snap
             confinement (its `home` interface grants $HOME, not /tmp),
             so the browser exited immediately with zero output, same
             defunct-PID symptom as (1). Confirmed on-device with
             `xprop -root _NET_CLIENT_LIST`: no window at all with the
             profile under /tmp, a real one appears within seconds with
             the identical command once the profile moves under $HOME.

        A $HOME-anchored profile is what avoids all three permanently.
        It's a SINGLE reused profile now, not a brand-new one every
        launch: a fresh profile paid Firefox's full first-run cost (cert
        DB generation, extension state, JS bytecode cache, snap's own
        first-touch confinement setup) on every single restart - measured
        on-device as a real, disproportionate chunk of the ~30s it took
        the kiosk window to first appear, worse under memory pressure.
        Reusing one profile keeps that cost one-time instead of paying it
        on every restart. To still avoid the two failure modes above
        without a disposable profile's inherent immunity to them, the
        exact artifacts that cause them - the lock and the crash-recovery
        session files - are removed before every launch instead of
        wiping the whole profile: same reliability, without discarding
        the warm cache that made repeat launches slow.
        """
        display = os.environ.get("DISPLAY") or os.environ.get("POCKETINFER_DISPLAY", ":0")
        env = {**os.environ, "DISPLAY": display}

        # Preferred path: a single embedded WebKit2GTK view via pywebview
        # (boards/webview_launcher.py, run as its own process - see that
        # file's docstring for why it can't just be a thread here).
        # Measured on this device: ~0.6s to a rendered window, vs. ~2.4s
        # for Firefox even with a fresh throwaway profile (real-world
        # Firefox launches are slower still - see the profile-cleanup
        # block below's own comments on snap confinement/cold-cache cost).
        # `import webview` failing (pywebview not installed) is the only
        # normal way this branch is skipped - falls straight through to
        # the exact same Chromium/Firefox launch this always used, so a
        # missing/broken pywebview install can never regress below the
        # previously-working kiosk, only skip the faster path.
        try:
            import webview  # noqa: F401
        except ImportError:
            webview = None

        if webview is not None:
            cmd = [sys.executable, "-m", "pocketinfer.boards.webview_launcher", self.url]
            try:
                self._kiosk_proc = subprocess.Popen(cmd, env=env)
                return
            except OSError:
                self.logger.exception(
                    "HDMI board: failed to launch pywebview kiosk - falling back to browser")

        profiles_root = os.path.expanduser("~/.cache/pocketinfer/kiosk-profiles")
        profile_dir = os.path.join(profiles_root, "kiosk-profile")
        os.makedirs(profile_dir, exist_ok=True)
        # Firefox: lock (symlink) + .parentlock (a hard-killed browser -
        # os._exit() in exit_app(), or an external kill -9 - leaves these
        # behind and the next launch against this profile refuses to open
        # at all otherwise) and the crash-recovery session files (an
        # unclean shutdown flags these, and the next launch can show a
        # blank/interstitial "restore session" page instead of the app).
        for stale in ("lock", ".parentlock"):
            try:
                os.remove(os.path.join(profile_dir, stale))
            except OSError:
                pass
        recovery_dir = os.path.join(profile_dir, "sessionstore-backups")
        for stale in ("recovery.jsonlz4", "recovery.baklz4", "previous.jsonlz4"):
            try:
                os.remove(os.path.join(recovery_dir, stale))
            except OSError:
                pass
        # Chromium/Chrome: the analogous singleton lock + crash-recovery
        # session state, in case this device ever gets a non-snap
        # Chromium installed (preferred below when present - no snap
        # confinement/cold-start overhead at all).
        for stale in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            try:
                os.remove(os.path.join(profile_dir, stale))
            except OSError:
                pass
        for stale in ("Last Session", "Last Tabs"):
            try:
                os.remove(os.path.join(profile_dir, "Default", stale))
            except OSError:
                pass

        def _on_path(name: str) -> bool:
            return subprocess.run(["which", name], capture_output=True).returncode == 0

        chromium = next((c for c in ("chromium-browser", "chromium", "google-chrome") if _on_path(c)), None)
        if chromium is not None:
            cmd = [chromium, "--kiosk", "--noerrdialogs", "--disable-infobars",
                   "--disable-session-crashed-bubble", f"--user-data-dir={profile_dir}",
                   f"--app={self.url}"]
        elif _on_path("firefox"):
            # -no-remote: don't try to hand the URL off to another
            # already-running Firefox instance (which is also what makes
            # a leftover profile lock cause a silent no-op launch instead
            # of a real error) - this process always owns its own
            # brand-new profile, so there's nothing to hand off to.
            cmd = ["firefox", "-kiosk", "-no-remote", "-profile", profile_dir, self.url]
        else:
            self.logger.warning("HDMI board: no supported browser (chromium/chrome/firefox) found "
                                 "on PATH - kiosk window not launched; the UI is still served at %s",
                                 self.url)
            return
        try:
            self._kiosk_proc = subprocess.Popen(cmd, env=env)
        except OSError:
            self.logger.exception("HDMI board: failed to launch kiosk browser (DISPLAY=%s)", display)

    def _kiosk_watchdog_loop(self) -> None:
        """
        Runs for the lifetime of the process on its own daemon thread,
        polling whether the kiosk browser is still alive. If it exits on
        its own (renderer crash, OOM kill, anything other than
        exit_app()'s deliberate terminate()) it's relaunched automatically
        - the backend, loaded models, and WS bridge are never touched, so
        a browser crash costs a few seconds instead of the full RAG/model
        cold-load a full service restart would re-pay (measured 58-140s
        on this device - see workflow.py).

        Rate-limited (_WATCHDOG_MAX_RESTARTS within
        _WATCHDOG_RESTART_WINDOW_S): a browser that keeps crashing
        immediately after each relaunch (corrupt build, out of memory,
        anything genuinely broken) should fail loudly and stay down
        rather than spin forever burning CPU - the backend/API keep
        working headless either way, exactly as if kiosk=False.
        """
        while self._kiosk_watchdog_enabled:
            time.sleep(_WATCHDOG_POLL_INTERVAL_S)
            proc = self._kiosk_proc
            if proc is None or proc.poll() is None:
                continue  # still starting up, or still running
            if self._kiosk_intentional_exit:
                self.logger.info("HDMI board: kiosk browser closed intentionally - watchdog standing down")
                return

            now = time.monotonic()
            self._kiosk_restart_times = [t for t in self._kiosk_restart_times
                                          if now - t < _WATCHDOG_RESTART_WINDOW_S]
            if len(self._kiosk_restart_times) >= _WATCHDOG_MAX_RESTARTS:
                self.logger.error(
                    "HDMI board: kiosk browser crashed %d times in %.0fs - giving up on "
                    "auto-restart (backend/API still running headless). Investigate and "
                    "restart pocketinfer-service manually.",
                    len(self._kiosk_restart_times), _WATCHDOG_RESTART_WINDOW_S)
                return

            self._kiosk_restart_times.append(now)
            self.logger.warning(
                "HDMI board: kiosk browser exited unexpectedly (code=%s) - relaunching (%d/%d "
                "restarts used in the last %.0fs)",
                proc.returncode, len(self._kiosk_restart_times), _WATCHDOG_MAX_RESTARTS,
                _WATCHDOG_RESTART_WINDOW_S)
            self._launch_kiosk()
