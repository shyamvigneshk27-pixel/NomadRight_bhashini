"""
Thread-safe UI state shared between PocketInferHDMIBoard (boards/hdmi.py)
and HDMIBridgeServer (server.py).

This is the HDMI-era equivalent of what used to live only inside the
ILI9341 framebuffer (ui/handheld.py) - every Board method that used to
issue a draw call now mutates this object instead. Kept as plain
old-fashioned locked mutable state (not e.g. an immutable/event-sourced
model) to match the rest of this codebase's style (see boards/base.py,
audio.py) rather than introducing a new pattern for one module.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

# Matches ui/handheld.py's log page depth order of magnitude - plenty for
# a reconnecting browser tab to catch up on recent history without the
# message growing unbounded over a long-running session.
LOG_MAX_LINES = 300


@dataclass
class UIState:
    mode: str = "HOME"
    top: str = ""
    bottom: str = ""
    status: str = ""
    radios: Dict[str, str] = field(default_factory=dict)  # prefix -> selected name
    volume_pct: int = 100
    brightness_pct: int = 100
    button_led_on: bool = False
    system_status: Dict[str, str] = field(default_factory=dict)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=LOG_MAX_LINES))

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "top": self.top,
                "bottom": self.bottom,
                "status": self.status,
                "radios": dict(self.radios),
                "volume_pct": self.volume_pct,
                "brightness_pct": self.brightness_pct,
                "button_led_on": self.button_led_on,
                "system_status": dict(self.system_status),
                "log_lines": list(self.log_lines),
            }

    def update_screen(self, mode: Optional[str] = None, top: Optional[str] = None,
                       bottom: Optional[str] = None, status: Optional[str] = None) -> dict:
        """Mirrors Board.update_screen()'s "only touch fields that were
        passed" contract (boards/base.py) and returns just the patch that
        actually changed, ready to broadcast as-is."""
        with self._lock:
            patch: Dict[str, str] = {}
            if mode is not None:
                self.mode = mode
                patch["mode"] = mode
            if top is not None:
                self.top = top
                patch["top"] = top
            if bottom is not None:
                self.bottom = bottom
                patch["bottom"] = bottom
            if status is not None:
                self.status = status
                patch["status"] = status
            return patch

    def add_log_line(self, text: str) -> None:
        with self._lock:
            self.log_lines.append(text)

    def clear_log(self) -> None:
        with self._lock:
            self.log_lines.clear()

    def select_radio(self, prefix: str, name: str) -> None:
        with self._lock:
            self.radios[prefix] = name

    def set_button_led(self, on: bool) -> None:
        with self._lock:
            self.button_led_on = on

    def set_volume(self, pct: int) -> None:
        with self._lock:
            self.volume_pct = pct

    def set_brightness(self, pct: int) -> None:
        with self._lock:
            self.brightness_pct = pct

    def set_system_status(self, status: Dict[str, str]) -> None:
        with self._lock:
            self.system_status = dict(status)
