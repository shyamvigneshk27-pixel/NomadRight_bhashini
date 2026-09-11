#!/usr/bin/env python3
"""
Standalone kiosk launcher using pywebview's native GTK/WebKit2GTK backend,
instead of a full Chromium/Firefox process - see boards/hdmi.py's
_launch_kiosk(), which Popen's this as `python -m
pocketinfer.boards.webview_launcher <url>` exactly where it used to Popen
a browser. Runs in its own OS process (not a thread in the main
pocketinfer-service process) because webview.start() blocks on a GTK main
loop that must own the thread it runs on - the main process's own loop
(service.py's `while app.running`) can't share a thread with it.

Why this exists instead of the browser: Chromium/Firefox on this device
are only available as snap packages (confirmed - no native chromium/
chromium-browser/google-chrome, and firefox is the snap build), and snap's
own confinement/mount setup was already a documented, measured contributor
to slow kiosk startup (see hdmi.py's _launch_kiosk() docstring). WebKit2GTK
and its Python (PyGObject/`gi`) bindings are already installed natively on
this device (gir1.2-webkit2-4.0, libwebkit2gtk-4.0-37, python3-gi) - no new
system packages needed, just the pure-Python `pywebview` wheel in this
repo's own venv. A single embedded WebKit2 view is a substantially thinner
process than a full browser (no tabs/extensions/profile-manager layer),
so this is expected to cold-start noticeably faster.

boards/hdmi.py falls back to the previous Chromium/Firefox launch path if
`import webview` fails here (or this process fails to start at all) - this
file being missing, broken, or pywebview being uninstalled can never break
the existing working kiosk, only skip this faster path.
"""
import os
import signal
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: webview_launcher.py <url>", file=sys.stderr)
        return 1
    url = sys.argv[1]

    import webview

    # GTK's C main loop doesn't reliably hand control back to Python's
    # default signal handling - exit_app()/service.py's signal handler
    # both terminate() this process expecting it to actually die, matching
    # every other kiosk-exit path in this codebase's own os._exit()
    # philosophy (see hdmi.py's exit_app() docstring on why a hard exit is
    # what makes this reliable instead of occasionally hanging).
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    webview.create_window(
        "PocketInfer",
        url,
        fullscreen=True,
        frameless=True,
        confirm_close=False,
    )
    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
