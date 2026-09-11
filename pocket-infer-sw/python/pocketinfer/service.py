#!/usr/bin/env python3
import os
import signal
import sys
import argparse
import logging
import time
import threading
import json
from pocketinfer.applications import *
from pocketinfer.applications.registry import ApplicationRegistry
from pocketinfer.boards.base import Board, DummyBoard
from psutil import virtual_memory


def _update_stats(board):
    while True:
        board.memory_text(f"{int(virtual_memory().percent)}%")
        time.sleep(2.0)


class _UILogHandler(logging.Handler):
    """
    Forwards every log record from every logger in this process to the
    on-screen pipeline log (board.log_line()) - the same channel/transport
    NomadRightApplication._log()'s curated stage lines already use, and
    already wired all the way to the frontend's Settings > Logs tab
    (state.log_lines, capped at 300 lines - see ui/hdmi/state.py).
    Without this, that panel only ever showed a handful of explicit
    _log() calls, not the full picture visible when running
    `pocketinfer-service` directly in a terminal (ASR/NMT/RAG/Qwen
    internals, warnings, tracebacks).

    Hardcoded to INFO regardless of --log-level: a DEBUG console session
    would otherwise push far more traffic than a 300-line kiosk log
    panel (or the websocket carrying it) is meant for.
    """

    def __init__(self, board):
        super().__init__(level=logging.INFO)
        self.board = board
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.board.log_line(self.format(record))
        except Exception:
            self.handleError(record)

def main():
    parser = argparse.ArgumentParser(description="PocketInfer Application Runner")
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--app', type=str, default="HearTheWorld", help='Name of the application to run')
    parser.add_argument('--list-apps', action='store_true', help='List available applications and exit')
    parser.add_argument('--update-app', action='store_true', default=False, help='Install dependencies for the specified application and exit')
    parser.add_argument('--dummy-board', action='store_true', default=False, help='Do not use hardware features - load audio and image from file')
    parser.add_argument('--headless', action='store_true', default=False, help='Use real hardware (trigger button, mic, speaker) but skip the display/UI layer entirely - status/text goes to the console log instead. Use this if no display is available.')
    parser.add_argument('--legacy-lcd', action='store_true', default=False, help='Use the LEGACY physical 2.4" ILI9341 touchscreen UI (ui/handheld.py) instead of the default HDMI browser UI (ui/hdmi/). Only needed if that display hardware is reattached.')
    parser.add_argument('--audio-file', type=str, help='Path to 16kHz 16-bit wav file to use with dummy board')
    parser.add_argument('--image-file', type=str, help='Path to image file to use with dummy board')
    parser.add_argument('--settings-file', default=None, type=str, help='Path to JSON file with application settings to override defaults')
    parser.add_argument('--setting', default=[], type=str, action='append', help='Override a specific application setting (can be used multiple times, e.g. --setting input_language=hi --setting output_language=en)')
    args = parser.parse_args()
    # Temporary code to test application startup
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    logging.debug(ApplicationRegistry._classes)
    if args.list_apps:
        print("Available applications:")
        for name in ApplicationRegistry._classes.keys():
            print(f"  {name}")
        sys.exit(0)
    app_cls = ApplicationRegistry.get_application(args.app)
    if app_cls is None: 
        logging.error("Application not found")
        sys.exit(1)

    if args.update_app:
        app_cls.update_dependencies()
        sys.exit(0)

    settings = {}
    if args.settings_file:
        with open(args.settings_file, 'r') as f:
            file_settings = json.load(f)
        if not isinstance(file_settings, dict):
            logging.error("Settings file must contain a JSON object (dictionary) at the top level")
            sys.exit(1)
        settings.update(file_settings)

    for setting_str in args.setting:
        if '=' not in setting_str:
            logging.error("Invalid setting format: %s. Must be key=value", setting_str)
            sys.exit(1)
        key, value = setting_str.split('=', 1)
        settings[key] = value

    if not args.dummy_board:
        board = Board.get_board(headless=args.headless, legacy_lcd=args.legacy_lcd)
    else:
        board = DummyBoard(vars(args))

    # Without this, Ctrl+C / a systemd stop only kills this Python process -
    # the HDMI board's kiosk browser subprocess (boards/hdmi.py's
    # _launch_kiosk(), a plain subprocess.Popen with no lifecycle tie to
    # this process) is orphaned and keeps running indefinitely, still
    # pointed at this now-dead process's UI server. HDMIBoard.exit_app()
    # already does exactly the right cleanup (kiosk terminate() + a bounded
    # hard os._exit()) for the one path that used to call it (the in-app
    # Admin "Exit Application" button) - this just wires the same path to
    # SIGINT/SIGTERM too. Boards without a kiosk browser (DummyBoard,
    # legacy LCD) simply don't define exit_app(), so this falls back to a
    # plain exit for them.
    def _handle_terminate_signal(signum, _frame):
        logging.info(f"Received signal {signum} - shutting down")
        exit_fn = getattr(board, "exit_app", None)
        if callable(exit_fn):
            exit_fn()
        else:
            os._exit(0)

    signal.signal(signal.SIGINT, _handle_terminate_signal)
    signal.signal(signal.SIGTERM, _handle_terminate_signal)

    logging.getLogger().addHandler(_UILogHandler(board))

    threading.Thread(target=_update_stats, args=(board,), daemon=True).start()
    board.statusbar("Starting: {}...".format(args.app))
    board.button_led(False)

    app_cls.verify_dependencies()
    logging.info(f"Starting application: {args.app}")
    board.mode_text(f"App {args.app}")
    app = app_cls(board, settings=settings)
    app.start()
    if args.dummy_board:
        # Only run application once
        app.running = False
    while app.running:
        time.sleep(1.0)
    app.stop()
    sys.exit(0)

if __name__ == "__main__":
    main()
