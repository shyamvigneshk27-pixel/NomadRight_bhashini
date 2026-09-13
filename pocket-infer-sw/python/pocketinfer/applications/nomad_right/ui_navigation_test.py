#!/usr/bin/env python3
"""
NomadRight UI Navigation / Mode-Switching Tests
===================================================
Regression tests for the Home / Voice Translation / Document Scanner mode
switching added in app.py (_on_camera_pressed, _on_home_pressed,
_stop_audio, _play). Does NOT touch or re-test the underlying voice/vision
pipeline (workflow.py, response.py, qwen_client.py) - see pipeline_test.py
and audit_fix_regression_test.py for that; those are explicitly unchanged
by this feature.

Uses a lightweight mock board (no real touchscreen/camera/audio hardware
required) and mocks AudioPlayer so playback tests are fast/deterministic
and never actually spawn ffplay or make sound.

Usage (run from repo root):
    python python/pocketinfer/applications/nomad_right/ui_navigation_test.py
"""

import sys
import os
import time
import threading
import unittest
import unittest.mock

# ── Path setup ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
os.chdir(_REPO_ROOT)

import logging
logging.basicConfig(level=logging.WARNING)

from pocketinfer.applications.nomad_right.app import NomadRightApplication


class MockBoard:
    """Records every LCD call so tests can assert on the exact screen state,
    without needing the real touchscreen UI subprocess or camera/audio
    hardware. camera_frame_jpg()/alsa_playback_device are configurable per
    test to simulate success/failure."""

    def __init__(self):
        self.calls = []
        self.screen_updates = []
        self.top = ""
        self.bottom = ""
        self.status = ""
        self.mode = ""
        self.alsa_playback_device = "hw:0,0"
        self._camera_frame = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
        self._camera_raises = None

    def top_text(self, t):
        self.top = t
        self.calls.append(("top", t))

    def select_radio(self, prefix, name):
        self.calls.append(("select_radio", prefix, name))
        return True

    def bottom_text(self, t):
        self.bottom = t
        self.calls.append(("bottom", t))

    def statusbar(self, t):
        self.status = t
        self.calls.append(("status", t))

    def mode_text(self, t):
        self.mode = t
        self.calls.append(("mode", t))

    def update_screen(self, mode=None, top=None, bottom=None, status=None):
        # Mirrors boards/jetson.py's PocketInferDevboardUI.update_screen():
        # the real board sends this as a single RPC round trip instead of
        # separate top_text()/bottom_text()/mode_text()/statusbar() calls,
        # which is what makes it atomic (see HandheldUI.update_screen()'s
        # docstring). Records into self.calls exactly like the individual
        # setters used to (field, text) - so existing per-field assertions
        # keep working unchanged - AND appends the whole group to
        # self.screen_updates in one shot, so TestScreenAtomicity can assert
        # on grouping directly instead of inferring it from gaps between
        # separate calls.
        if mode is not None:
            self.mode = mode
            self.calls.append(("mode", mode))
        if top is not None:
            self.top = top
            self.calls.append(("top", top))
        if bottom is not None:
            self.bottom = bottom
            self.calls.append(("bottom", bottom))
        if status is not None:
            self.status = status
            self.calls.append(("status", status))
        self.screen_updates.append({"mode": mode, "top": top, "bottom": bottom, "status": status})

    def clear_screen(self):
        pass

    def button_led(self, v):
        pass

    def camera_frame_jpg(self):
        if self._camera_raises:
            raise self._camera_raises
        return self._camera_frame


def _make_app():
    """Builds a NomadRightApplication with a MockBoard, skipping start()
    (no BhashiniBridge/WorkflowController construction - these tests only
    exercise UI/navigation methods that don't touch either)."""
    app = NomadRightApplication.__new__(NomadRightApplication)
    app.logger = logging.getLogger("test")
    app.board = MockBoard()
    app.settings = {}
    app.last_answer_en = ""
    app.last_scheme_code = None
    app.pending_form_image = None
    app.pending_form_image_ts = 0.0
    app._image_lock = threading.Lock()
    app._camera_busy = threading.Event()
    app._mode = "HOME"
    app._current_player = None
    app._player_lock = threading.Lock()
    app._audio_stop_requested = threading.Event()
    app._home_requested = threading.Event()
    return app


class TestCameraButton(unittest.TestCase):
    """Document Scanner entry point (Camera icon -> _on_camera_pressed)."""

    def test_successful_capture_sets_pending_image_and_scanner_mode(self):
        app = _make_app()
        app._on_camera_pressed()
        self.assertIsNotNone(app.pending_form_image)
        self.assertEqual(app._mode, "DOCUMENT SCANNER")
        self.assertIn("CAPTURED", app.board.top)
        self.assertFalse(app._camera_busy.is_set(), "must be cleared after capture")

    def test_camera_exception_falls_back_to_home_with_error_screen(self):
        app = _make_app()
        app.board._camera_raises = RuntimeError("camera disconnected")
        app._on_camera_pressed()
        self.assertIsNone(app.pending_form_image)
        self.assertEqual(app._mode, "HOME")
        self.assertIn("ERROR", app.board.top)
        self.assertFalse(app._camera_busy.is_set())

    def test_empty_frame_falls_back_to_home_with_error_screen(self):
        app = _make_app()
        app.board._camera_frame = None
        app._on_camera_pressed()
        self.assertIsNone(app.pending_form_image)
        self.assertEqual(app._mode, "HOME")
        self.assertIn("ERROR", app.board.top)

    def test_pressing_camera_again_replaces_pending_photo(self):
        """Re-scanning (Camera pressed again while a photo is already
        pending) must replace it, not stack/leak the old one."""
        app = _make_app()
        app._on_camera_pressed()
        first_photo = app.pending_form_image
        app.board._camera_frame = b"\xff\xd8\xff\xe0second-fake-jpeg"
        app._on_camera_pressed()
        self.assertNotEqual(app.pending_form_image, first_photo)
        self.assertEqual(app._mode, "DOCUMENT SCANNER")

    def test_camera_press_stops_any_playing_audio_first(self):
        """A fresh scan shouldn't have the previous turn's answer still
        talking over it (see _on_camera_pressed's _stop_audio() call)."""
        app = _make_app()
        fake_player = unittest.mock.MagicMock()
        app._current_player = fake_player
        app._on_camera_pressed()
        fake_player.stop.assert_called_once()


class TestHomeButton(unittest.TestCase):
    """The single always-available cancel/back/stop control."""

    def test_home_with_nothing_active_is_a_safe_noop(self):
        app = _make_app()
        app._on_home_pressed()
        self.assertEqual(app._mode, "HOME")
        self.assertIsNone(app.pending_form_image)

    def test_home_cancels_pending_scanner_photo(self):
        app = _make_app()
        app._on_camera_pressed()
        self.assertIsNotNone(app.pending_form_image)
        app._on_home_pressed()
        self.assertIsNone(app.pending_form_image)
        self.assertEqual(app._mode, "HOME")
        self.assertIn("CANCELLED", app.board.top)

    def test_home_stops_playing_audio_and_shows_stopped_screen(self):
        app = _make_app()
        fake_player = unittest.mock.MagicMock()
        app._current_player = fake_player
        app._on_home_pressed()
        fake_player.stop.assert_called_once()
        self.assertTrue(app._audio_stop_requested.is_set())
        self.assertIn("AUDIO STOPPED", app.board.top)

    def test_home_sets_cooperative_cancel_flag_for_run_loop(self):
        app = _make_app()
        app._on_home_pressed()
        self.assertTrue(app._home_requested.is_set())


class TestAsrLanguageGating(unittest.TestCase):
    """Settings-page ASR language buttons: SOURCE_LANGUAGES (constants.py)
    lists 7 codes the touchscreen renders buttons for, but BHASHINI only
    ships ASR checkpoints for 2 of them (hi, ta - see infer.py). Selecting
    one of the other 5 used to be accepted silently, leaving every
    subsequent turn's /asr call 500 forever with the screen still
    confidently highlighting the dead button (observed live: worker
    language set to 'hne', every turn after logged 'ASR empty' with no
    visible explanation). ui_cb() now gates on ASR_SUPPORTED_LANGUAGES."""

    def test_supported_language_selection_is_applied(self):
        app = _make_app()
        app.ui_cb("ASR Hi")
        self.assertEqual(app.settings["input_language"], "hi")
        app.ui_cb("ASR Ta")
        self.assertEqual(app.settings["input_language"], "ta")

    def test_unsupported_language_selection_is_refused(self):
        """The exact failure observed live: BHASHINI has no ASR model for
        Chhattisgarhi, so selecting it must not take."""
        app = _make_app()
        app.ui_cb("ASR Hi")
        app.ui_cb("ASR Hne")
        self.assertEqual(
            app.settings["input_language"], "hi",
            "an unsupported ASR language must not overwrite a working one",
        )

    def test_unsupported_selection_snaps_the_highlight_back(self):
        app = _make_app()
        app.ui_cb("ASR Hi")
        app.ui_cb("ASR Bho")
        self.assertIn(("select_radio", "ASR ", "ASR Hi"), app.board.calls)

    def test_unsupported_selection_survives_a_broken_select_radio(self):
        """select_radio() goes over the UI subprocess RPC (ui/handheld.py) -
        if that call itself fails, the refusal must still complete instead
        of taking down the UI callback thread."""
        app = _make_app()
        app.board.select_radio = unittest.mock.MagicMock(side_effect=RuntimeError("rpc down"))
        app.ui_cb("ASR Sat")  # must not raise
        self.assertNotEqual(app.settings.get("input_language"), "sat")

    def test_every_unsupported_language_is_refused(self):
        app = _make_app()
        app.ui_cb("ASR Hi")
        for bad in ["ASR Or", "ASR Bho", "ASR Mai", "ASR Sat", "ASR Hne"]:
            app.ui_cb(bad)
            self.assertEqual(app.settings["input_language"], "hi", f"{bad!r} was wrongly accepted")

    def test_asr_supported_languages_matches_real_bhashini_checkpoints(self):
        """Locks the gating set to what infer.py actually loads, so this
        test breaks (loudly, in CI) instead of the LCD (silently, live) the
        next time someone edits one without the other."""
        import json, os
        from pocketinfer.applications.nomad_right import constants
        # the speech service's own mapping (~/bhashini_models/asr/asr_config.json) is the truth:
        # every language it serves (Conformer, Whisper or the lazily loaded English engine)
        cfg = os.path.expanduser("~/bhashini_models/asr/asr_config.json")
        if os.path.exists(cfg):
            served = {lang for lang, backend in json.load(open(cfg)).get("backends", {}).items() if backend}
            self.assertEqual(constants.ASR_SUPPORTED_LANGUAGES, served)
        else:
            self.assertEqual(constants.ASR_SUPPORTED_LANGUAGES, {"hi", "ta", "en"})
        self.assertTrue(constants.ASR_SUPPORTED_LANGUAGES.issubset(constants.SOURCE_LANGUAGES.keys()))


class TestStopAudio(unittest.TestCase):
    def test_returns_false_when_nothing_playing(self):
        app = _make_app()
        self.assertFalse(app._stop_audio())

    def test_kills_current_player_and_returns_true(self):
        app = _make_app()
        fake_player = unittest.mock.MagicMock()
        app._current_player = fake_player
        result = app._stop_audio()
        self.assertTrue(result)
        fake_player.stop.assert_called_once()

    def test_thread_safe_from_a_different_thread(self):
        """_stop_audio() is designed to be called from the UI callback
        thread while _play() runs on the main thread - verify no
        exception/deadlock calling it from a real second thread."""
        app = _make_app()
        fake_player = unittest.mock.MagicMock()
        app._current_player = fake_player
        results = []
        t = threading.Thread(target=lambda: results.append(app._stop_audio()))
        t.start()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(results, [True])


class TestPlayCancellation(unittest.TestCase):
    """_play() with AudioPlayer mocked out - no real ffplay/audio hardware,
    no actual sound, deterministic and fast."""

    def _mock_audio_player(self, on_play=None):
        """Returns a MagicMock standing in for AudioPlayer's `with
        AudioPlayer(...) as player:` context-manager usage."""
        instance = unittest.mock.MagicMock()
        instance.__enter__ = unittest.mock.MagicMock(return_value=instance)
        instance.__exit__ = unittest.mock.MagicMock(return_value=False)
        if on_play:
            instance.play.side_effect = on_play
        return instance

    def test_empty_bytes_returns_true_immediately(self):
        app = _make_app()
        self.assertTrue(app._play(b""))

    def test_normal_playback_returns_true(self):
        app = _make_app()
        mock_player = self._mock_audio_player()
        with unittest.mock.patch(
            "pocketinfer.applications.nomad_right.app.AudioPlayer",
            return_value=mock_player,
        ), unittest.mock.patch("wave.open") as mock_wave:
            mock_wave.return_value.getframerate.return_value = 16000
            mock_wave.return_value.readframes.return_value = b"\x00\x00" * 100
            mock_wave.return_value.getnframes.return_value = 100
            result = app._play(b"RIFF....fake wav....")
        self.assertTrue(result)
        self.assertIsNone(app._current_player, "must be cleared after playback")

    def test_cancelled_mid_playback_returns_false(self):
        """Simulates Home being pressed during play(): the play() call
        itself sets _audio_stop_requested (standing in for a concurrent
        _stop_audio() call from the UI thread) - _play() must report the
        cancellation via its return value."""
        app = _make_app()

        def _during_play(_bytes):
            app._audio_stop_requested.set()

        mock_player = self._mock_audio_player(on_play=_during_play)
        with unittest.mock.patch(
            "pocketinfer.applications.nomad_right.app.AudioPlayer",
            return_value=mock_player,
        ), unittest.mock.patch("wave.open") as mock_wave:
            mock_wave.return_value.getframerate.return_value = 16000
            mock_wave.return_value.readframes.return_value = b"\x00\x00" * 100
            mock_wave.return_value.getnframes.return_value = 100
            result = app._play(b"RIFF....fake wav....")
        self.assertFalse(result)

    def test_playback_exception_returns_false_and_is_caught(self):
        app = _make_app()
        with unittest.mock.patch(
            "pocketinfer.applications.nomad_right.app.AudioPlayer",
            side_effect=RuntimeError("ffplay not found"),
        ):
            result = app._play(b"RIFF....fake wav....")
        self.assertFalse(result)
        self.assertIsNone(app._current_player)

    def test_current_player_tracked_during_playback_for_cancellation(self):
        """Verifies the mechanism _stop_audio() depends on: while play()
        is executing, self._current_player must already reference the
        live player (not just after _play() returns)."""
        app = _make_app()
        seen = {}

        def _during_play(_bytes):
            seen["player_during_play"] = app._current_player

        mock_player = self._mock_audio_player(on_play=_during_play)
        with unittest.mock.patch(
            "pocketinfer.applications.nomad_right.app.AudioPlayer",
            return_value=mock_player,
        ), unittest.mock.patch("wave.open") as mock_wave:
            mock_wave.return_value.getframerate.return_value = 16000
            mock_wave.return_value.readframes.return_value = b"\x00\x00" * 100
            mock_wave.return_value.getnframes.return_value = 100
            app._play(b"RIFF....fake wav....")
        self.assertIs(seen.get("player_during_play"), mock_player)


class TestModeSwitchingSequence(unittest.TestCase):
    """Section 29's explicit navigation sequence: Home -> Voice -> Home ->
    Scanner -> Home -> Voice -> Scanner, plus rapid repeated switching -
    verified via direct ui_cb()/_on_*_pressed() calls (the same entry
    points a real button press drives), asserting mode/state after each
    step and checking for resource leaks across repeated cycles."""

    def test_full_navigation_sequence(self):
        app = _make_app()

        self.assertEqual(app._mode, "HOME")

        # -> Voice (simulated: no dedicated "enter voice mode" button: it's
        # implicit in pressing the trigger button, handled by run() itself
        # rather than ui_cb - here we just verify Home is a no-op mid-idle).
        app._on_home_pressed()
        self.assertEqual(app._mode, "HOME")

        # -> Scanner
        app.ui_cb("Camera")
        self.assertEqual(app._mode, "DOCUMENT SCANNER")
        self.assertIsNotNone(app.pending_form_image)

        # -> Home (cancels the scan)
        app.ui_cb("Home")
        self.assertEqual(app._mode, "HOME")
        self.assertIsNone(app.pending_form_image)

        # -> Scanner again
        app.ui_cb("Camera")
        self.assertEqual(app._mode, "DOCUMENT SCANNER")

        # -> Home again
        app.ui_cb("Home")
        self.assertEqual(app._mode, "HOME")

    def test_repeated_scanner_home_cycles_do_not_leak_pending_image(self):
        """Home -> Scanner -> Home -> Scanner ... repeated 25x must never
        leave a stale pending_form_image or a stuck _camera_busy flag."""
        app = _make_app()
        for _ in range(25):
            app.ui_cb("Camera")
            self.assertIsNotNone(app.pending_form_image)
            self.assertFalse(app._camera_busy.is_set())
            app.ui_cb("Home")
            self.assertIsNone(app.pending_form_image)
            self.assertFalse(app._camera_busy.is_set())
        self.assertEqual(app._mode, "HOME")

    def test_rapid_camera_presses_do_not_deadlock_or_stack(self):
        """Rapid repeated Camera presses (no Home in between) - each must
        cleanly replace the previous pending photo, never hang."""
        app = _make_app()
        for i in range(10):
            app.board._camera_frame = f"frame-{i}".encode()
            app.ui_cb("Camera")
        self.assertEqual(app.pending_form_image, b"frame-9")
        self.assertFalse(app._camera_busy.is_set())

    def test_unrecognized_ui_messages_are_ignored_safely(self):
        """Settings-page language buttons and any future button names must
        not be mistaken for Home/Camera or raise."""
        app = _make_app()
        for msg in ["Settings", "Reset", "Reboot", "Shutdown", "ASR Hindi", "Bridge Tamil", ""]:
            app.ui_cb(msg)  # must not raise
        self.assertEqual(app._mode, "HOME")
        self.assertIsNone(app.pending_form_image)


class TestScreenTextSafety(unittest.TestCase):
    """Regression tests for the "overlay" bug: an earlier version used a
    checkmark glyph the LCD's bitmap font doesn't contain (silently
    contributes 0 width to text-wrap math while still occupying a
    character slot) and several strings well beyond this app's own
    established length precedent, on the *same* TextBox widgets shared
    with the answer text response.py already keeps short. Every literal
    screen string this module can produce must (a) use only glyphs the
    real LCD font actually has, and (b) stay within a safe length."""

    # Same font actually used for toptext/bottomtext on real hardware
    # (see ui/handheld.py's HandheldUI.HINDI_FONT) - loaded once for the
    # whole class since it's a real several-hundred-KB bitmap font file.
    _FONT = None

    # Generous but real safety margins: app.py's own pre-existing code
    # (untouched by this feature) already uses up to 80 chars for
    # top_text (f"You: {native_query}"[:80]) and 180 for bottom_text
    # (bottom_hint[:180]) successfully - these bounds are intentionally
    # looser than response.py's DISPLAY_HEADER_MAX_LEN(30)/BODY_MAX_LEN(60)
    # (which govern text embedded *into* those wider strings, not the
    # wrapping TextBox's own outer budget) so this test doesn't false-fail
    # on that already-proven-safe precedent.
    _MAX_TOP_LEN = 80
    _MAX_BOTTOM_LEN = 180
    _MAX_STATUSBAR_LEN = 52  # matches ui/handheld.py's statusbar placeholder width

    @classmethod
    def setUpClass(cls):
        from adafruit_bitmap_font import bitmap_font
        from importlib.resources import files
        cls._FONT = bitmap_font.load_font(
            str(files("pocketinfer.ui").joinpath("NotoSansDevanagari-Regular-12.pcf"))
        )

    def _assert_renders_safely(self, board_field, text, max_len):
        for ch in text:
            self.assertIsNotNone(
                self._FONT.get_glyph(ord(ch)),
                f"{board_field}={text!r} contains {ch!r} (U+{ord(ch):04X}), "
                f"which has no glyph in the LCD's bitmap font",
            )
        self.assertLessEqual(
            len(text), max_len,
            f"{board_field}={text!r} is {len(text)} chars, over the {max_len}-char safe budget",
        )

    def _drive_every_screen_and_collect(self):
        """Exercises every UI method that can set screen text and returns
        every (field, text) pair MockBoard recorded, across the full
        Camera/Home navigation surface."""
        app = _make_app()
        app.ui_cb("Camera")                 # -> capturing, then captured
        app.ui_cb("Home")                   # -> cancelled
        app.board._camera_raises = RuntimeError("no camera")
        app.ui_cb("Camera")                 # -> camera error
        app.board._camera_raises = None
        app.board._camera_frame = None
        app.ui_cb("Camera")                 # -> empty-frame error
        fake_player = unittest.mock.MagicMock()
        app._current_player = fake_player
        app.ui_cb("Home")                   # -> audio stopped
        return app.board.calls

    def test_every_screen_string_renders_safely(self):
        calls = self._drive_every_screen_and_collect()
        limits = {"top": self._MAX_TOP_LEN, "bottom": self._MAX_BOTTOM_LEN, "status": self._MAX_STATUSBAR_LEN}
        checked = 0
        for field, text in calls:
            if field == "mode" or not text:
                continue
            self._assert_renders_safely(field, text, limits[field])
            checked += 1
        self.assertGreater(checked, 0, "sanity: the drive above should have produced screen text to check")

    def test_home_hint_constant_renders_safely(self):
        self._assert_renders_safely("HOME_HINT", NomadRightApplication.HOME_HINT, self._MAX_BOTTOM_LEN)


class TestScreenAtomicity(unittest.TestCase):
    """A "screen update" (mode/top/bottom/status together, describing one
    coherent state) used to be assembled out of several separate
    mode_text()/top_text()/bottom_text()/statusbar() calls, serialized by a
    private app.py lock (_screen_lock) - which only covered app.py's own two
    call sites and did nothing for a concurrent caller elsewhere (e.g. the
    framework's memory_text() stats thread, see service.py). It's been
    replaced by board.update_screen(mode=..., top=..., bottom=..., status=...):
    every field goes out as ONE call. On real hardware that's one RPC round
    trip that HandheldUI.update_screen() applies as a single unit inside the
    UI subprocess's own serial dispatch loop (multiprocess_launch(), see its
    docstring) - nothing else can run partway through it, no matter who else
    is calling into the UI subprocess. These tests verify (a) app.py's own
    navigation/pipeline code never assembles a multi-field screen state out
    of separate calls any more, and (b) update_screen() itself only ever
    sets the fields it was actually given."""

    def test_navigation_never_calls_individual_setters_directly(self):
        """If any code path in app.py regressed to calling top_text()/
        bottom_text()/mode_text()/statusbar() directly instead of going
        through update_screen(), this fails loudly instead of silently
        reintroducing the interleaving window those individual calls open
        up. Drives the same full Camera/Home/voice-bridge navigation
        surface _drive_every_screen_and_collect() does elsewhere in this
        file."""
        app = _make_app()

        class NoDirectSetterBoard(MockBoard):
            def top_text(self, t):
                raise AssertionError("app.py called top_text() directly - should go through update_screen()")

            def bottom_text(self, t):
                raise AssertionError("app.py called bottom_text() directly - should go through update_screen()")

            def mode_text(self, t):
                raise AssertionError("app.py called mode_text() directly - should go through update_screen()")

            def statusbar(self, t):
                raise AssertionError("app.py called statusbar() directly - should go through update_screen()")

        app.board = NoDirectSetterBoard()
        app.ui_cb("Camera")
        app.ui_cb("Home")
        app.board._camera_raises = RuntimeError("no camera")
        app.ui_cb("Camera")
        app.board._camera_raises = None
        app.board._camera_frame = None
        app.ui_cb("Camera")
        fake_player = unittest.mock.MagicMock()
        fake_player.terminated = False
        app._current_player = fake_player
        app.ui_cb("Home")

        self.assertGreater(len(app.board.screen_updates), 0, "sanity: the drive above should have updated the screen")

    def test_update_screen_only_touches_given_fields(self):
        board = MockBoard()
        board.top = "existing top"
        board.bottom = "existing bottom"
        board.status = "existing status"
        board.mode = "existing mode"

        board.update_screen(status="[LISTENING]")

        self.assertEqual(board.status, "[LISTENING]")
        self.assertEqual(board.top, "existing top", "update_screen() must not touch fields it wasn't given")
        self.assertEqual(board.bottom, "existing bottom")
        self.assertEqual(board.mode, "existing mode")
        self.assertEqual(board.screen_updates[-1], {"mode": None, "top": None, "bottom": None, "status": "[LISTENING]"})

    def test_jetson_update_screen_sends_exactly_one_rpc_call(self):
        """boards/jetson.py's PocketInferDevboardUI.update_screen() must
        forward to self.UI as a single RPC call, not several - that single
        call is what the UI subprocess's serial dispatch loop treats as one
        atomic unit (see HandheldUI.update_screen()'s docstring)."""
        from pocketinfer.boards.jetson import PocketInferDevboardUI

        board = PocketInferDevboardUI.__new__(PocketInferDevboardUI)
        board.UI = unittest.mock.MagicMock()

        board.update_screen(mode="HOME", top="hello", bottom="world", status="[READY]")

        board.UI.update_screen.assert_called_once_with("HOME", "hello", "world", "[READY]")
        board.UI.top_text.assert_not_called()
        board.UI.bottom_text.assert_not_called()
        board.UI.mode_text.assert_not_called()
        board.UI.statusbar_text.assert_not_called()


def _run_all():
    print("=" * 60)
    print("  NOMADRIGHT — UI NAVIGATION / MODE-SWITCHING TESTS")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCameraButton)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHomeButton))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAsrLanguageGating))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopAudio))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlayCancellation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModeSwitchingSequence))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestScreenTextSafety))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestScreenAtomicity))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("=" * 60)
    total = result.testsRun
    fails = len(result.failures) + len(result.errors)
    print(f"  TOTAL : {total}   PASS : {total - fails}   FAIL : {fails}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(_run_all())
