"""
Integration tests wiring ui/touch_calibration.py into HandheldUI itself:
that check_touch() actually applies a loaded/fitted calibration before
dispatching to check_buttons(), and that calibrate_interactive() drives a
full tap-and-fit cycle against a scripted mock touch controller.

Uses a real HandheldUI against mocked display/touch objects, same pattern
as handheld_button_test.py - construction doesn't need real SPI hardware.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from pocketinfer.ui.handheld import HandheldUI
from pocketinfer.ui.touch_calibration import TouchCalibration


def _make_ui():
    display = MagicMock()
    touch = MagicMock()
    return HandheldUI(display, touch), touch


class TestCheckTouchAppliesCalibration(unittest.TestCase):
    def test_uncalibrated_dispatch_uses_raw_screen_point(self):
        ui, touch = _make_ui()
        seen = []
        ui.check_buttons = lambda x, y: seen.append((x, y))

        touch.is_pressed.return_value = True
        # _raw_screen_point does y, x = args; y = 240 - y - so a raw
        # reading of (100, 40) becomes screen point (40, 240 - 100) = (40, 140).
        touch.get_coordinates.return_value = (100, 40)

        ui.check_touch()

        self.assertEqual(seen, [(40, 140)])

    def test_calibration_correction_is_applied_before_dispatch(self):
        ui, touch = _make_ui()
        seen = []
        ui.check_buttons = lambda x, y: seen.append((x, y))
        ui._touch_calibration = TouchCalibration(ax=1.0, bx=-5, ay=1.0, by=10)

        touch.is_pressed.return_value = True
        touch.get_coordinates.return_value = (100, 40)  # raw_screen_point -> (40, 140)

        ui.check_touch()

        # (40 - 5, 140 + 10) = (35, 150)
        self.assertEqual(seen, [(35, 150)])


class TestLoadTouchCalibration(unittest.TestCase):
    def test_no_file_falls_back_to_identity(self):
        ui, _ = _make_ui()
        ui.load_touch_calibration("/nonexistent/path.json")
        self.assertEqual(ui._touch_calibration, TouchCalibration.identity())

    def test_loads_a_previously_saved_calibration(self):
        ui, _ = _make_ui()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cal.json")
            from pocketinfer.ui.touch_calibration import save
            saved = TouchCalibration(ax=1.1, bx=2.0, ay=0.9, by=-1.0)
            save(saved, path)

            ui.load_touch_calibration(path)

            self.assertEqual(ui._touch_calibration, saved)


class TestCalibrateInteractive(unittest.TestCase):
    """calibrate_interactive() end-to-end, with the physical tap-waiting
    loop (touch_calibration._wait_for_tap, real SPI polling on hardware)
    replaced by a canned sequence of raw screen points - one per target -
    so the test is deterministic and instant rather than timing-based."""

    def test_full_tap_cycle_fits_and_persists_a_correction(self):
        ui, _touch = _make_ui()
        ui.display.width = 320
        ui.display.height = 240

        # Every real device is 8px right / 4px down of where it should be
        # (constant offset, no scale error) - calibrate_interactive() must
        # recover exactly that from taps at the five default targets.
        OFFSET_X, OFFSET_Y = 8, -4
        raw_points_in_target_order = [
            (tx + OFFSET_X, ty + OFFSET_Y)
            for tx, ty in [(24, 24), (296, 24), (24, 216), (296, 216), (160, 120)]  # default_targets(320, 240)
        ]

        with unittest.mock.patch(
            "pocketinfer.ui.touch_calibration._wait_for_tap",
            side_effect=raw_points_in_target_order,
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "cal.json")
                ui._touch_calibration_path = path
                calibration = ui.calibrate_interactive(save_result=True)

                self.assertAlmostEqual(calibration.ax, 1.0, delta=1e-6)
                self.assertAlmostEqual(calibration.bx, -OFFSET_X, delta=1e-6)
                self.assertAlmostEqual(calibration.ay, 1.0, delta=1e-6)
                self.assertAlmostEqual(calibration.by, -OFFSET_Y, delta=1e-6)
                self.assertEqual(ui._touch_calibration, calibration)
                self.assertTrue(os.path.exists(path), "calibrate_interactive(save_result=True) must persist the fit")

                # The fitted correction is live immediately - a tap at a
                # raw point now dispatches to the intended screen target.
                seen = []
                ui.check_buttons = lambda x, y: seen.append((x, y))
                _touch.is_pressed.return_value = True
                # raw_screen_point of (240 - (24 + OFFSET_Y), 24 + OFFSET_X)
                # is (24 + OFFSET_X, 24 + OFFSET_Y) - the first raw sample.
                _touch.get_coordinates.return_value = (240 - (24 + OFFSET_Y), 24 + OFFSET_X)
                ui.check_touch()
                self.assertEqual(seen, [(24, 24)])

    def test_save_result_false_does_not_write_a_file(self):
        ui, _touch = _make_ui()
        ui.display.width = 320
        ui.display.height = 240
        raw_points = [(tx, ty) for tx, ty in [(24, 24), (296, 24), (24, 216), (296, 216), (160, 120)]]

        with unittest.mock.patch(
            "pocketinfer.ui.touch_calibration._wait_for_tap", side_effect=raw_points
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "cal.json")
                ui._touch_calibration_path = path
                ui.calibrate_interactive(save_result=False)
                self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
