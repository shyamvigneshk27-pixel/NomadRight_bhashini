"""
Unit tests for the pure math in ui/touch_calibration.py - the linear
correction fitted from "tap this labelled point" samples that corrects for
XPT2046 panels being left at the library's generic, uncalibrated ADC bounds
(see that module's docstring for the full story). No hardware needed: fit()/
TouchCalibration.apply() are plain arithmetic, and load()/save() only touch
a throwaway temp file.
"""
import json
import os
import tempfile
import unittest

from pocketinfer.ui.touch_calibration import (
    TouchCalibration,
    default_targets,
    fit,
    load,
    save,
)


class TestTouchCalibrationIdentity(unittest.TestCase):
    def test_identity_leaves_points_unchanged(self):
        cal = TouchCalibration.identity()
        self.assertEqual(cal.apply(10, 20), (10, 20))
        self.assertEqual(cal.apply(0, 0), (0, 0))
        self.assertEqual(cal.apply(319, 239), (319, 239))

    def test_identity_is_the_default_construction(self):
        self.assertEqual(TouchCalibration(), TouchCalibration.identity())


class TestFit(unittest.TestCase):
    def test_two_point_fit_is_exact(self):
        # A device reading everything 10px too far right and 5px too far
        # down, with no scale error: target = raw - 10 (x), raw - 5 (y).
        samples = [
            ((10, 5), (0, 0)),
            ((330, 245), (320, 240)),
        ]
        cal = fit(samples)
        self.assertAlmostEqual(cal.apply(10, 5)[0], 0)
        self.assertAlmostEqual(cal.apply(10, 5)[1], 0)
        self.assertAlmostEqual(cal.apply(330, 245)[0], 320)
        self.assertAlmostEqual(cal.apply(330, 245)[1], 240)
        # And a point never sampled directly still falls on the same line.
        x, y = cal.apply(170, 125)
        self.assertAlmostEqual(x, 160, delta=1)
        self.assertAlmostEqual(y, 120, delta=1)

    def test_scaled_and_offset_axes_recovered(self):
        # Simulates a panel whose raw range is compressed/offset relative to
        # the screen: raw x in [50, 1550] should map to screen x in [0, 320].
        def raw_x(screen_x):
            return 50 + screen_x * (1500 / 320)

        def raw_y(screen_y):
            return 100 + screen_y * (1900 / 240)

        targets = default_targets(320, 240)
        samples = [((raw_x(tx), raw_y(ty)), (tx, ty)) for tx, ty in targets]
        cal = fit(samples)

        for tx, ty in targets:
            x, y = cal.apply(raw_x(tx), raw_y(ty))
            self.assertAlmostEqual(x, tx, delta=1)
            self.assertAlmostEqual(y, ty, delta=1)

    def test_noisy_samples_average_out(self):
        # Same underlying linear relationship as the exact case, plus small
        # symmetric +/-jitter per point - the least-squares fit should still
        # recover essentially the same line, since the jitter cancels out.
        exact = [
            ((10, 5), (0, 0)),
            ((170, 125), (160, 120)),
            ((330, 245), (320, 240)),
        ]
        noisy = (
            [((rx + 3, ry - 3), t) for (rx, ry), t in exact]
            + [((rx - 3, ry + 3), t) for (rx, ry), t in exact]
        )
        cal = fit(noisy)
        x, y = cal.apply(10, 5)
        self.assertAlmostEqual(x, 0, delta=5)
        self.assertAlmostEqual(y, 0, delta=5)

    def test_degenerate_samples_do_not_raise_or_divide_by_zero(self):
        # Every raw reading identical (e.g. a stuck/noisy controller) - fit()
        # must fall back to an offset-only correction, not crash.
        samples = [((100, 100), (0, 0)), ((100, 100), (320, 240))]
        cal = fit(samples)
        self.assertEqual(cal.ax, 1.0)
        self.assertEqual(cal.ay, 1.0)

    def test_requires_at_least_two_samples(self):
        with self.assertRaises(ValueError):
            fit([((10, 5), (0, 0))])


class TestDefaultTargets(unittest.TestCase):
    def test_five_points_within_bounds(self):
        targets = default_targets(320, 240, margin=24)
        self.assertEqual(len(targets), 5)
        for x, y in targets:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 320)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 240)


class TestLoadSave(unittest.TestCase):
    def test_missing_file_returns_identity(self):
        cal = load("/nonexistent/path/does-not-exist.json")
        self.assertEqual(cal, TouchCalibration.identity())

    def test_corrupt_file_returns_identity(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name
        try:
            cal = load(path)
            self.assertEqual(cal, TouchCalibration.identity())
        finally:
            os.unlink(path)

    def test_save_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "touch_calibration.json")
            original = TouchCalibration(ax=1.05, bx=-3.2, ay=0.97, by=2.1)
            save(original, path)
            self.assertTrue(os.path.exists(path))
            loaded = load(path)
            self.assertEqual(loaded, original)

    def test_saved_file_is_plain_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cal.json")
            save(TouchCalibration(ax=2.0, bx=1.0, ay=3.0, by=4.0), path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data, {"ax": 2.0, "bx": 1.0, "ay": 3.0, "by": 4.0})


if __name__ == "__main__":
    unittest.main()
