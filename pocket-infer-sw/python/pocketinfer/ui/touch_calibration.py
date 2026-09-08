"""
Empirical correction for XPT2046 touch coordinates.

check_touch() in handheld.py turns a raw XPT2046 reading into a screen point
via a fixed axis-swap-and-flip transform tuned for this panel's physical
rotation (see HandheldUI._raw_screen_point()'s docstring). That transform
assumes the touch controller's own raw-to-normalized scaling
(xpt2046_circuitpython's x_min/x_max/y_min/y_max, left at the library's
generic defaults) exactly matches this specific physical panel. It doesn't,
in general - XPT2046 units vary enough in real ADC range that the same
generic bounds on two "identical" boards can land a tap several pixels off,
worse near the edges.

Rather than re-deriving the library's raw ADC bounds (which would mean
reading raw, pre-normalization samples and reasoning about the axis swap in
ADC space), this module fits a small correction *after* the existing
pipeline: one linear term (scale + offset) per axis, computed from a
handful of "tap this labelled point" samples taken with the existing
pipeline still in effect. Perfectly calibrated hardware yields ax=ay=1,
bx=by=0 - the identity - so an uncalibrated device behaves exactly as it
does today.
"""
import json
import os
from typing import List, NamedTuple, Optional, Tuple

DEFAULT_CALIBRATION_PATH = os.path.expanduser("~/.pocketinfer/touch_calibration.json")

# Screen-corner targets are inset from the true edge - a tap right at the
# physical bezel is where XPT2046 panels are least linear, so calibrating
# there would fit the correction to the least representative samples.
DEFAULT_TARGET_MARGIN = 24


class TouchCalibration(NamedTuple):
    ax: float = 1.0
    bx: float = 0.0
    ay: float = 1.0
    by: float = 0.0

    @classmethod
    def identity(cls) -> "TouchCalibration":
        return cls()

    def apply(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(self.ax * x + self.bx)), int(round(self.ay * y + self.by))

    def to_dict(self) -> dict:
        return {"ax": self.ax, "bx": self.bx, "ay": self.ay, "by": self.by}

    @classmethod
    def from_dict(cls, d: dict) -> "TouchCalibration":
        return cls(ax=d["ax"], bx=d["bx"], ay=d["ay"], by=d["by"])


def load(path: str = DEFAULT_CALIBRATION_PATH) -> TouchCalibration:
    """Load a saved calibration, or the identity if none exists / it can't
    be read. A missing or corrupt file must never block startup - worst
    case is today's uncalibrated behavior, not a crash."""
    try:
        with open(path, "r") as f:
            return TouchCalibration.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return TouchCalibration.identity()


def save(calibration: TouchCalibration, path: str = DEFAULT_CALIBRATION_PATH) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibration.to_dict(), f)


def default_targets(width: int, height: int, margin: int = DEFAULT_TARGET_MARGIN) -> List[Tuple[int, int]]:
    """Four corner points, inset by margin, plus the center - five samples
    is enough to average out a noisy tap and still cheaply least-squares-fit
    a straight line per axis in fit()."""
    return [
        (margin, margin),
        (width - margin, margin),
        (margin, height - margin),
        (width - margin, height - margin),
        (width // 2, height // 2),
    ]


def fit(samples: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> TouchCalibration:
    """samples: [((raw_x, raw_y), (target_x, target_y)), ...] where raw_x/
    raw_y are what today's uncalibrated pipeline (HandheldUI._raw_screen_point)
    produced for a tap the worker was asked to place at (target_x, target_y).
    Fits target = a*raw + b independently per axis - a plain 1D linear
    regression, exact through 2 points and an averaging fit through more."""
    if len(samples) < 2:
        raise ValueError("Need at least 2 calibration samples to fit ax/bx and ay/by")
    ax, bx = _fit_axis([raw[0] for raw, _ in samples], [tgt[0] for _, tgt in samples])
    ay, by = _fit_axis([raw[1] for raw, _ in samples], [tgt[1] for _, tgt in samples])
    return TouchCalibration(ax=ax, bx=bx, ay=ay, by=by)


def _fit_axis(raw_values: List[float], target_values: List[float]) -> Tuple[float, float]:
    n = len(raw_values)
    mean_raw = sum(raw_values) / n
    mean_target = sum(target_values) / n
    numerator = sum((r - mean_raw) * (t - mean_target) for r, t in zip(raw_values, target_values))
    denominator = sum((r - mean_raw) ** 2 for r in raw_values)
    if denominator == 0:
        # Every raw sample landed on the same point (e.g. a stuck reading) -
        # there's nothing to fit a slope from. Fall back to identity scale
        # and just correct the offset, rather than dividing by zero.
        return 1.0, mean_target - mean_raw
    a = numerator / denominator
    b = mean_target - a * mean_raw
    return a, b


def _wait_for_tap(ui, settle_reads: int = 3, poll_interval: float = 0.02) -> Tuple[int, int]:
    """Block until a fresh touch-down is detected and its (pre-calibration)
    screen point settles, then return it. Requires a prior release (mirrors
    check_touch()'s own edge detection) so back-to-back targets need a
    distinct lift-and-tap, not one finger dragged across all of them."""
    import time
    while ui.touch.is_pressed():
        time.sleep(poll_interval)
    while not ui.touch.is_pressed():
        time.sleep(poll_interval)
    last = None
    stable = 0
    while stable < settle_reads:
        if not ui.touch.is_pressed():
            stable = 0
            continue
        args = ui.touch.get_coordinates()
        if args is None:
            continue
        point = ui._raw_screen_point(args)
        if point == last:
            stable += 1
        else:
            stable = 1
            last = point
        time.sleep(poll_interval)
    return last


def run_interactive_calibration(ui, targets: Optional[List[Tuple[int, int]]] = None) -> TouchCalibration:
    """Walks a technician through tapping `targets` (default:
    default_targets() sized to ui's own display) one at a time, prompting via
    ui's own top_text/bottom_text and sampling through
    ui._raw_screen_point() - the exact pre-calibration pipeline
    check_touch() otherwise uses, so the fit corrects precisely what's
    currently wrong. Returns the fitted calibration; does not persist it
    (see save()) or apply it to `ui` (see HandheldUI.calibrate_interactive()).

    Blocks on physical taps - intended for a deliberate maintenance call
    (e.g. `board.UI.calibrate_interactive()` from a maintenance shell, which
    an RPC round trip carries straight into the UI subprocess), never the
    hot path."""
    width = getattr(ui.display, "width", 320)
    height = getattr(ui.display, "height", 240)
    targets = targets or default_targets(width, height)
    samples = []
    for i, (tx, ty) in enumerate(targets):
        ui.top_text(f"CALIBRATE {i + 1}/{len(targets)}")
        ui.bottom_text(f"Tap the marker at ({tx},{ty})")
        raw = _wait_for_tap(ui)
        samples.append((raw, (tx, ty)))
    ui.top_text("CALIBRATION DONE")
    ui.bottom_text("")
    return fit(samples)
