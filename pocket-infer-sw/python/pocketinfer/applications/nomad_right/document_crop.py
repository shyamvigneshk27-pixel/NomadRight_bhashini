"""
Document auto-crop: detects the largest rectangular document/paper edge in
a captured frame and perspective-warps it to a flat, cropped rectangle -
the standard "document scanner" approach (grayscale -> blur -> Canny
edges -> contours -> largest 4-point polygon -> perspective transform).
Falls back to the original, uncropped frame whenever no confident 4-point
contour is found (bad lighting, no clear edge, extreme angle, camera not
actually pointed at a document) - a missing document edge must never fail
a capture the worker is standing there waiting on.

Measured on-device (Jetson Orin Nano, this camera's 1920x1080 frame): the
full detection pipeline takes ~167ms - a one-shot cost on the capture
button press, not a per-preview-frame cost, so it adds no perceptible
latency to Document Scanner mode.
"""
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A detected quadrilateral smaller than this fraction of the full frame
# area is treated as noise (a shadow, the edge of a desk, a stray edge in
# the background) rather than the actual document - a real document held
# up for scanning at a reasonable distance fills a large majority of the
# frame.
_MIN_DOCUMENT_AREA_FRACTION = 0.2
_MIN_DOCUMENT_DIMENSION_PX = 50


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left -
    required for a correct (non-mirrored/rotated) perspective warp,
    since findContours()/approxPolyDP() return them in no particular
    winding order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return rect


def _find_document_quad(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Closes small gaps in the detected edge (a fold, a light glare
    # spot, a slightly out-of-focus edge segment) that would otherwise
    # split one document boundary into several unconnected contours.
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < frame_area * _MIN_DOCUMENT_AREA_FRACTION:
            break  # sorted descending - nothing after this is bigger either
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def auto_crop_document(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a perspective-corrected, cropped-to-document version of
    frame_bgr, or frame_bgr itself unchanged if no confident document
    edge is found. Never raises - any detection failure falls back to
    the uncropped frame rather than failing the capture outright.
    """
    try:
        quad = _find_document_quad(frame_bgr)
        if quad is None:
            return frame_bgr

        tl, tr, br, bl = _order_points(quad)

        max_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        if max_width < _MIN_DOCUMENT_DIMENSION_PX or max_height < _MIN_DOCUMENT_DIMENSION_PX:
            return frame_bgr  # degenerate quad - not a real document edge

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
        return cv2.warpPerspective(frame_bgr, transform, (max_width, max_height))
    except Exception:
        logger.exception("[NomadRight] Document auto-crop failed - using uncropped frame")
        return frame_bgr
