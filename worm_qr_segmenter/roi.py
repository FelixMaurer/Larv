from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def detect_tray_roi(
    img_rgb: np.ndarray,
    inset: int = 15,
    debug_path: Path | None = None,
) -> tuple[int, int, int, int]:
    """Detect the tray ROI by finding the dark rectangular tray frame.

    Returns y0, y1, x0, x1 in full-image coordinates. If no plausible tray is
    found, a conservative centered crop is returned.
    """
    H, W = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _center_fallback(H, W)

    img_area = H * W
    best: tuple[int, int, int, int] | None = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.15 * img_area or area > 0.95 * img_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area else 0.0
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 2.8:
            continue

        score = area * rectangularity
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return _center_fallback(H, W)

    x, y, w, h = best
    y0 = max(0, y + inset)
    y1 = min(H, y + h - inset)
    x0 = max(0, x + inset)
    x1 = min(W, x + w - inset)

    if (y1 - y0) * (x1 - x0) < 0.3 * img_area:
        return _center_fallback(H, W)

    if debug_path is not None:
        vis = img_rgb.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 8)
        cv2.imwrite(str(debug_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return y0, y1, x0, x1


def _center_fallback(H: int, W: int) -> tuple[int, int, int, int]:
    margin_y = int(H * 0.08)
    margin_x = int(W * 0.08)
    return margin_y, H - margin_y, margin_x, W - margin_x
