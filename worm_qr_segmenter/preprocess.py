from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from skimage import measure, morphology

from .config import Config
from .qr import QRResult


def resize_rgb(img_rgb: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize an RGB image for the working-pixel pipeline."""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")
    if abs(scale_factor - 1.0) < 1e-12:
        return img_rgb

    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(
        img_rgb,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=interpolation,
    )


def scale_qr_geometry(qr: QRResult, scale_factor: float) -> QRResult:
    """Return a QRResult whose geometry is mapped to working-image pixels.

    The decoded text and parsed metadata are left unchanged. Only points and
    bbox coordinates are scaled.
    """
    if not qr.detected or abs(scale_factor - 1.0) < 1e-12:
        return qr

    points = None
    if qr.points:
        points = (np.asarray(qr.points, dtype=float) * scale_factor).tolist()

    bbox = None
    if qr.bbox_xyxy:
        x0, y0, x1, y1 = qr.bbox_xyxy
        bbox = [
            int(np.floor(x0 * scale_factor)),
            int(np.floor(y0 * scale_factor)),
            int(np.ceil(x1 * scale_factor)),
            int(np.ceil(y1 * scale_factor)),
        ]

    return QRResult(
        detected=qr.detected,
        text=qr.text,
        method=qr.method,
        points=points,
        bbox_xyxy=bbox,
        parsed=qr.parsed,
    )


def make_valid_segmentation_region(img_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Detect the bright tray interior used as the valid segmentation domain.

    The mask is intentionally based on a strongly blurred luminance image. This
    keeps small dark worms inside the valid domain while excluding large dark or
    grey regions such as the QR card, black background, tray frame, ruler, and
    other non-tray areas.
    """
    H, W = img_rgb.shape[:2]
    if not cfg.exclude_dark_regions_from_segmentation:
        return np.ones((H, W), dtype=bool)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blur_px = max(3, int(cfg.valid_region_blur_px))
    if blur_px % 2 == 0:
        blur_px += 1
    background = cv2.GaussianBlur(gray, (blur_px, blur_px), 0)

    valid = background >= int(cfg.min_valid_background_luma)

    close_px = max(0, int(cfg.valid_region_close_px))
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # Remove tiny bright islands such as labels, QR finder fragments, and dust.
    min_object_area = max(64, int(cfg.valid_region_min_area_fraction * H * W))
    # scikit-image changed these keyword names in recent releases.
    try:
        valid = morphology.remove_small_objects(valid, max_size=min_object_area)
    except TypeError:
        valid = morphology.remove_small_objects(valid, min_size=min_object_area)
    try:
        valid = morphology.remove_small_holes(valid, max_size=max(64, min_object_area // 4))
    except TypeError:
        valid = morphology.remove_small_holes(valid, area_threshold=max(64, min_object_area // 4))

    if cfg.valid_region_keep_largest:
        labels = measure.label(valid)
        if labels.max() > 0:
            regions = measure.regionprops(labels)
            largest = max(regions, key=lambda r: r.area).label
            valid = labels == largest

    erode_px = max(0, int(cfg.valid_region_erode_px))
    if erode_px > 0 and valid.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)

    return valid.astype(bool)
