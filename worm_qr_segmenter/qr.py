from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass
class QRResult:
    detected: bool
    text: str = ""
    method: str = "none"
    points: list[list[float]] | None = None  # [[x,y], ...] in full-image coordinates
    bbox_xyxy: list[int] | None = None       # [x0, y0, x1, y1] in full-image coordinates
    parsed: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        parsed = record.pop("parsed") or {}
        for key, value in parsed.items():
            record[f"qr_{key}"] = value
        return record


_EXAMPLE_PATTERN = re.compile(
    r"^Plot(?P<plot>\d+)_Spalte(?P<spalte>\d+)_Reihe(?P<reihe>\d+)_(?P<condition>[A-Za-z0-9]+)_(?P<sample_id>\d+)_?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QRCandidate:
    name: str
    bbox_xyxy: tuple[int, int, int, int]
    score: float = 0.0


def parse_qr_text(text: str) -> dict[str, Any]:
    """Parse known sample-code formats while keeping a generic fallback.

    Example QR text decoded from the provided sample:
        Plot203_Spalte4_Reihe23_R4S_448_

    Parsed fields are intentionally simple and non-destructive. The raw QR text
    remains the authoritative metadata field.
    """
    raw = text.strip()
    parsed: dict[str, Any] = {"raw": raw}

    match = _EXAMPLE_PATTERN.match(raw)
    if match:
        groups = match.groupdict()
        parsed.update(
            {
                "format": "plot_spalte_reihe_condition_sample",
                "plot": int(groups["plot"]),
                "spalte": int(groups["spalte"]),
                "reihe": int(groups["reihe"]),
                "condition": groups["condition"],
                "sample_id": int(groups["sample_id"]),
            }
        )
        return parsed

    # Generic fallback: expose underscore-separated tokens without guessing.
    tokens = [tok for tok in raw.strip("_").split("_") if tok]
    parsed["format"] = "generic_underscore_tokens" if tokens else "unparsed"
    parsed["token_count"] = len(tokens)
    for idx, token in enumerate(tokens):
        parsed[f"token_{idx}"] = token
    return parsed


def decode_qr(img_rgb: np.ndarray, try_harder: bool = True) -> QRResult:
    """Decode the first readable QR code in an RGB image.

    The primary backend is OpenCV's QRCodeDetector, which has no native zbar
    dependency. The function first tries the whole image. With try_harder=True
    it then tries likely QR crops, contrast-normalized crops, thresholded crops,
    and stronger upscaling. If pyzbar is installed, it remains a final fallback.
    """
    # Whole image first. This is fast and often gives the cleanest geometry.
    result = _decode_qr_opencv_region(img_rgb, offset_xy=(0, 0), region_name="full", try_harder=False)
    if result.detected:
        return result

    if try_harder:
        # Try crops before very aggressive full-image preprocessing. QR codes in
        # these photos are often small relative to the full frame, so cropping
        # gives the detector a much easier problem and allows stronger upscaling.
        for cand in find_qr_candidates(img_rgb):
            x0, y0, x1, y1 = cand.bbox_xyxy
            crop = img_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            # Texture candidates are small and likely, so use the stronger QR
            # preprocessing. Broad right-side fallback windows are intentionally
            # included to avoid cutting off a code near the border; allow the
            # stronger pass once the crop is not too large, especially on the
            # working-scale image.
            crop_h, crop_w = crop.shape[:2]
            candidate_try_harder = (not cand.name.startswith("right_")) or max(crop_h, crop_w) <= 1200
            result = _decode_qr_opencv_region(
                crop,
                offset_xy=(x0, y0),
                region_name=cand.name,
                try_harder=candidate_try_harder,
            )
            if result.detected:
                return result

    # Optional zbar fallback. Useful on some machines, but it requires the zbar
    # native library in addition to the Python package.
    return _decode_qr_pyzbar(img_rgb)


def _decode_qr_opencv_region(
    img_rgb: np.ndarray,
    offset_xy: tuple[int, int],
    region_name: str,
    try_harder: bool,
) -> QRResult:
    detector = cv2.QRCodeDetector()
    H, W = img_rgb.shape[:2]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    variants: list[tuple[str, np.ndarray]] = [("rgb", img_rgb), ("gray", gray)]

    if try_harder:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe", clahe))

        # QR modules are black-on-light in the current images. Binary variants
        # often rescue cases where the grey card or illumination gradient reduces
        # contrast enough that OpenCV detects a candidate but cannot decode it.
        try:
            _thr, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(("otsu", otsu))
        except Exception:
            pass

        # Adaptive threshold needs an odd block size and should be smaller than
        # the crop. Keep it conservative; too small a block breaks the finder
        # patterns.
        block = max(31, int(round(min(H, W) / 6)))
        if block % 2 == 0:
            block += 1
        block = min(block, max(31, (min(H, W) // 2) * 2 - 1))
        if max(H, W) < 1400 and block >= 31 and min(H, W) > block:
            try:
                adap = cv2.adaptiveThreshold(
                    clahe,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block,
                    5,
                )
                variants.append(("adaptive", adap))
            except Exception:
                pass

    # The allowed upscale is crop-size dependent. Avoid making huge full-frame
    # images, but allow strong upscaling for small QR candidates.
    if try_harder:
        if max(H, W) < 450:
            scales = (1.0, 2.0, 3.0, 4.0, 6.0)
        elif max(H, W) < 900:
            scales = (1.0, 1.5, 2.0, 3.0)
        elif max(H, W) < 1400:
            scales = (1.0, 1.5)
        else:
            scales = (1.0,)
    else:
        scales = (1.0, 2.0, 3.0) if max(H, W) < 1200 else (1.0, 1.5, 2.0)

    for variant_name, variant in variants:
        for scale in scales:
            if abs(scale - 1.0) < 1e-12:
                scaled = variant
            else:
                interp = cv2.INTER_NEAREST if variant.ndim == 2 else cv2.INTER_CUBIC
                scaled = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=interp)

            method = f"opencv_{region_name}_{variant_name}_scale_{scale:g}"
            result = _try_opencv_detector(detector, scaled, scale=scale, offset_xy=offset_xy, method=method)
            if result.detected:
                return result

    return QRResult(detected=False)


def _try_opencv_detector(
    detector: cv2.QRCodeDetector,
    img: np.ndarray,
    scale: float,
    offset_xy: tuple[int, int],
    method: str,
) -> QRResult:
    ox, oy = offset_xy

    # Multi-code path first. Some OpenCV builds raise on certain binary images,
    # so keep this guarded.
    try:
        ok, decoded_info, points, _straight = detector.detectAndDecodeMulti(img)
        if ok and decoded_info is not None and points is not None:
            for text, pts in zip(decoded_info, points):
                if text:
                    pts = _map_points_to_full_image(pts, scale, ox, oy)
                    return _result_from_points(text=text, points=pts, method=method)
    except Exception:
        pass

    try:
        text, pts, _straight = detector.detectAndDecode(img)
    except Exception:
        return QRResult(detected=False)

    if text:
        pts = _map_points_to_full_image(pts, scale, ox, oy)
        return _result_from_points(text=text, points=pts, method=method)

    return QRResult(detected=False)


def _map_points_to_full_image(points: Any, scale: float, ox: int, oy: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.size == 0:
        return pts.reshape(0, 2)
    pts = pts / scale
    pts[:, 0] += ox
    pts[:, 1] += oy
    return pts


def find_qr_candidates(img_rgb: np.ndarray, max_candidates: int = 12) -> list[QRCandidate]:
    """Return likely QR-code crop boxes in full-image coordinates.

    The decoder always tries the full image first. Candidate crops are only a
    second stage. To avoid cutting off QR codes that sit close to the right image
    border, this function deliberately includes broad right-card windows in
    addition to tight texture-derived boxes. The broad windows are slower but
    they are much safer than square crops when the QR quiet zone is partly near
    the frame edge.
    """
    H, W = img_rgb.shape[:2]

    def clamp_box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int] | None:
        x0 = max(0, min(W - 1, int(x0)))
        y0 = max(0, min(H - 1, int(y0)))
        x1 = max(x0 + 1, min(W, int(x1)))
        y1 = max(y0 + 1, min(H, int(y1)))
        if (x1 - x0) < 40 or (y1 - y0) < 40:
            return None
        return x0, y0, x1, y1

    def make(name: str, x0: int, y0: int, x1: int, y1: int, score: float) -> QRCandidate | None:
        box = clamp_box(x0, y0, x1, y1)
        if box is None:
            return None
        return QRCandidate(name=name, bbox_xyxy=box, score=float(score))

    # Layout priors for the present acquisition geometry. These are intentionally
    # broad, so they cannot cut off the code even when the right border is close.
    # They are added before texture boxes and preserved during de-duplication.
    layout_candidates = [
        make("right_strip_full_height", int(0.62 * W), 0, W, H, 1.00),
        make("right_card_upper_large", int(0.66 * W), int(0.04 * H), W, int(0.58 * H), 0.95),
        make("right_card_middle_large", int(0.66 * W), int(0.18 * H), W, int(0.74 * H), 0.90),
        make("right_upper_window", int(0.70 * W), int(0.08 * H), W, int(0.45 * H), 0.65),
        make("right_mid_window", int(0.70 * W), int(0.24 * H), W, int(0.62 * H), 0.60),
        make("right_lower_window", int(0.70 * W), int(0.40 * H), W, int(0.78 * H), 0.55),
    ]
    layout_candidates = [c for c in layout_candidates if c is not None]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # High-contrast dark structures. Adaptive threshold is better than a global
    # threshold because tray, label card, and background have different luminance.
    block = max(31, int(round(min(H, W) / 18)))
    if block % 2 == 0:
        block += 1
    try:
        dark = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            7,
        )
    except Exception:
        _thr, dark = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect QR modules into one component. Keep the kernel conservative; too
    # much dilation can merge QR modules with printed text on the card.
    k = max(3, int(round(min(H, W) * 0.003)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    connected = cv2.dilate(dark, kernel, iterations=1)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(connected, connectivity=8)
    edges = cv2.Canny(clahe, 50, 150)

    texture_candidates: list[QRCandidate] = []
    img_area = H * W
    for label in range(1, n):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        if w < 35 or h < 35:
            continue
        if w * h > 0.45 * img_area:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.8:
            continue

        # Expand to include the quiet zone. If the square box would be clipped by
        # the image border, the broad right-card windows above still provide a
        # non-cut crop for the decoder.
        side = int(max(w, h) * 1.85)
        cx = x + w / 2
        cy = y + h / 2
        x0 = int(round(cx - side / 2))
        y0 = int(round(cy - side / 2))
        x1 = x0 + side
        y1 = y0 + side
        box = clamp_box(x0, y0, x1, y1)
        if box is None:
            continue
        x0, y0, x1, y1 = box

        crop_dark = dark[y0:y1, x0:x1]
        crop_edges = edges[y0:y1, x0:x1]
        box_area = max(1, (x1 - x0) * (y1 - y0))
        dark_density = float(np.count_nonzero(crop_dark)) / box_area
        edge_density = float(np.count_nonzero(crop_edges)) / box_area
        square_score = min(w, h) / max(w, h)
        size_score = np.sqrt(w * h) / max(np.sqrt(img_area), 1.0)
        right_bias = 0.25 if x > 0.55 * W else 0.0
        score = (1.0 + 10.0 * dark_density + 6.0 * edge_density) * square_score + 2.0 * size_score + right_bias
        texture_candidates.append(QRCandidate(name=f"texture_{label}", bbox_xyxy=(x0, y0, x1, y1), score=score))

    texture_candidates = sorted(texture_candidates, key=lambda c: c.score, reverse=True)

    # Preserve broad right-card candidates, then add the strongest texture boxes.
    ordered = layout_candidates + texture_candidates
    unique: list[QRCandidate] = []
    for cand in ordered:
        # Do not let a tight texture box remove a broad right-card window. The
        # broad boxes are explicitly there to avoid cutting off the QR code.
        iou_threshold = 0.92 if cand.name.startswith("right_") else 0.88
        if all(_bbox_iou(cand.bbox_xyxy, old.bbox_xyxy) < iou_threshold for old in unique):
            unique.append(cand)
        if len(unique) >= max_candidates:
            break
    return unique

def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def _decode_qr_pyzbar(img_rgb: np.ndarray) -> QRResult:
    try:
        from pyzbar.pyzbar import ZBarSymbol, decode
    except Exception:
        return QRResult(detected=False)

    try:
        decoded = decode(img_rgb, symbols=[ZBarSymbol.QRCODE])
    except Exception:
        return QRResult(detected=False)

    if not decoded:
        return QRResult(detected=False)

    item = decoded[0]
    text = item.data.decode("utf-8", errors="replace")
    rect = item.rect
    x0, y0 = int(rect.left), int(rect.top)
    x1, y1 = int(rect.left + rect.width), int(rect.top + rect.height)
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return QRResult(
        detected=True,
        text=text,
        method="pyzbar",
        points=points,
        bbox_xyxy=[x0, y0, x1, y1],
        parsed=parse_qr_text(text),
    )


def _result_from_points(text: str, points: np.ndarray, method: str) -> QRResult:
    if points.size == 0:
        return QRResult(detected=True, text=text, method=method, parsed=parse_qr_text(text))
    x0 = int(np.floor(points[:, 0].min()))
    y0 = int(np.floor(points[:, 1].min()))
    x1 = int(np.ceil(points[:, 0].max()))
    y1 = int(np.ceil(points[:, 1].max()))
    return QRResult(
        detected=True,
        text=text,
        method=method,
        points=points.astype(float).tolist(),
        bbox_xyxy=[x0, y0, x1, y1],
        parsed=parse_qr_text(text),
    )


def qr_exclusion_mask(shape_hw: tuple[int, int], qr: QRResult, pad_px: int = 10) -> np.ndarray:
    """Return a boolean full-image mask covering the QR code plus padding."""
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=bool)
    if not qr.detected:
        return mask

    if qr.points:
        pts = np.asarray(qr.points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        tmp = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(tmp, [pts], 1)
        mask = tmp.astype(bool)
    elif qr.bbox_xyxy:
        x0, y0, x1, y1 = qr.bbox_xyxy
        mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True

    if pad_px > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad_px + 1, 2 * pad_px + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask
