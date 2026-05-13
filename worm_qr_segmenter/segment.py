from __future__ import annotations

import cv2
import numpy as np
from skimage import measure, morphology

from .config import Config
from .measure import measure_skeleton_length


def run_segmentation(img_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Run the configured instance segmentation backend."""
    if cfg.backend == "cellpose_sam":
        return _run_cellpose_sam(img_rgb, cfg)
    if cfg.backend == "threshold":
        return _run_threshold_debug_backend(img_rgb)
    raise ValueError(f"Unknown segmentation backend: {cfg.backend}")


def _run_cellpose_sam(img_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "Cellpose is not installed. Install with `pip install -e .[cellpose]` "
            "or run with `--backend threshold` for a lightweight debug backend."
        ) from exc

    model = models.CellposeModel(gpu=cfg.use_gpu)
    out = model.eval(
        img_rgb,
        diameter=cfg.diameter,
        flow_threshold=cfg.flow_threshold,
        cellprob_threshold=cfg.cellprob_threshold,
    )
    masks = out[0]
    return masks.astype(np.int32)


def _run_threshold_debug_backend(img_rgb: np.ndarray) -> np.ndarray:
    """Small dependency-free backend for smoke tests only.

    It segments dark elongated structures on a bright background. It is not a
    replacement for Cellpose-SAM on real worm images.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    labels = measure.label(bw.astype(bool))
    keep = np.zeros_like(labels, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area >= 20:
            keep[labels == region.label] = True
    keep = morphology.opening(keep, morphology.disk(1))
    return measure.label(keep).astype(np.int32)


def paint_exclusion_as_background(img_rgb: np.ndarray, exclusion_mask: np.ndarray) -> np.ndarray:
    """Replace excluded pixels by the median non-excluded color before segmentation."""
    if exclusion_mask is None or not exclusion_mask.any():
        return img_rgb
    out = img_rgb.copy()
    valid = ~exclusion_mask
    if valid.any():
        fill = np.median(out[valid], axis=0).astype(np.uint8)
    else:
        fill = np.array([255, 255, 255], dtype=np.uint8)
    out[exclusion_mask] = fill
    return out


def remove_labels_overlapping_exclusion(
    mask: np.ndarray,
    exclusion_mask: np.ndarray,
    min_overlap_fraction: float = 0.01,
) -> np.ndarray:
    """Remove labels whose pixels overlap an exclusion mask."""
    if exclusion_mask is None or not exclusion_mask.any() or mask.max() == 0:
        return mask

    out = mask.copy()
    labels = np.unique(mask[exclusion_mask])
    labels = labels[labels > 0]
    for label in labels:
        label_pixels = mask == label
        overlap = np.count_nonzero(label_pixels & exclusion_mask) / max(np.count_nonzero(label_pixels), 1)
        if overlap >= min_overlap_fraction:
            out[label_pixels] = 0
    return out.astype(np.int32)


def remove_labels_outside_domain(
    mask: np.ndarray,
    valid_domain: np.ndarray,
    min_inside_fraction: float = 0.80,
) -> np.ndarray:
    """Remove labels that do not mostly lie inside the valid segmentation domain."""
    if valid_domain is None or valid_domain.all() or mask.max() == 0:
        return mask.astype(np.int32)

    out = mask.copy()
    for label in np.unique(mask):
        if label == 0:
            continue
        label_pixels = mask == label
        inside_fraction = np.count_nonzero(label_pixels & valid_domain) / max(np.count_nonzero(label_pixels), 1)
        if inside_fraction < min_inside_fraction:
            out[label_pixels] = 0
    return out.astype(np.int32)


def rescue_dark_worms(
    img_rgb: np.ndarray,
    existing_mask: np.ndarray,
    valid_domain: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Append dark elongated objects that Cellpose did not return at all.

    This is intentionally a rescue pass, not the primary segmentation method:
    it only adds candidates that overlap existing masks weakly. The final
    morphology filter is still applied later, so added candidates can still be
    rejected if they do not look worm-like.

    Returns:
        merged_mask: existing labels plus appended rescue labels.
        rescue_mask: label image containing only appended rescue candidates.
        stats: simple counts for metadata/debugging.
    """
    if not cfg.dark_worm_rescue:
        empty = np.zeros_like(existing_mask, dtype=np.int32)
        return existing_mask.astype(np.int32), empty, {"n_rescue_candidates": 0, "n_rescue_added": 0}

    H, W = img_rgb.shape[:2]
    if valid_domain is None:
        valid_domain = np.ones((H, W), dtype=bool)
    else:
        valid_domain = valid_domain.astype(bool)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Local dark-object score. The Gaussian background handles slow lighting
    # gradients, while black-hat emphasizes narrow dark worms and suppresses
    # broad grey regions.
    bg_px = _odd_at_least(cfg.dark_rescue_bg_blur_px, 3)
    background = cv2.GaussianBlur(gray, (bg_px, bg_px), 0).astype(np.int16)
    local_contrast = np.maximum(background - gray.astype(np.int16), 0).astype(np.uint8)

    blackhat_px = _odd_at_least(cfg.dark_rescue_blackhat_px, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blackhat_px, blackhat_px))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    score = np.maximum(local_contrast, blackhat)
    candidate = (score >= float(cfg.dark_rescue_min_contrast)) & valid_domain

    close_px = max(0, int(cfg.dark_rescue_close_px))
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    # Tiny dark specks are common in the tray and should not enter regionprops.
    candidate = _remove_small_components(candidate, min_area=max(1, int(cfg.dark_rescue_min_area_px)))

    existing_bool = existing_mask > 0
    if cfg.dark_rescue_existing_dilate_px > 0 and existing_bool.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(cfg.dark_rescue_existing_dilate_px) + 1, 2 * int(cfg.dark_rescue_existing_dilate_px) + 1),
        )
        existing_for_overlap = cv2.dilate(existing_bool.astype(np.uint8), k, iterations=1).astype(bool)
    else:
        existing_for_overlap = existing_bool

    cand_labels = measure.label(candidate)
    merged = existing_mask.copy().astype(np.int32)
    rescue_only = np.zeros_like(existing_mask, dtype=np.int32)
    next_label = int(merged.max()) + 1
    rescue_id = 1
    n_candidates = 0
    n_added = 0

    for region in measure.regionprops(cand_labels, intensity_image=score):
        n_candidates += 1
        region_pixels = cand_labels == region.label
        area = int(region.area)
        if area < int(cfg.dark_rescue_min_area_px):
            continue
        if area > int(cfg.dark_rescue_max_area_px):
            continue

        minr, minc, maxr, maxc = region.bbox
        sub = region_pixels[minr:maxr, minc:maxc]
        skel_len = measure_skeleton_length(sub)
        if skel_len < float(cfg.dark_rescue_min_skeleton_length_px):
            continue

        if region.axis_minor_length > 0:
            aspect = float(region.axis_major_length / region.axis_minor_length)
        else:
            aspect = float("inf")
        if aspect < float(cfg.dark_rescue_min_aspect_ratio):
            continue
        if region.axis_minor_length > float(cfg.dark_rescue_max_minor_axis_px):
            continue

        # Skip candidates that are probably just a second rendering of a mask
        # that Cellpose already found.
        overlap = np.count_nonzero(region_pixels & existing_for_overlap) / max(np.count_nonzero(region_pixels), 1)
        if overlap > float(cfg.dark_rescue_max_existing_overlap):
            continue

        # Low-contrast candidates are often dirt or tray texture. Requiring a
        # contrast margin keeps the rescue conservative.
        try:
            mean_intensity = region.intensity_mean
        except AttributeError:
            mean_intensity = region.mean_intensity
        if float(mean_intensity) < float(cfg.dark_rescue_min_contrast):
            continue

        merged[region_pixels] = next_label
        rescue_only[region_pixels] = rescue_id
        next_label += 1
        rescue_id += 1
        n_added += 1

    return merged.astype(np.int32), rescue_only.astype(np.int32), {
        "n_rescue_candidates": int(n_candidates),
        "n_rescue_added": int(n_added),
    }


def count_instance_labels(mask: np.ndarray) -> int:
    """Count non-zero instance labels robustly, even when labels have gaps."""
    if mask.size == 0:
        return 0
    labels = np.unique(mask)
    return int(np.count_nonzero(labels > 0))


def _odd_at_least(value: int | float, minimum: int) -> int:
    value_i = max(int(round(value)), int(minimum))
    if value_i % 2 == 0:
        value_i += 1
    return value_i


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels = measure.label(mask.astype(bool))
    if labels.max() == 0:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            out[labels == region.label] = True
    return out
