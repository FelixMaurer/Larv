from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure, morphology

from .config import Config


def measure_skeleton_length(mask_bool: np.ndarray) -> float:
    """Approximate skeleton length in pixels.

    This is a simple pixel count on the one-pixel-wide skeleton. It is stable
    enough for relative morphology statistics but not a subpixel contour length.
    """
    skel = morphology.skeletonize(mask_bool)
    return float(skel.sum())


def filter_and_measure(
    mask: np.ndarray,
    cfg: Config,
    crop_origin_yx: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Filter raw instance labels and measure kept/rejected objects.

    The returned kept mask is relabeled 1..N, and `worm_id` in the kept table
    corresponds exactly to the pixel values in the kept label mask.
    """
    origin_y, origin_x = crop_origin_yx
    props = measure.regionprops(mask)

    kept_records: list[dict] = []
    rejected_records: list[dict] = []
    keep_labels: list[int] = []

    for p in props:
        minr, minc, maxr, maxc = p.bbox
        sub = mask[minr:maxr, minc:maxc] == p.label
        skel_len = measure_skeleton_length(sub)
        aspect = float(p.axis_major_length / p.axis_minor_length) if p.axis_minor_length > 0 else float("inf")

        record = {
            "raw_label": int(p.label),
            "area_px": int(p.area),
            "skeleton_length_px": skel_len,
            "axis_major_px": float(p.axis_major_length),
            "axis_minor_px": float(p.axis_minor_length),
            "aspect_ratio": aspect,
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "perimeter_px": float(p.perimeter),
            "orientation_rad": float(p.orientation),
            "equivalent_diameter_area_px": float(p.equivalent_diameter_area),
            "centroid_y_crop": float(p.centroid[0]),
            "centroid_x_crop": float(p.centroid[1]),
            "centroid_y_image": float(p.centroid[0] + origin_y),
            "centroid_x_image": float(p.centroid[1] + origin_x),
            "bbox_y0_crop": int(minr),
            "bbox_x0_crop": int(minc),
            "bbox_y1_crop": int(maxr),
            "bbox_x1_crop": int(maxc),
            "bbox_y0_image": int(minr + origin_y),
            "bbox_x0_image": int(minc + origin_x),
            "bbox_y1_image": int(maxr + origin_y),
            "bbox_x1_image": int(maxc + origin_x),
        }

        reasons: list[str] = []
        if p.area < cfg.min_area_px:
            reasons.append(f"area<{cfg.min_area_px}")
        if p.area > cfg.max_area_px:
            reasons.append(f"area>{cfg.max_area_px}")
        if skel_len < cfg.min_skeleton_length_px:
            reasons.append(f"skeleton_length<{cfg.min_skeleton_length_px}")
        if aspect < cfg.min_aspect_ratio:
            reasons.append(f"aspect_ratio<{cfg.min_aspect_ratio}")
        if p.solidity > cfg.max_solidity:
            reasons.append(f"solidity>{cfg.max_solidity}")
        if p.eccentricity < cfg.min_eccentricity:
            reasons.append(f"eccentricity<{cfg.min_eccentricity}")

        if reasons:
            record["reject_reason"] = ";".join(reasons)
            rejected_records.append(record)
        else:
            keep_labels.append(int(p.label))
            kept_records.append(record)

    kept_mask = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_label in enumerate(keep_labels, start=1):
        kept_mask[mask == old_label] = new_id

    for new_id, rec in enumerate(kept_records, start=1):
        rec["worm_id"] = new_id

    kept_df = pd.DataFrame(kept_records)
    rejected_df = pd.DataFrame(rejected_records)

    if not kept_df.empty:
        cols = ["worm_id"] + [c for c in kept_df.columns if c != "worm_id"]
        kept_df = kept_df[cols]

    return kept_mask, kept_df, rejected_df
