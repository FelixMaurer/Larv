from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .config import Config, OutputLayout
from .io_utils import copy_original_with_metadata_name, output_basename
from .measure import filter_and_measure
from .preprocess import make_valid_segmentation_region, resize_rgb, scale_qr_geometry
from .qr import QRResult, decode_qr, find_qr_candidates, qr_exclusion_mask
from .roi import detect_tray_roi
from .segment import (
    count_instance_labels,
    paint_exclusion_as_background,
    remove_labels_outside_domain,
    remove_labels_overlapping_exclusion,
    rescue_dark_worms,
    run_segmentation,
)
from .visualize import (
    save_histograms,
    save_instance_overlay,
    save_label_mask,
    save_labeled_overlay,
    save_qr_debug,
    save_qr_search_debug,
    save_rejected_overlay,
    save_valid_region_debug,
)


def process_image(image_path: Path, layout: OutputLayout, cfg: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    """Process one image and write all per-image outputs."""
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Work from the raw image only. The first processing step is loading; the
    # second is the global working-scale resize. QR decoding is attempted before
    # any tray ROI crop, first on the original full-resolution image and then,
    # if necessary, on the complete working-scale image. The working-image pass
    # is important for these photos because downscaling can remove camera noise
    # while keeping the QR code intact.
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb_original.shape[:2]
    img_rgb = resize_rgb(img_rgb_original, cfg.scale_factor)
    H, W = img_rgb.shape[:2]

    qr_original_fast = decode_qr(img_rgb_original, try_harder=cfg.qr_try_harder)
    qr_working_direct = QRResult(detected=False)
    if (not qr_original_fast.detected) and cfg.qr_decode_on_working_image:
        qr_working_direct = decode_qr(img_rgb, try_harder=cfg.qr_try_harder)

    if qr_original_fast.detected:
        qr_original = qr_original_fast
        qr_working = scale_qr_geometry(qr_original, cfg.scale_factor)
    elif qr_working_direct.detected:
        # Store the selected QR geometry in original-image coordinates for
        # metadata/filenames, and keep the direct working-coordinate result for
        # QR exclusion during segmentation.
        qr_original = scale_qr_geometry(qr_working_direct, 1.0 / cfg.scale_factor)
        qr_original.method = f"working_image_mapped_to_original:{qr_working_direct.method}"
        qr_working = qr_working_direct
    else:
        qr_original = qr_original_fast
        qr_working = QRResult(detected=False)

    basename = output_basename(image_path, qr_original)

    if cfg.debug:
        candidate_boxes_original = [c.bbox_xyxy for c in find_qr_candidates(img_rgb_original)]
        save_qr_search_debug(
            img_rgb_original,
            candidate_boxes_original,
            qr_original.points,
            layout.debug / f"{basename}__qr_search_debug_original.png",
        )
        candidate_boxes_working = [c.bbox_xyxy for c in find_qr_candidates(img_rgb)]
        save_qr_search_debug(
            img_rgb,
            candidate_boxes_working,
            qr_working.points,
            layout.debug / f"{basename}__qr_search_debug_working.png",
        )

    metadata: dict[str, Any] = {
        "input_path": str(image_path),
        "original_filename": image_path.name,
        "output_basename": basename,
        "scale_factor": float(cfg.scale_factor),
        "original_image_height_px": H0,
        "original_image_width_px": W0,
        "working_image_height_px": H,
        "working_image_width_px": W,
        **_flatten_qr_record(qr_original),
    }
    if qr_working.bbox_xyxy:
        metadata["qr_bbox_xyxy_working"] = json.dumps(qr_working.bbox_xyxy, ensure_ascii=False)
    if qr_working.points:
        metadata["qr_points_working"] = json.dumps(qr_working.points, ensure_ascii=False)

    qr_text_path, qr_json_path = _write_per_image_qr_files(layout, basename, qr_original)
    metadata["qr_text_file"] = str(qr_text_path)
    metadata["qr_json_file"] = str(qr_json_path)

    if cfg.copy_originals_with_metadata_name:
        copied = copy_original_with_metadata_name(image_path, layout.originals_named, basename)
        metadata["metadata_named_original_copy"] = str(copied)

    if cfg.debug and qr_working.detected:
        save_qr_debug(img_rgb, qr_working.points, layout.debug / f"{basename}__qr_debug_working.png")

    if cfg.manual_roi is not None:
        y0, y1, x0, x1 = cfg.manual_roi
        roi_source = "manual"
    else:
        debug_path = layout.debug / f"{basename}__roi_debug.png" if cfg.debug else None
        y0, y1, x0, x1 = detect_tray_roi(img_rgb, inset=cfg.tray_inset_px, debug_path=debug_path)
        roi_source = "auto"

    metadata.update(
        {
            "roi_source": roi_source,
            "roi_y0": int(y0),
            "roi_y1": int(y1),
            "roi_x0": int(x0),
            "roi_x1": int(x1),
            "crop_height_px": int(y1 - y0),
            "crop_width_px": int(x1 - x0),
            "coordinate_scale": "working_pixels",
        }
    )

    crop = img_rgb[y0:y1, x0:x1]

    if cfg.metadata_only:
        metadata.update(
            {
                "n_raw_masks": 0,
                "n_kept_worms": 0,
                "n_rejected_masks": 0,
                "labels_tif": "",
                "overlay_png": "",
                "stats_csv": "",
            }
        )
        return metadata, pd.DataFrame()

    qr_exclusion_crop = np.zeros(crop.shape[:2], dtype=bool)
    if cfg.exclude_qr_from_segmentation and qr_working.detected:
        full_qr_exclusion = qr_exclusion_mask((H, W), qr_working, pad_px=cfg.qr_pad_px)
        qr_exclusion_crop = full_qr_exclusion[y0:y1, x0:x1]

    valid_region_crop = make_valid_segmentation_region(crop, cfg)
    combined_exclusion_crop = (~valid_region_crop) | qr_exclusion_crop

    if cfg.debug:
        save_valid_region_debug(
            crop,
            valid_region_crop,
            layout.debug / f"{basename}__valid_region_debug.png",
        )

    crop_for_seg = paint_exclusion_as_background(crop, combined_exclusion_crop)
    raw_mask_initial = run_segmentation(crop_for_seg, cfg)
    n_raw_initial = count_instance_labels(raw_mask_initial)

    raw_mask_after_exclusion = remove_labels_overlapping_exclusion(raw_mask_initial, qr_exclusion_crop)
    raw_mask_after_exclusion = remove_labels_outside_domain(
        raw_mask_after_exclusion,
        valid_region_crop,
        min_inside_fraction=cfg.valid_region_min_inside_fraction,
    )
    n_raw_after_exclusion = count_instance_labels(raw_mask_after_exclusion)

    raw_mask, rescue_mask, rescue_stats = rescue_dark_worms(
        crop,
        raw_mask_after_exclusion,
        valid_region_crop & (~qr_exclusion_crop),
        cfg,
    )
    n_raw = count_instance_labels(raw_mask)

    kept_mask, kept_df, rejected_df = filter_and_measure(raw_mask, cfg, crop_origin_yx=(y0, x0))

    # Attach image- and QR-level metadata to every worm row. The statistics CSV
    # stays raw and flat: one row per worm, no aggregate summaries hidden inside.
    insert_cols = {
        "original_filename": image_path.name,
        "output_basename": basename,
        "scale_factor": float(cfg.scale_factor),
        "coordinate_scale": "working_pixels",
        "qr_detected": bool(qr_original.detected),
        "qr_text": qr_original.text,
    }
    if qr_original.parsed:
        for key, value in qr_original.parsed.items():
            insert_cols[f"qr_{key}"] = value

    if not kept_df.empty:
        for col, value in reversed(list(insert_cols.items())):
            kept_df.insert(0, col, value)
    else:
        measurement_cols = [
            "worm_id", "raw_label", "area_px", "skeleton_length_px",
            "axis_major_px", "axis_minor_px", "aspect_ratio", "eccentricity",
            "solidity", "perimeter_px", "orientation_rad",
            "equivalent_diameter_area_px", "centroid_y_crop",
            "centroid_x_crop", "centroid_y_image", "centroid_x_image",
            "bbox_y0_crop", "bbox_x0_crop", "bbox_y1_crop", "bbox_x1_crop",
            "bbox_y0_image", "bbox_x0_image", "bbox_y1_image", "bbox_x1_image",
        ]
        kept_df = pd.DataFrame(columns=list(insert_cols.keys()) + measurement_cols)

    labels_path = layout.labels / f"{basename}__labels.tif"
    overlay_path = layout.overlays / f"{basename}__labeled.png"
    stats_path = layout.stats / f"{basename}__worms.csv"

    save_label_mask(kept_mask, labels_path)
    kept_df.to_csv(stats_path, index=False)
    save_labeled_overlay(crop, kept_mask, kept_df, f"{basename}: {len(kept_df)} worms", overlay_path)

    if cfg.debug:
        raw_initial_path = layout.debug / f"{basename}__raw_mask_initial.npy"
        np.save(raw_initial_path, raw_mask_initial)
        save_instance_overlay(
            crop,
            raw_mask_initial,
            f"{basename}: raw Cellpose/primary masks before exclusion",
            layout.debug / f"{basename}__raw_mask_initial_overlay.png",
        )

        raw_after_exclusion_path = layout.debug / f"{basename}__raw_mask_after_exclusion.npy"
        np.save(raw_after_exclusion_path, raw_mask_after_exclusion)
        save_instance_overlay(
            crop,
            raw_mask_after_exclusion,
            f"{basename}: raw masks after QR/domain exclusion",
            layout.debug / f"{basename}__raw_mask_after_exclusion_overlay.png",
        )

        rescue_path = layout.debug / f"{basename}__dark_rescue_mask.npy"
        np.save(rescue_path, rescue_mask)
        save_instance_overlay(
            crop,
            rescue_mask,
            f"{basename}: added dark-worm rescue candidates",
            layout.debug / f"{basename}__dark_rescue_overlay.png",
        )

        raw_final_path = layout.debug / f"{basename}__raw_mask_final.npy"
        np.save(raw_final_path, raw_mask)
        save_instance_overlay(
            crop,
            raw_mask,
            f"{basename}: final raw masks before morphology filtering",
            layout.debug / f"{basename}__raw_mask_final_overlay.png",
        )

        rejected_path = layout.debug / f"{basename}__rejected.csv"
        rejected_df.to_csv(rejected_path, index=False)
        save_rejected_overlay(crop, raw_mask, rejected_df, layout.debug / f"{basename}__rejected_overlay.png")
        save_histograms(kept_df, f"{basename}: geometry distributions", layout.debug / f"{basename}__histograms.png")

    metadata.update(
        {
            "n_raw_masks_initial": n_raw_initial,
            "n_raw_masks_after_exclusion": n_raw_after_exclusion,
            **rescue_stats,
            "n_raw_masks": n_raw,
            "valid_region_fraction": float(np.mean(valid_region_crop)),
            "n_kept_worms": int(len(kept_df)),
            "n_rejected_masks": int(len(rejected_df)),
            "labels_tif": str(labels_path),
            "overlay_png": str(overlay_path),
            "stats_csv": str(stats_path),
        }
    )
    return metadata, kept_df


def run_batch(images: list[Path], out_dir: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full batch and write global metadata/statistics files."""
    layout = OutputLayout.create(
        out_dir,
        debug=cfg.debug,
        copy_originals=cfg.copy_originals_with_metadata_name,
    )

    metadata_records: list[dict[str, Any]] = []
    worm_tables: list[pd.DataFrame] = []

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}")
        metadata, worms = process_image(image_path, layout, cfg)
        metadata_records.append(metadata)
        if not worms.empty or len(worms.columns) > 0:
            worm_tables.append(worms)
        qr_text = metadata.get("qr_text") or ""
        qr_short = qr_text if len(qr_text) <= 60 else qr_text[:57] + "..."
        print(
            f"    QR: {'yes' if metadata.get('qr_detected') else 'no'}"
            f"{(' | ' + qr_short) if qr_short else ''} | "
            f"worms: {metadata.get('n_kept_worms', 0)} | "
            f"raw: {metadata.get('n_raw_masks', 0)} | "
            f"rescue: {metadata.get('n_rescue_added', 0)} | "
            f"rejected: {metadata.get('n_rejected_masks', 0)}"
        )

    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv = layout.metadata / "images_metadata.csv"
    metadata_jsonl = layout.metadata / "images_metadata.jsonl"
    metadata_df.to_csv(metadata_csv, index=False)
    with open(metadata_jsonl, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if worm_tables:
        all_worms_df = pd.concat(worm_tables, ignore_index=True)
    else:
        all_worms_df = pd.DataFrame()
    all_worms_csv = layout.stats / "all_worms.csv"
    all_worms_df.to_csv(all_worms_csv, index=False)

    with open(layout.metadata / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    if cfg.debug and not all_worms_df.empty:
        save_histograms(all_worms_df, "All images combined", layout.debug / "all_histograms.png")

    print(f"\nWrote metadata: {metadata_csv}")
    print(f"Wrote all worm rows: {all_worms_csv}")
    return metadata_df, all_worms_df


def _write_per_image_qr_files(layout: OutputLayout, basename: str, qr: QRResult) -> tuple[Path, Path]:
    """Write one small text file and one JSON file containing the decoded QR metadata."""
    text_path = layout.metadata / f"{basename}__qr.txt"
    json_path = layout.metadata / f"{basename}__qr.json"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text((qr.text or "") + "\n", encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(qr), f, indent=2, ensure_ascii=False)
    return text_path, json_path


def _flatten_qr_record(qr: QRResult) -> dict[str, Any]:
    record = qr.to_record()
    # Use concise top-level names in images_metadata.csv.
    record["qr_detected"] = record.pop("detected")
    record["qr_text"] = record.pop("text")
    record["qr_method"] = record.pop("method")
    record["qr_points"] = json.dumps(record.pop("points"), ensure_ascii=False)
    record["qr_bbox_xyxy"] = json.dumps(record.pop("bbox_xyxy"), ensure_ascii=False)
    return record
