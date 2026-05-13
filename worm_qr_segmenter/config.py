from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """Central configuration for one batch run.

    All geometry values are in pixels. No physical calibration is applied.
    """

    # --- Preprocessing / working pixel scale ---
    # The raw image is resized first. All subsequent ROI coordinates, label
    # masks, and worm measurements are in this working pixel scale.
    scale_factor: float = 0.5

    # --- Segmentation backend ---
    # cellpose_sam is the intended production backend. threshold is only a
    # lightweight fallback for code testing without Cellpose installed.
    backend: Literal["cellpose_sam", "threshold"] = "cellpose_sam"
    # In working pixels. At the default half-scale this corresponds roughly
    # to the previous 30 px full-resolution diameter.
    diameter: float | None = 15.0
    flow_threshold: float = 0.6
    cellprob_threshold: float = -1.0
    use_gpu: bool = True

    # --- Dark-worm rescue pass ---
    # Cellpose-SAM can miss low-contrast curved worms completely. If a worm is
    # absent from the raw mask it will not appear in rejected-mask diagnostics.
    # This optional second pass finds dark elongated objects on the valid tray
    # background and appends non-overlapping candidates to the raw label mask.
    dark_worm_rescue: bool = True
    dark_rescue_bg_blur_px: int = 61
    dark_rescue_blackhat_px: int = 31
    dark_rescue_min_contrast: float = 10.0
    dark_rescue_min_area_px: int = 18
    dark_rescue_max_area_px: int = 5000
    dark_rescue_min_skeleton_length_px: float = 14.0
    dark_rescue_min_aspect_ratio: float = 1.35
    dark_rescue_max_minor_axis_px: float = 22.0
    dark_rescue_existing_dilate_px: int = 2
    dark_rescue_max_existing_overlap: float = 0.20
    dark_rescue_close_px: int = 1

    # --- Tray ROI detection ---
    tray_inset_px: int = 15
    manual_roi: tuple[int, int, int, int] | None = None  # y0, y1, x0, x1

    # --- Valid segmentation domain ---
    # Large dark/grey regions are not worm habitat in this imaging layout.
    # They are replaced by background before segmentation and labels outside
    # the bright tray region are removed after segmentation. Small dark worms
    # remain detectable because the domain is computed from a blurred image.
    exclude_dark_regions_from_segmentation: bool = True
    min_valid_background_luma: int = 125
    valid_region_blur_px: int = 41
    valid_region_close_px: int = 25
    valid_region_erode_px: int = 2
    valid_region_keep_largest: bool = True
    valid_region_min_area_fraction: float = 0.02
    valid_region_min_inside_fraction: float = 0.80

    # --- QR detection/exclusion ---
    # Try candidate crops, contrast-normalized crops and stronger upscales after
    # the cheap whole-image OpenCV QR pass. Keep this enabled for production.
    qr_try_harder: bool = True
    # Also try decoding on the already-rescaled working image, before any ROI
    # cropping. This avoids a failure mode where only later cropped/debug images
    # are inspected while the QR code was never retried after the scale step.
    qr_decode_on_working_image: bool = True
    qr_pad_px: int = 10
    exclude_qr_from_segmentation: bool = True
    copy_originals_with_metadata_name: bool = False

    # --- Worm morphology filters ---
    min_area_px: int = 40
    max_area_px: int = 20_000
    min_skeleton_length_px: float = 15
    min_aspect_ratio: float = 1.5
    max_solidity: float = 0.98
    min_eccentricity: float = 0.7

    # --- Output control ---
    debug: bool = False
    metadata_only: bool = False

    # --- File handling ---
    image_extensions: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
    )


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    labels: Path
    overlays: Path
    stats: Path
    metadata: Path
    debug: Path
    originals_named: Path

    @classmethod
    def create(cls, root: Path, debug: bool = False, copy_originals: bool = False) -> "OutputLayout":
        layout = cls(
            root=root,
            labels=root / "labels",
            overlays=root / "overlays",
            stats=root / "stats",
            metadata=root / "metadata",
            debug=root / "debug",
            originals_named=root / "originals_named",
        )
        layout.root.mkdir(parents=True, exist_ok=True)
        layout.labels.mkdir(parents=True, exist_ok=True)
        layout.overlays.mkdir(parents=True, exist_ok=True)
        layout.stats.mkdir(parents=True, exist_ok=True)
        layout.metadata.mkdir(parents=True, exist_ok=True)
        if debug:
            layout.debug.mkdir(parents=True, exist_ok=True)
        if copy_originals:
            layout.originals_named.mkdir(parents=True, exist_ok=True)
        return layout
