from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .io_utils import collect_images
from .pipeline import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worm-qr-segment",
        description="Batch worm segmentation with QR-code metadata extraction.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image to process")
    source.add_argument("--image_dir", type=Path, help="Directory of images to process")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--recursive", action="store_true", help="Process image_dir recursively")
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.5,
        help="Resize factor applied before ROI detection and segmentation. All measurements are in this working pixel scale. Default: 0.5.",
    )

    parser.add_argument(
        "--backend",
        choices=["cellpose_sam", "threshold"],
        default="cellpose_sam",
        help="Segmentation backend. threshold is only for smoke tests/debugging.",
    )
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU for Cellpose-SAM")
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Cellpose object width in working pixels. Use 0 for Cellpose auto-estimation. Default: 15 at scale_factor 0.5.",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=None,
        help="Cellpose-SAM flow-error threshold. Lower is more permissive for irregular masks; higher is stricter. Default: 0.6.",
    )
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=None,
        help="Cellpose-SAM mask-probability threshold. Lower values recover fainter worms; higher values are stricter. Default: -1.0.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Manual tray ROI in working-image pixels after scale_factor has been applied. Overrides ROI auto-detection.",
    )

    parser.add_argument("--debug", action="store_true", help="Save debug masks, rejected overlays, ROI/QR diagnostics and histograms")
    parser.add_argument("--metadata_only", action="store_true", help="Only decode QR codes and write image metadata; do not segment")
    parser.add_argument("--copy_originals", action="store_true", help="Copy original images to originals_named/ with QR metadata in the filename")
    parser.add_argument("--no_qr_try_harder", action="store_true", help="Disable robust QR candidate-crop/preprocessing search and use only the fast whole-image QR pass")
    parser.add_argument("--no_qr_working_image_search", action="store_true", help="Disable the second QR pass on the complete rescaled working image")
    parser.add_argument("--no_dark_region_exclusion", action="store_true", help="Disable exclusion of large dark/grey non-tray regions before segmentation")
    parser.add_argument("--min_valid_background_luma", type=int, default=None, help="Minimum blurred luminance considered valid tray background. Default: 125")
    parser.add_argument("--valid_region_blur_px", type=int, default=None, help="Blur radius/kernel scale for valid-region detection in working pixels. Default: 41")
    parser.add_argument("--valid_region_close_px", type=int, default=None, help="Closing radius for valid-region detection in working pixels. Default: 25")
    parser.add_argument("--valid_region_erode_px", type=int, default=None, help="Erosion radius for valid-region detection in working pixels. Default: 2")
    parser.add_argument("--valid_region_min_inside_fraction", type=float, default=None, help="Minimum fraction of a mask that must lie in the valid tray region. Default: 0.80")

    # Dark-worm rescue pass. This adds dark elongated candidates that the primary
    # backend missed completely, then sends them through the normal morphology
    # filter. Values are in working pixels.
    parser.add_argument(
        "--no_dark_worm_rescue",
        "--no_dark_rescue",
        action="store_true",
        help="Disable the dark elongated-object rescue pass. --no_dark_rescue is a short alias.",
    )
    parser.add_argument("--dark_rescue_bg_blur_px", type=int, default=None, help="Background blur kernel for rescue contrast image. Default: 61")
    parser.add_argument("--dark_rescue_blackhat_px", type=int, default=None, help="Black-hat kernel for dark worm enhancement. Default: 31")
    parser.add_argument("--dark_rescue_min_contrast", type=float, default=None, help="Minimum grey-level dark contrast for rescue candidates. Default: 10")
    parser.add_argument("--dark_rescue_min_area_px", type=int, default=None, help="Minimum rescue candidate area. Default: 18")
    parser.add_argument("--dark_rescue_max_area_px", type=int, default=None, help="Maximum rescue candidate area. Default: 5000")
    parser.add_argument("--dark_rescue_min_skeleton_length_px", type=float, default=None, help="Minimum rescue skeleton length. Default: 14")
    parser.add_argument("--dark_rescue_min_aspect_ratio", type=float, default=None, help="Minimum rescue candidate aspect ratio. Default: 1.35")
    parser.add_argument("--dark_rescue_max_minor_axis_px", type=float, default=None, help="Maximum rescue candidate minor axis. Default: 22")
    parser.add_argument("--dark_rescue_max_existing_overlap", type=float, default=None, help="Maximum overlap with existing primary masks before rescue candidate is considered duplicate. Default: 0.20")

    # Morphology filter overrides
    parser.add_argument("--min_area_px", type=int, default=None)
    parser.add_argument("--max_area_px", type=int, default=None)
    parser.add_argument("--min_skeleton_length_px", type=float, default=None)
    parser.add_argument("--min_aspect_ratio", type=float, default=None)
    parser.add_argument("--max_solidity", type=float, default=None)
    parser.add_argument("--min_eccentricity", type=float, default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.scale_factor = args.scale_factor
    cfg.backend = args.backend
    cfg.use_gpu = not args.no_gpu
    cfg.debug = bool(args.debug)
    cfg.metadata_only = bool(args.metadata_only)
    cfg.copy_originals_with_metadata_name = bool(args.copy_originals)
    cfg.qr_try_harder = not bool(args.no_qr_try_harder)
    cfg.qr_decode_on_working_image = not bool(args.no_qr_working_image_search)
    cfg.exclude_dark_regions_from_segmentation = not bool(args.no_dark_region_exclusion)
    cfg.dark_worm_rescue = not bool(args.no_dark_worm_rescue)

    for name in (
        "min_valid_background_luma",
        "valid_region_blur_px",
        "valid_region_close_px",
        "valid_region_erode_px",
        "valid_region_min_inside_fraction",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    for name in (
        "dark_rescue_bg_blur_px",
        "dark_rescue_blackhat_px",
        "dark_rescue_min_contrast",
        "dark_rescue_min_area_px",
        "dark_rescue_max_area_px",
        "dark_rescue_min_skeleton_length_px",
        "dark_rescue_min_aspect_ratio",
        "dark_rescue_max_minor_axis_px",
        "dark_rescue_max_existing_overlap",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    if args.diameter is not None:
        cfg.diameter = None if args.diameter == 0 else args.diameter
    if args.flow_threshold is not None:
        cfg.flow_threshold = args.flow_threshold
    if args.cellprob_threshold is not None:
        cfg.cellprob_threshold = args.cellprob_threshold
    if args.roi is not None:
        cfg.manual_roi = tuple(args.roi)

    for name in (
        "min_area_px",
        "max_area_px",
        "min_skeleton_length_px",
        "min_aspect_ratio",
        "max_solidity",
        "min_eccentricity",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    input_path = args.image if args.image is not None else args.image_dir
    images = collect_images(input_path, cfg, recursive=args.recursive)
    if not images:
        raise SystemExit("No image files found.")

    run_batch(images, args.out, cfg)


if __name__ == "__main__":
    main()
