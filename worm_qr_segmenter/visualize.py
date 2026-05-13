from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib import cm
from skimage import measure


def save_label_mask(mask: np.ndarray, path: Path) -> None:
    """Save a label mask as integer TIFF.

    Pixel value 0 is background. Pixel values 1..N correspond to worm_id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(mask.max())
    if max_label <= np.iinfo(np.uint16).max:
        tifffile.imwrite(path, mask.astype(np.uint16), photometric="minisblack")
    else:
        tifffile.imwrite(path, mask.astype(np.uint32), photometric="minisblack")


def save_labeled_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Save the main human-readable output image with colored labels and IDs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 16), dpi=120)
    ax.imshow(img_rgb)

    if mask.max() > 0:
        n = int(mask.max())
        rng = np.random.default_rng(42)
        colors = cm.tab20(rng.random(n))
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        for i in range(1, n + 1):
            overlay[mask == i] = colors[i - 1]
        overlay[..., 3] *= 0.45
        ax.imshow(overlay)

        if not df.empty and "worm_id" in df.columns:
            for _, row in df.iterrows():
                ax.text(
                    row["centroid_x_crop"],
                    row["centroid_y_crop"],
                    str(int(row["worm_id"])),
                    color="white",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
                )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_instance_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    title: str,
    out_path: Path,
    alpha: float = 0.45,
    label_ids: bool = True,
) -> None:
    """Save a generic colored instance-mask overlay. Debug-only helper."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 16), dpi=120)
    ax.imshow(img_rgb)

    if mask.max() > 0:
        labels = [int(x) for x in np.unique(mask) if int(x) > 0]
        rng = np.random.default_rng(123)
        colors = cm.tab20(rng.random(len(labels)))
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        for color_idx, label in enumerate(labels):
            overlay[mask == label] = colors[color_idx]
        overlay[..., 3] *= float(alpha)
        ax.imshow(overlay)

        if label_ids:
            for region in measure.regionprops(mask.astype(int)):
                ax.text(
                    region.centroid[1],
                    region.centroid[0],
                    str(int(region.label)),
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
                )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_rejected_overlay(
    img_rgb: np.ndarray,
    raw_mask: np.ndarray,
    rejected_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save rejected objects for threshold tuning. Debug-only output."""
    if rejected_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 16), dpi=120)
    ax.imshow(img_rgb)

    overlay = np.zeros((*raw_mask.shape, 4), dtype=float)
    for _, row in rejected_df.iterrows():
        overlay[raw_mask == int(row["raw_label"])] = [1, 0, 0, 0.5]
    ax.imshow(overlay)

    for _, row in rejected_df.iterrows():
        ax.text(
            row["centroid_x_crop"],
            row["centroid_y_crop"],
            row["reject_reason"],
            color="yellow",
            fontsize=5,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
        )
    ax.set_title(f"Rejected masks ({len(rejected_df)})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_histograms(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Save summary histograms. Debug-only output."""
    if df.empty:
        return

    metrics = [
        ("skeleton_length_px", "Skeleton length (px)", 30),
        ("axis_major_px", "Major axis length (px)", 30),
        ("area_px", "Area (px)", 30),
        ("aspect_ratio", "Aspect ratio", 30),
        ("eccentricity", "Eccentricity", 20),
        ("solidity", "Solidity", 20),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=110)
    for ax, (col, label, bins) in zip(axes.ravel(), metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].dropna()
        ax.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        if len(vals):
            med = float(vals.median())
            mean = float(vals.mean())
            ax.axvline(med, linestyle="--", linewidth=1.2, label=f"median={med:.1f}")
            ax.axvline(mean, linestyle=":", linewidth=1.2, label=f"mean={mean:.1f}")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{title}  (n={len(df)})", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_qr_debug(img_rgb: np.ndarray, points: list[list[float]] | None, out_path: Path) -> None:
    if not points:
        return
    vis = img_rgb.copy()
    pts = np.asarray(points, dtype=np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def save_valid_region_debug(img_rgb: np.ndarray, valid_region: np.ndarray, out_path: Path) -> None:
    """Save a debug overlay of the valid segmentation domain."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = img_rgb.copy()
    invalid = ~valid_region
    # Darken invalid pixels but keep the original image recognizable.
    vis[invalid] = (0.35 * vis[invalid]).astype(np.uint8)
    boundary = cv2.Canny(valid_region.astype(np.uint8) * 255, 50, 150)
    vis[boundary > 0] = np.array([0, 255, 0], dtype=np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def save_qr_search_debug(
    img_rgb: np.ndarray,
    candidate_boxes: list[tuple[int, int, int, int]],
    detected_points: list[list[float]] | None,
    out_path: Path,
) -> None:
    """Save a debug image showing QR search candidates and the detected QR polygon."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = img_rgb.copy()
    for i, (x0, y0, x1, y1) in enumerate(candidate_boxes, start=1):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 170, 0), 2)
        cv2.putText(
            vis,
            str(i),
            (int(x0), max(0, int(y0) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 170, 0),
            2,
            cv2.LINE_AA,
        )
    if detected_points:
        pts = np.asarray(detected_points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
