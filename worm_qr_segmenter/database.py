from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DB_NAME = "worm_database"


WORM_ID_LIKE_COLUMNS = {
    "worm_id",
    "raw_label",
    "label",
}

WORM_QR_AND_IMAGE_COLUMNS = {
    "original_filename",
    "output_basename",
    "scale_factor",
    "coordinate_scale",
    "qr_detected",
    "qr_text",
    "qr_raw",
    "qr_format",
    "qr_plot",
    "qr_spalte",
    "qr_reihe",
    "qr_condition",
    "qr_sample_id",
}


class DatabaseBuildError(RuntimeError):
    pass


def read_segmentation_tables(segmentation_out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read image metadata and worm rows from a segmentation output folder.

    Expected layout is produced by ``worm-qr-segment``:

    - metadata/images_metadata.csv
    - stats/all_worms.csv, or per-image ``stats/*__worms.csv`` files
    """
    segmentation_out = Path(segmentation_out)
    metadata_path = segmentation_out / "metadata" / "images_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    metadata_df = _normalize_metadata(metadata_df)

    all_worms_path = segmentation_out / "stats" / "all_worms.csv"
    if all_worms_path.exists():
        worms_df = pd.read_csv(all_worms_path)
    else:
        worm_files = sorted((segmentation_out / "stats").glob("*__worms.csv"))
        if not worm_files:
            worms_df = pd.DataFrame()
        else:
            tables = []
            for path in worm_files:
                try:
                    df = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                if not df.empty or len(df.columns) > 0:
                    tables.append(df)
            worms_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    worms_df = _normalize_worms(worms_df, metadata_df)
    return metadata_df, worms_df


def build_database(segmentation_out: Path, db_dir: Path, allow_csv_only: bool = False) -> dict[str, str]:
    """Build a small analysis database from segmentation outputs.

    The database is intentionally simple and Git-friendly. It contains three
    parquet tables plus CSV mirrors for inspection and diffing:

    - images: one row per processed image, including QR metadata and output paths
    - worms: one row per worm, including image/QR metadata and raw measurements
    - image_summary: one row per image with count and aggregate worm metrics
    """
    segmentation_out = Path(segmentation_out)
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    images_df, worms_df = read_segmentation_tables(segmentation_out)
    image_summary_df = make_image_summary(images_df, worms_df)

    outputs: dict[str, str] = {}
    for name, df in {
        "images": images_df,
        "worms": worms_df,
        "image_summary": image_summary_df,
    }.items():
        outputs[f"{name}_csv"] = str(db_dir / f"{name}.csv")
        df.to_csv(db_dir / f"{name}.csv", index=False)
        try:
            outputs[f"{name}_parquet"] = str(db_dir / f"{name}.parquet")
            df.to_parquet(db_dir / f"{name}.parquet", index=False)
        except ImportError as exc:
            if not allow_csv_only:
                raise DatabaseBuildError(
                    "Writing parquet requires pyarrow or fastparquet. Install the app extra with:\n"
                    "    pip install -e .[app]\n"
                    "or install pyarrow directly with:\n"
                    "    pip install pyarrow\n"
                    "Use --allow_csv_only only for temporary debugging."
                ) from exc
            outputs.pop(f"{name}_parquet", None)

    manifest = {
        "segmentation_out": str(segmentation_out.resolve()),
        "db_dir": str(db_dir.resolve()),
        "tables": outputs,
        "n_images": int(len(images_df)),
        "n_worms": int(len(worms_df)),
        "n_image_summary_rows": int(len(image_summary_df)),
        "coordinate_scale": _single_value_or_empty(images_df, "coordinate_scale"),
        "scale_factor_values": _unique_jsonable(images_df.get("scale_factor", pd.Series(dtype=float))),
    }
    manifest_path = db_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return outputs


def make_image_summary(images_df: pd.DataFrame, worms_df: pd.DataFrame) -> pd.DataFrame:
    """Create one image-level summary table with counts and aggregate metrics."""
    if images_df.empty:
        return pd.DataFrame()

    summary = images_df.copy()
    if "output_basename" not in summary.columns:
        summary["output_basename"] = summary.get("original_filename", pd.Series(range(len(summary)))).astype(str)

    # Start from metadata count if available. It is the authoritative pipeline
    # output and also keeps zero-worm images in the summary.
    if "n_kept_worms" in summary.columns:
        summary["count"] = pd.to_numeric(summary["n_kept_worms"], errors="coerce").fillna(0).astype(int)
    else:
        summary["count"] = 0

    if worms_df.empty or "output_basename" not in worms_df.columns:
        return _order_grid_columns(summary)

    numeric_cols = _numeric_worm_measurement_columns(worms_df)
    grouped = worms_df.groupby("output_basename", dropna=False)

    count_df = grouped.size().rename("count_from_worm_rows").reset_index()
    aggregate_tables = [count_df]

    if numeric_cols:
        agg = grouped[numeric_cols].agg(["mean", "median", "std", "min", "max", "sum"])
        agg.columns = [f"{stat}_{col}" for col, stat in agg.columns]
        agg = agg.reset_index()
        aggregate_tables.append(agg)

    aggregates = aggregate_tables[0]
    for table in aggregate_tables[1:]:
        aggregates = aggregates.merge(table, on="output_basename", how="outer")

    summary = summary.merge(aggregates, on="output_basename", how="left")
    if "count_from_worm_rows" in summary.columns:
        # Use actual rows when present; preserve metadata zeros otherwise.
        row_count = pd.to_numeric(summary["count_from_worm_rows"], errors="coerce")
        summary["count"] = row_count.fillna(summary["count"]).fillna(0).astype(int)

    return _order_grid_columns(summary)


def load_database(db_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load images, worms and image_summary tables from parquet or CSV."""
    db_dir = Path(db_dir)
    images = _read_table(db_dir / "images")
    worms = _read_table(db_dir / "worms")
    image_summary = _read_table(db_dir / "image_summary")
    return images, worms, image_summary


def _read_table(base_without_suffix: Path) -> pd.DataFrame:
    parquet = base_without_suffix.with_suffix(".parquet")
    csv = base_without_suffix.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Could not find {parquet} or {csv}")


def _normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "output_basename" not in df.columns and "original_filename" in df.columns:
        df["output_basename"] = df["original_filename"].astype(str)
    for col in ["qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "scale_factor", "n_kept_worms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "qr_condition" in df.columns:
        df["qr_condition"] = df["qr_condition"].astype("string")
    return df


def _normalize_worms(worms_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    worms_df = worms_df.copy()
    if worms_df.empty and len(worms_df.columns) == 0:
        return worms_df

    # Join QR metadata into worm rows if older output files did not include it.
    if "output_basename" in worms_df.columns and "output_basename" in metadata_df.columns:
        metadata_cols = [
            c for c in [
                "output_basename",
                "original_filename",
                "qr_detected",
                "qr_text",
                "qr_plot",
                "qr_spalte",
                "qr_reihe",
                "qr_condition",
                "qr_sample_id",
                "scale_factor",
                "coordinate_scale",
            ]
            if c in metadata_df.columns
        ]
        if metadata_cols:
            meta_small = metadata_df[metadata_cols].drop_duplicates("output_basename")
            missing_cols = [c for c in meta_small.columns if c != "output_basename" and c not in worms_df.columns]
            if missing_cols:
                worms_df = worms_df.merge(meta_small[["output_basename"] + missing_cols], on="output_basename", how="left")

    for col in worms_df.columns:
        if col in WORM_QR_AND_IMAGE_COLUMNS:
            continue
        converted = pd.to_numeric(worms_df[col], errors="coerce")
        if converted.notna().sum() > 0 or worms_df[col].dropna().empty:
            worms_df[col] = converted
    for col in ["qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "scale_factor"]:
        if col in worms_df.columns:
            worms_df[col] = pd.to_numeric(worms_df[col], errors="coerce")
    return worms_df


def _numeric_worm_measurement_columns(worms_df: pd.DataFrame) -> list[str]:
    numeric = []
    for col in worms_df.columns:
        if col in WORM_ID_LIKE_COLUMNS or col in WORM_QR_AND_IMAGE_COLUMNS:
            continue
        if col.startswith("qr_"):
            continue
        if col.startswith("bbox_") or col.startswith("centroid_"):
            # Coordinates are useful in the raw table but usually not meaningful
            # as image-level biological readouts.
            continue
        if pd.api.types.is_numeric_dtype(worms_df[col]):
            numeric.append(col)
    return numeric


def _order_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "qr_plot",
        "qr_spalte",
        "qr_reihe",
        "qr_condition",
        "qr_sample_id",
        "qr_text",
        "original_filename",
        "output_basename",
        "count",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def _single_value_or_empty(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return ""
    vals = [v for v in df[col].dropna().unique().tolist()]
    if len(vals) == 1:
        return str(vals[0])
    return ""


def _unique_jsonable(series: pd.Series) -> list[object]:
    values = []
    for value in series.dropna().unique().tolist():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        values.append(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worm-qr-build-db",
        description="Build a parquet analysis database from worm-qr-segment outputs.",
    )
    parser.add_argument(
        "--segmentation_out",
        type=Path,
        required=True,
        help="Output folder produced by worm-qr-segment.",
    )
    parser.add_argument(
        "--db_dir",
        type=Path,
        default=Path("data") / DEFAULT_DB_NAME,
        help="Destination database folder. Default: data/worm_database",
    )
    parser.add_argument(
        "--allow_csv_only",
        action="store_true",
        help="Write CSV mirrors even if parquet support is missing. Not recommended for production.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = build_database(args.segmentation_out, args.db_dir, allow_csv_only=args.allow_csv_only)
    print("Wrote database tables:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
