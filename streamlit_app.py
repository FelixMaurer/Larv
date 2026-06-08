from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Larvae QR Grid Explorer", layout="wide")

APP_DIR = Path(__file__).resolve().parent
GRID_X_DEFAULT = "qr_reihe"   # Row on x-axis
GRID_Y_DEFAULT = "qr_spalte"  # Column on y-axis
DEFAULT_MM_PER_PX = 0.14
MISSING_LABEL = "<missing>"

DISPLAY_NAMES = {
    "qr_spalte": "Column",
    "qr_reihe": "Row",
    "qr_plot": "Plot",
    "qr_condition": "Condition",
    "qr_sample_id": "Sample ID",
    "qr_extra_suffix": "QR suffix",
    "qr_text": "QR text",
    "qr_raw": "Raw QR text",
    "qr_detected": "QR detected",
    "qr_parsed_relaxed": "QR parsed",
    "original_filename": "Original filename",
    "output_basename": "Output basename",
    "count": "Larva count",
    "count_from_worm_rows": "Larva rows in table",
    "mean_skeleton_length_px": "Mean skeleton length (px)",
    "median_skeleton_length_px": "Median skeleton length (px)",
    "mean_axis_major_px": "Mean major axis length (px)",
    "median_axis_major_px": "Median major axis length (px)",
    "mean_axis_minor_px": "Mean minor axis length (px)",
    "median_axis_minor_px": "Median minor axis length (px)",
    "mean_area_px": "Mean area (px²)",
    "median_area_px": "Median area (px²)",
    "mean_aspect_ratio": "Mean aspect ratio",
    "mean_eccentricity": "Mean eccentricity",
    "mean_solidity": "Mean solidity",
    "mean_perimeter_px": "Mean perimeter (px)",
    "n_raw_masks": "Raw masks",
    "n_rejected_masks": "Rejected masks",
    "valid_region_fraction": "Valid region fraction",
    "count_absolute": "Absolute larva count",
    "count_from_worm_rows_absolute": "Absolute larva rows in table",
    "count_per_kg_plant_weight": "Larvae per kg plant weight",
    "count_from_worm_rows_per_kg_plant_weight": "Larva rows per kg plant weight",
    "count_per_plant": "Larvae per plant",
    "plant_weight_g": "Plant weight (g)",
    "plant_weight_kg": "Plant weight (kg)",
    "n_plants": "Number of plants",
    "plant_weight_per_plant_g": "Plant weight per plant (g)",
    "larvae_in_tray_manual": "Manual larvae/tray note",
    "plot_trap_disassemble": "Plot/trap disassembly note",
    "parcel_weight_match": "Plant-weight metadata matched",
    "parcel_r4s": "Parcel R4S/sample ID",
    "parcel_plot": "Parcel plot",
    "parcel_spalte": "Parcel column",
    "parcel_reihe": "Parcel row",
}

METRIC_COLOR_SCALES = {
    "count": "Viridis",
    "length": "Cividis",
    "skeleton": "Cividis",
    "area": "YlOrBr",
    "aspect": "Plasma",
    "eccentricity": "Turbo",
    "solidity": "Greens",
    "perimeter": "Blues",
    "raw": "Magma",
    "rejected": "Reds",
    "plant": "Greens",
    "weight": "Greens",
    "n_plants": "Teal",
}

QR_RE = re.compile(
    r"Plot\s*(?P<plot>\d+)\s*[_\-\s]+"
    r"Spalte\s*(?P<spalte>\d+)\s*[_\-\s]+"
    r"Reihe\s*(?P<reihe>\d+)\s*[_\-\s]+"
    r"(?P<condition>[A-Za-z0-9]+)\s*[_\-\s]+"
    r"(?P<sample_id>\d+)"
    r"(?:\s*[_\-\s]+(?P<suffix>.*?))?\s*$",
    flags=re.IGNORECASE,
)


def main() -> None:
    st.title("Larvae QR Grid Explorer")
    st.caption(
        "Flat GitHub/Streamlit version. Loads image_summary.parquet or image_summary.csv from the repository root, "
        "repairs QR row/column fields from the decoded QR text, supports grid maps, trend analysis, clustering, and QC/missing-field diagnostics."
    )

    try:
        summary_df, source_label = load_image_summary()
    except Exception as exc:
        st.error(f"Could not load image_summary.parquet or image_summary.csv from repository root: {exc}")
        st.stop()

    summary_df = repair_qr_metadata(summary_df)
    parcel_df, parcel_source_label = load_parcel_metadata()
    summary_df = attach_parcel_metadata(summary_df, parcel_df)
    images_df, images_source_label = load_optional_table("images")
    worms_df, worms_source_label = load_optional_table("worms")

    if summary_df.empty:
        st.warning("The image summary table is empty.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        st.caption(f"Loaded: `{source_label}`")
        if parcel_source_label:
            matched = int(summary_df.get("parcel_weight_match", pd.Series(False, index=summary_df.index)).fillna(False).sum())
            st.caption(f"Parcel metadata: `{parcel_source_label}`; matched weights for {matched:,}/{len(summary_df):,} image rows")
        else:
            st.caption("Parcel metadata: not found (`parcel_metadata.csv` missing)")
        if (APP_DIR / "manifest.json").exists():
            with st.expander("Manifest", expanded=False):
                st.json(load_manifest())
        with st.expander("Loaded tables", expanded=False):
            st.write(f"image_summary: `{source_label}`")
            st.write(f"images: `{images_source_label or 'not found'}`")
            st.write(f"worms: `{worms_source_label or 'not found'}`")
            st.write(f"parcel_metadata: `{parcel_source_label or 'not found'}`")

        st.header("Filters")
        filtered = apply_filters(summary_df)

        st.header("Grid")
        grid_cols = [c for c in filtered.columns if c.startswith("qr_") or pd.api.types.is_numeric_dtype(filtered[c])]
        x_col = st.selectbox(
            "X axis",
            grid_cols,
            index=index_or_zero(grid_cols, GRID_X_DEFAULT),
            format_func=display_name,
        )
        y_col = st.selectbox(
            "Y axis",
            grid_cols,
            index=index_or_zero(grid_cols, GRID_Y_DEFAULT),
            format_func=display_name,
        )

        st.header("Units")
        unit_mode = st.radio(
            "Measurement units",
            ["pixels", "metric"],
            index=1,
            format_func=lambda v: "Pixels" if v == "pixels" else "Metric dimensions",
            help="Pixel mode shows the raw working-pixel values. Metric mode converts length-like metrics to mm and area-like metrics to mm².",
        )
        mm_per_px = st.number_input(
            "Scale factor (mm/px)",
            min_value=0.0001,
            max_value=100.0,
            value=DEFAULT_MM_PER_PX,
            step=0.01,
            format="%.4f",
            help="Conversion factor for one working pixel. Default is 0.14 mm/px.",
        )

        st.header("Plant-weight normalization")
        weight_available = "plant_weight_kg" in filtered.columns and pd.to_numeric(filtered.get("plant_weight_kg"), errors="coerce").gt(0).any()
        normalize_counts_by_weight = st.checkbox(
            "Normalize larva counts by plant weight",
            value=False,
            disabled=not weight_available,
            help="When enabled, larva count metrics are transformed to larvae per kg plant weight for maps, trends and clustering. Absolute counts are retained in separate columns.",
        )
        if normalize_counts_by_weight:
            n_weight = int(pd.to_numeric(filtered.get("plant_weight_kg"), errors="coerce").gt(0).sum())
            st.caption(f"Using plant_weight_kg for {n_weight:,}/{len(filtered):,} currently filtered image rows. Count metric is shown as larvae/kg.")
        elif not weight_available:
            st.caption("No matched positive plant weights are available for the current filters.")

        set_count_display_names(normalize_counts_by_weight)
        analysis_df = apply_count_weight_normalization(filtered, enabled=normalize_counts_by_weight)

        metric_cols = metric_columns(analysis_df, exclude={x_col, y_col})
        if not metric_cols:
            st.error("No numeric metric columns are available in image_summary.")
            st.stop()
        default_metric = "count" if "count" in metric_cols else metric_cols[0]
        metric = st.selectbox(
            "Metric",
            metric_cols,
            index=index_or_zero(metric_cols, default_metric),
            format_func=lambda c: display_name(c, unit_mode),
        )
        metric_label = display_name(metric, unit_mode)
        if unit_mode == "metric" and metric_unit_exponent(metric) == 0:
            st.caption(f"{metric_label} is dimensionless or count-like and is not converted by the scale factor.")

        agg_options = ["mean", "median", "sum", "min", "max", "count"]
        agg = st.selectbox(
            "Duplicate grid positions",
            agg_options,
            index=agg_options.index("mean"),
            help="How to combine multiple images with the same row and column. Mean is the default for all metrics, including larva count.",
        )

        z_scale = st.selectbox(
            "3D bar height transform",
            ["linear", "sqrt", "log1p"],
            index=0,
            help="Transform applied before visual height normalization. Color still encodes the selected metric value.",
        )
        z_height_fraction = st.slider(
            "3D maximum height",
            min_value=0.05,
            max_value=1.50,
            value=0.35,
            step=0.05,
            help="Maximum bar height as a fraction of the grid footprint. Lower values make the 3D plot less vertically exaggerated.",
        )
        show_value_table = st.checkbox("Show pivot table", value=True)

    filtered_for_analysis = analysis_df
    grid_assigned_mask = coordinates_available(filtered_for_analysis, x_col, y_col)
    grid = make_grid(filtered_for_analysis, x_col=x_col, y_col=y_col, metric=metric, agg=agg, unit_mode=unit_mode, mm_per_px=mm_per_px)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total images", f"{len(filtered_for_analysis):,}")
    kpi2.metric("Grid-assigned images", f"{int(grid_assigned_mask.sum()):,}")
    kpi3.metric("Occupied grid cells", f"{len(grid):,}")
    if "count_absolute" in filtered_for_analysis.columns:
        kpi4.metric("Total larvae", f"{int(pd.to_numeric(filtered_for_analysis['count_absolute'], errors='coerce').fillna(0).sum()):,}")
    elif "count" in filtered_for_analysis.columns:
        kpi4.metric("Total larvae", f"{int(pd.to_numeric(filtered_for_analysis['count'], errors='coerce').fillna(0).sum()):,}")
    else:
        kpi4.metric("Total larvae", "n/a")

    missing_grid = int((~grid_assigned_mask).sum())
    if missing_grid:
        st.info(
            f"{missing_grid} image(s) are not assigned to a grid cell for the selected axes. "
            "They remain in the total image and larva counts, but are omitted from the grid plots."
        )

    tab_map, tab_3d, tab_table, tab_rows, tab_metadata, tab_trend, tab_cluster, tab_qc = st.tabs(
        ["2D map", "3D bars", "Counter grid", "Rows", "Metadata maps", "Trend analysis", "Clustering", "QC / missing"]
    )

    with tab_map:
        st.plotly_chart(
            make_heatmap_figure(grid, x_col=x_col, y_col=y_col, metric=metric, metric_label=metric_label),
            use_container_width=True,
        )

    with tab_3d:
        st.plotly_chart(
            make_3d_bar_figure(
                grid,
                x_col=x_col,
                y_col=y_col,
                metric=metric,
                metric_label=metric_label,
                z_scale=z_scale,
                z_height_fraction=z_height_fraction,
            ),
            use_container_width=True,
        )

    with tab_table:
        pivot = pivot_grid(grid)
        st.dataframe(pivot, use_container_width=True)
        st.download_button(
            "Download current grid as CSV",
            pivot.to_csv().encode("utf-8"),
            file_name=f"grid_{metric}_by_{x_col}_{y_col}.csv",
            mime="text/csv",
        )

    with tab_rows:
        columns_to_show = unique_existing_columns(
            [y_col, x_col, metric, "count", "count_absolute", "count_per_kg_plant_weight", "plant_weight_kg", "plant_weight_g", "n_plants", "qr_plot", "qr_condition", "qr_sample_id", "qr_extra_suffix", "qr_text", "original_filename", "output_basename"],
            filtered_for_analysis.columns,
        )
        extra = [c for c in filtered_for_analysis.columns if c not in columns_to_show]
        rows_df = filtered_for_analysis[columns_to_show + extra[:25]].copy()
        rows_df = convert_dataframe_units(rows_df, unit_mode=unit_mode, mm_per_px=mm_per_px)
        rows_df = rename_for_display(rows_df, unit_mode=unit_mode)
        st.dataframe(rows_df, use_container_width=True, height=500)
        st.download_button(
            "Download filtered rows as CSV",
            rows_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_image_summary.csv",
            mime="text/csv",
        )

    with tab_metadata:
        render_metadata_maps(
            filtered_for_analysis,
            x_col=x_col,
            y_col=y_col,
            unit_mode=unit_mode,
            mm_per_px=mm_per_px,
            z_scale=z_scale,
            z_height_fraction=z_height_fraction,
        )

    with tab_trend:
        render_trend_analysis(filtered_for_analysis, metric_cols=metric_cols, default_metric=default_metric, unit_mode=unit_mode, mm_per_px=mm_per_px)

    with tab_cluster:
        render_clustering_analysis(filtered_for_analysis, metric_cols=metric_cols, x_col=x_col, y_col=y_col, unit_mode=unit_mode, mm_per_px=mm_per_px)

    with tab_qc:
        render_qc_analysis(summary_df, filtered, images_df, worms_df, parcel_df, x_col=x_col, y_col=y_col)

    if show_value_table:
        st.subheader("Current grid values")
        st.dataframe(pivot_grid(grid), use_container_width=True)


@st.cache_data(show_spinner=False)
def load_image_summary() -> tuple[pd.DataFrame, str]:
    parquet_path = APP_DIR / "image_summary.parquet"
    csv_path = APP_DIR / "image_summary.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), parquet_path.name
        except Exception:
            if not csv_path.exists():
                raise
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path.name
    raise FileNotFoundError("Expected image_summary.parquet or image_summary.csv next to streamlit_app.py")


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    try:
        import json
        return json.loads((APP_DIR / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(show_spinner=False)
def load_optional_table(stem: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load an optional flat-table artifact from the app root.

    Preference is parquet, then CSV. The app remains usable when auxiliary
    tables are absent, which is useful for lightweight Streamlit deployments.
    """
    parquet_path = APP_DIR / f"{stem}.parquet"
    csv_path = APP_DIR / f"{stem}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), parquet_path.name
        except Exception:
            if not csv_path.exists():
                return None, None
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path), csv_path.name
        except Exception:
            return None, None
    return None, None


@st.cache_data(show_spinner=False)
def load_parcel_metadata() -> tuple[pd.DataFrame | None, str | None]:
    """Load parcel-level plant-weight metadata from the app root."""
    parquet_path = APP_DIR / "parcel_metadata.parquet"
    csv_path = APP_DIR / "parcel_metadata.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), parquet_path.name
        except Exception:
            if not csv_path.exists():
                return None, None
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path), csv_path.name
        except Exception:
            return None, None
    return None, None


def attach_parcel_metadata(summary_df: pd.DataFrame, parcel_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach plant weight / parcel metadata to image summary rows.

    Matching uses QR-derived plot, column, row and R4S/sample_id fields:
    qr_plot -> parcel_plot, qr_spalte -> parcel_spalte, qr_reihe -> parcel_reihe,
    qr_sample_id -> parcel_r4s.
    """
    out = summary_df.copy()
    if parcel_df is None or parcel_df.empty:
        out["parcel_weight_match"] = False
        return add_weight_normalized_columns(out)

    parcels = parcel_df.copy()
    for col in ["parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s", "plant_weight_g", "plant_weight_kg", "n_plants", "plant_weight_per_plant_g"]:
        if col in parcels.columns:
            parcels[col] = pd.to_numeric(parcels[col], errors="coerce")
    join_left = ["qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id"]
    join_right = ["parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s"]
    for col in join_left:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in join_right:
        if col not in parcels.columns:
            parcels[col] = np.nan
    keep_cols = [c for c in [
        "barcode", "parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s",
        "plant_weight_g", "plant_weight_kg", "n_plants", "plant_weight_per_plant_g",
        "observations", "plot_trap_disassemble", "larvae_in_tray_manual", "tray_observation",
    ] if c in parcels.columns]
    parcels = parcels[keep_cols].drop_duplicates(subset=join_right)
    out = out.merge(parcels, left_on=join_left, right_on=join_right, how="left", suffixes=("", "_parcel"))
    out["parcel_weight_match"] = pd.to_numeric(out.get("plant_weight_kg"), errors="coerce").gt(0)
    return add_weight_normalized_columns(out)


def add_weight_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "plant_weight_g" in out.columns and "plant_weight_kg" not in out.columns:
        out["plant_weight_kg"] = pd.to_numeric(out["plant_weight_g"], errors="coerce") / 1000.0
    if "plant_weight_kg" not in out.columns:
        out["plant_weight_kg"] = np.nan
    weight_kg = pd.to_numeric(out["plant_weight_kg"], errors="coerce")
    valid_weight = weight_kg.gt(0)
    for count_col in ["count", "count_from_worm_rows"]:
        if count_col in out.columns:
            counts = pd.to_numeric(out[count_col], errors="coerce")
            out[f"{count_col}_absolute"] = counts
            out[f"{count_col}_per_kg_plant_weight"] = np.where(valid_weight, counts / weight_kg, np.nan)
    if "count" in out.columns and "n_plants" in out.columns:
        n_plants = pd.to_numeric(out["n_plants"], errors="coerce")
        out["count_per_plant"] = np.where(n_plants.gt(0), pd.to_numeric(out["count"], errors="coerce") / n_plants, np.nan)
    return out


def apply_count_weight_normalization(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Globally transform count columns to per-kg plant-weight values when requested."""
    out = df.copy()
    if not enabled:
        return out
    if "plant_weight_kg" not in out.columns:
        return out
    weight_kg = pd.to_numeric(out["plant_weight_kg"], errors="coerce")
    valid_weight = weight_kg.gt(0)
    for col in ["count", "count_from_worm_rows"]:
        if col in out.columns:
            counts = pd.to_numeric(out[col], errors="coerce")
            if f"{col}_absolute" not in out.columns:
                out[f"{col}_absolute"] = counts
            out[col] = np.where(valid_weight, counts / weight_kg, np.nan)
    return out


def set_count_display_names(normalize_counts_by_weight: bool) -> None:
    if normalize_counts_by_weight:
        DISPLAY_NAMES["count"] = "Larvae per kg plant weight"
        DISPLAY_NAMES["count_from_worm_rows"] = "Larva rows per kg plant weight"
    else:
        DISPLAY_NAMES["count"] = "Larva count"
        DISPLAY_NAMES["count_from_worm_rows"] = "Larva rows in table"


def repair_qr_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill QR fields from qr_text/qr_raw, accepting optional suffixes after the numeric sample ID."""
    out = df.copy()
    for col in ["qr_plot", "qr_spalte", "qr_reihe", "qr_condition", "qr_sample_id", "qr_extra_suffix", "qr_parsed_relaxed"]:
        if col not in out.columns:
            out[col] = np.nan if col not in {"qr_condition", "qr_extra_suffix", "qr_parsed_relaxed"} else None

    parsed_flags = []
    for idx, row in out.iterrows():
        parsed = parse_qr_row(row)
        parsed_flags.append(parsed is not None)
        if parsed is None:
            continue
        # Always allow relaxed parser to fill missing fields. Keep existing values when they are already present.
        for col, key in [
            ("qr_plot", "plot"),
            ("qr_spalte", "spalte"),
            ("qr_reihe", "reihe"),
            ("qr_condition", "condition"),
            ("qr_sample_id", "sample_id"),
            ("qr_extra_suffix", "suffix"),
        ]:
            current = out.at[idx, col] if col in out.columns else np.nan
            if pd.isna(current) or current == "":
                out.at[idx, col] = parsed.get(key)
        out.at[idx, "qr_parsed_relaxed"] = True

    if "qr_parsed_relaxed" in out.columns:
        out["qr_parsed_relaxed"] = out["qr_parsed_relaxed"].fillna(False).astype(bool)
    for col in ["qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_qr_row(row: pd.Series) -> dict | None:
    for col in ["qr_text", "qr_raw"]:
        text = row.get(col, None)
        if pd.isna(text) or str(text).strip() == "":
            continue
        parsed = parse_qr_text(str(text))
        if parsed is not None:
            return parsed
    return None


def parse_qr_text(text: str) -> dict | None:
    cleaned = str(text).strip()
    match = QR_RE.search(cleaned)
    if not match:
        return None
    suffix = match.group("suffix")
    if suffix is not None:
        suffix = suffix.strip("_ -") or None
    return {
        "plot": int(match.group("plot")),
        "spalte": int(match.group("spalte")),
        "reihe": int(match.group("reihe")),
        "condition": match.group("condition"),
        "sample_id": int(match.group("sample_id")),
        "suffix": suffix,
    }


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    for col, label, max_distinct in [
        ("qr_plot", "Plot", 200),
        ("qr_condition", "Condition", 100),
        ("qr_sample_id", "Sample ID", 60),
    ]:
        if col in filtered.columns:
            options = sorted_options_with_missing(filtered[col])
            if len(options) <= max_distinct:
                selected = st.multiselect(label, options, default=options, format_func=lambda x: MISSING_LABEL if x == MISSING_LABEL else str(format_axis_label(x)))
                filtered = filter_including_missing(filtered, col, selected)
            else:
                st.caption(f"{label} filter hidden because there are {len(options)} distinct values.")
    return filtered


def filter_including_missing(df: pd.DataFrame, col: str, selected: list[object]) -> pd.DataFrame:
    if not selected:
        return df.iloc[0:0]
    selected_non_missing = [x for x in selected if x != MISSING_LABEL]
    mask = df[col].isin(selected_non_missing)
    if MISSING_LABEL in selected:
        mask = mask | df[col].isna()
    return df[mask]


def sorted_options_with_missing(series: pd.Series) -> list[object]:
    vals = series.dropna().unique().tolist()
    try:
        vals = sorted(vals)
    except TypeError:
        vals = sorted(vals, key=lambda x: str(x))
    if series.isna().any():
        vals.append(MISSING_LABEL)
    return vals


def metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    blocked_prefixes = ("bbox_", "centroid_", "roi_", "crop_", "original_image_", "working_image_")
    blocked_exact = {"qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "scale_factor"}
    cols = []
    for col in df.columns:
        if col in exclude or col in blocked_exact:
            continue
        if col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    preferred = [
        "count",
        "count_per_kg_plant_weight",
        "count_per_plant",
        "plant_weight_kg",
        "plant_weight_g",
        "n_plants",
        "plant_weight_per_plant_g",
        "larvae_in_tray_manual",
        "mean_skeleton_length_px",
        "median_skeleton_length_px",
        "mean_axis_major_px",
        "mean_axis_minor_px",
        "mean_area_px",
        "mean_aspect_ratio",
        "mean_eccentricity",
        "mean_solidity",
        "mean_perimeter_px",
        "n_raw_masks",
        "n_rejected_masks",
        "valid_region_fraction",
    ]
    ordered = [c for c in preferred if c in cols]
    ordered += [c for c in cols if c not in ordered]
    return ordered


def coordinates_available(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    return pd.to_numeric(df[x_col], errors="coerce").notna() & pd.to_numeric(df[y_col], errors="coerce").notna()


def make_grid(df: pd.DataFrame, x_col: str, y_col: str, metric: str, agg: str, unit_mode: str, mm_per_px: float) -> pd.DataFrame:
    work = df[[x_col, y_col, metric]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce") if x_col in {"qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id"} else work[x_col]
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce") if y_col in {"qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id"} else work[y_col]
    work[metric] = convert_metric_series(metric, work[metric], unit_mode=unit_mode, mm_per_px=mm_per_px)
    work = work.dropna(subset=[x_col, y_col, metric])
    if work.empty:
        return pd.DataFrame(columns=[y_col, x_col, "value"])

    group = work.groupby([y_col, x_col], dropna=True)[metric]
    if agg == "count":
        grouped = group.count()
    else:
        grouped = getattr(group, agg)()
    grid = grouped.reset_index(name="value")
    return grid.sort_values([y_col, x_col])


def pivot_grid(grid: pd.DataFrame, complete_numeric_grid: bool = True) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    y_col, x_col = grid.columns[0], grid.columns[1]
    pivot = grid.pivot(index=y_col, columns=x_col, values="value")
    try:
        pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    except Exception:
        pass
    if complete_numeric_grid:
        pivot = complete_numeric_pivot(pivot)
    pivot.index.name = display_name(y_col)
    pivot.columns.name = display_name(x_col)
    return pivot


def complete_numeric_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    if pivot.empty:
        return pivot
    out = pivot
    x_numeric = pd.to_numeric(pd.Index(out.columns), errors="coerce")
    y_numeric = pd.to_numeric(pd.Index(out.index), errors="coerce")
    if len(x_numeric) and np.isfinite(x_numeric).all() and is_integer_like(x_numeric):
        x_full = np.arange(int(np.nanmin(x_numeric)), int(np.nanmax(x_numeric)) + 1)
        out = out.copy()
        out.columns = x_numeric.astype(int)
        out = out.reindex(columns=x_full)
    if len(y_numeric) and np.isfinite(y_numeric).all() and is_integer_like(y_numeric):
        y_full = np.arange(int(np.nanmin(y_numeric)), int(np.nanmax(y_numeric)) + 1)
        out = out.copy()
        out.index = y_numeric.astype(int)
        out = out.reindex(index=y_full)
    return out


def is_integer_like(values: Iterable[object]) -> bool:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return len(arr) > 0 and bool(np.allclose(arr, np.round(arr)))


def axis_positions_ticks_and_range(values: Iterable[object]) -> tuple[list[float], list[float], list[str], list[float]]:
    labels = list(values)
    if not labels:
        return [], [], [], [0.0, 1.0]
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric.notna().all():
        positions = numeric.astype(float).tolist()
    else:
        positions = [float(i) for i in range(1, len(labels) + 1)]
    tickvals = positions
    ticktext = [format_axis_label(v) for v in labels]
    step = min_positive_step(np.asarray(positions, dtype=float))
    pad = step / 2.0
    axis_range = [float(np.nanmin(positions) - pad), float(np.nanmax(positions) + pad)]
    return positions, tickvals, ticktext, axis_range


def min_positive_step(values: np.ndarray) -> float:
    vals = np.sort(np.unique(values[np.isfinite(values)]))
    if len(vals) == 0:
        return 1.0
    if np.allclose(vals, np.round(vals)):
        return 1.0
    if len(vals) < 2:
        return 1.0
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    return float(np.min(diffs)) if len(diffs) else 1.0


def format_axis_label(value: object) -> str:
    try:
        f = float(value)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return str(value)


def heatmap_customdata(pivot: pd.DataFrame) -> np.ndarray:
    data = np.empty((*pivot.shape, 2), dtype=object)
    for yi, y_label in enumerate(pivot.index):
        for xi, x_label in enumerate(pivot.columns):
            data[yi, xi, 0] = format_axis_label(x_label)
            data[yi, xi, 1] = format_axis_label(y_label)
    return data


def square_grid_height(x_range: list[float], y_range: list[float]) -> int:
    x_span = max(abs(float(x_range[1]) - float(x_range[0])), 1.0)
    y_span = max(abs(float(y_range[1]) - float(y_range[0])), 1.0)
    nominal_width_px = 880.0
    height = int(nominal_width_px * y_span / x_span + 140)
    return max(360, min(height, 1600))


def make_heatmap_figure(grid: pd.DataFrame, x_col: str, y_col: str, metric: str, metric_label: str) -> go.Figure:
    pivot = pivot_grid(grid, complete_numeric_grid=True)
    if pivot.empty:
        return empty_figure("No data for current filters")
    x_pos, x_tickvals, x_ticktext, x_range = axis_positions_ticks_and_range(pivot.columns)
    y_pos, y_tickvals, y_ticktext, y_range = axis_positions_ticks_and_range(pivot.index)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_pos,
            y=y_pos,
            colorscale=colorscale_for_metric(metric),
            colorbar=dict(title=metric_label),
            hovertemplate=(
                f"{display_name(x_col)}=%{{customdata[0]}}<br>"
                f"{display_name(y_col)}=%{{customdata[1]}}<br>"
                f"{metric_label}=%{{z:.3g}}<extra></extra>"
            ),
            customdata=heatmap_customdata(pivot),
        )
    )
    fig.update_layout(
        title=f"{metric_label} by {display_name(x_col)} and {display_name(y_col)}",
        xaxis=dict(title=display_name(x_col), tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext, range=x_range, constrain="domain"),
        yaxis=dict(title=display_name(y_col), tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext, range=[y_range[1], y_range[0]], scaleanchor="x", scaleratio=1, constrain="domain"),
        height=square_grid_height(x_range, y_range),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_3d_bar_figure(grid: pd.DataFrame, x_col: str, y_col: str, metric: str, metric_label: str, z_scale: str, z_height_fraction: float) -> go.Figure:
    if grid.empty:
        return empty_figure("No data for current filters")
    grid = grid.copy()
    grid["x_num"] = axis_to_numeric(grid[x_col])
    grid["y_num"] = axis_to_numeric(grid[y_col])
    grid = grid.dropna(subset=["x_num", "y_num", "value"])
    if grid.empty:
        return empty_figure("No finite grid values for current filters")

    raw_values = grid["value"].astype(float).to_numpy()
    transformed = scale_heights(raw_values, z_scale)
    heights = normalize_heights(transformed, grid["x_num"], grid["y_num"], z_height_fraction)
    if np.all(~np.isfinite(heights)):
        return empty_figure("No finite values after height scaling")

    x_width = bar_width(grid["x_num"])
    y_width = bar_width(grid["y_num"])
    x, y, z, i, j, k, intensity = cuboid_mesh(grid["x_num"].to_numpy(float), grid["y_num"].to_numpy(float), heights, raw_values, x_width=x_width, y_width=y_width)

    fig = go.Figure(
        data=go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            intensity=intensity,
            colorscale=colorscale_for_metric(metric),
            colorbar=dict(title=metric_label),
            flatshading=True,
            opacity=0.95,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=grid["x_num"],
            y=grid["y_num"],
            z=np.maximum(heights, 0) + label_lift(heights),
            mode="text",
            text=[short_number(v) for v in raw_values],
            textposition="middle center",
            hovertemplate=(
                f"{display_name(x_col)}=%{{customdata[0]}}<br>"
                f"{display_name(y_col)}=%{{customdata[1]}}<br>"
                f"{metric_label}=%{{customdata[2]:.4g}}<extra></extra>"
            ),
            customdata=np.column_stack([grid[x_col].astype(str), grid[y_col].astype(str), raw_values]),
            showlegend=False,
        )
    )
    x_tickvals, x_ticktext = unique_axis_ticks(grid["x_num"], grid[x_col])
    y_tickvals, y_ticktext = unique_axis_ticks(grid["y_num"], grid[y_col])
    aspect = square_scene_aspect(grid["x_num"], grid["y_num"], heights)
    fig.update_layout(
        title=f"{metric_label} by {display_name(x_col)} and {display_name(y_col)}",
        height=760,
        scene=dict(
            xaxis=dict(title=display_name(x_col), tickmode="array", tickvals=x_tickvals, ticktext=x_ticktext, range=numeric_axis_range_with_padding(grid["x_num"])),
            yaxis=dict(title=display_name(y_col), tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext, range=numeric_axis_range_with_padding(grid["y_num"])),
            zaxis=dict(title=f"{metric_label} display height", range=numeric_z_range_with_padding(heights)),
            camera=dict(eye=dict(x=0.001, y=0.001, z=3.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0), projection=dict(type="orthographic")),
            aspectmode="manual",
            aspectratio=aspect,
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def unique_axis_ticks(numeric_values: pd.Series, labels: pd.Series) -> tuple[list[float], list[str]]:
    tmp = pd.DataFrame({"pos": pd.to_numeric(numeric_values, errors="coerce"), "label": labels.astype(str)})
    tmp = tmp.dropna(subset=["pos"]).drop_duplicates(subset=["pos"]).sort_values("pos")
    return tmp["pos"].astype(float).tolist(), tmp["label"].map(format_axis_label).tolist()


def square_scene_aspect(x_values: pd.Series, y_values: pd.Series, heights: np.ndarray) -> dict[str, float]:
    x_span = axis_span_with_padding(x_values)
    y_span = axis_span_with_padding(y_values)
    z_span = float(np.nanmax(np.abs(heights[np.isfinite(heights)]))) if np.isfinite(heights).any() else 1.0
    z_span = max(z_span, max(x_span, y_span) * 0.04)
    max_span = max(x_span, y_span, z_span, 1.0)
    return {"x": float(x_span / max_span), "y": float(y_span / max_span), "z": float(z_span / max_span)}


def axis_span_with_padding(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return 1.0
    step = min_positive_step(vals)
    return float(max(np.nanmax(vals) - np.nanmin(vals) + step, step))


def numeric_axis_range_with_padding(values: pd.Series, extra_pad_steps: float = 0.65) -> list[float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return [0.0, 1.0]
    step = min_positive_step(vals)
    pad = step * float(extra_pad_steps)
    return [float(np.nanmin(vals) - pad), float(np.nanmax(vals) + pad)]


def numeric_z_range_with_padding(heights: np.ndarray) -> list[float]:
    vals = np.asarray(heights, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [0.0, 1.0]
    z_min = min(0.0, float(np.nanmin(vals)))
    z_max = max(0.0, float(np.nanmax(vals)))
    span = max(z_max - z_min, 1.0)
    return [z_min - 0.05 * span, z_max + 0.18 * span]


def axis_to_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    categories = {value: idx for idx, value in enumerate(sorted_options_with_missing(series), start=1) if value != MISSING_LABEL}
    return series.map(categories).astype(float)


def bar_width(values: pd.Series) -> float:
    unique = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    if len(unique) == 0:
        return 0.65
    return float(min_positive_step(unique) * 0.72)


def scale_heights(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    if mode == "log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values


def normalize_heights(values: np.ndarray, x_values: pd.Series, y_values: pd.Series, height_fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    vmax = float(np.nanmax(np.abs(values[finite])))
    if vmax <= 0:
        out[finite] = 0.0
        return out
    grid_footprint = max(axis_range(x_values), axis_range(y_values), 1.0)
    out[finite] = values[finite] / vmax * grid_footprint * float(height_fraction)
    return out


def axis_range(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return 1.0
    return float(max(vals.max() - vals.min(), 1.0))


def cuboid_mesh(xs: np.ndarray, ys: np.ndarray, heights: np.ndarray, raw_values: np.ndarray, x_width: float, y_width: float) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int], list[float]]:
    vertices_x, vertices_y, vertices_z, intensity = [], [], [], []
    ii, jj, kk = [], [], []
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    for x0, y0, h, raw in zip(xs, ys, heights, raw_values):
        if not np.isfinite(x0) or not np.isfinite(y0) or not np.isfinite(h):
            continue
        z0, z1 = (0.0, float(h)) if h >= 0 else (float(h), 0.0)
        x_left, x_right = float(x0 - x_width / 2), float(x0 + x_width / 2)
        y_front, y_back = float(y0 - y_width / 2), float(y0 + y_width / 2)
        base = len(vertices_x)
        cube = [(x_left, y_front, z0), (x_right, y_front, z0), (x_right, y_back, z0), (x_left, y_back, z0), (x_left, y_front, z1), (x_right, y_front, z1), (x_right, y_back, z1), (x_left, y_back, z1)]
        for vx, vy, vz in cube:
            vertices_x.append(vx); vertices_y.append(vy); vertices_z.append(vz); intensity.append(float(raw))
        for a, b, c in faces:
            ii.append(base + a); jj.append(base + b); kk.append(base + c)
    return vertices_x, vertices_y, vertices_z, ii, jj, kk, intensity


def label_lift(heights: np.ndarray) -> float:
    finite = np.abs(heights[np.isfinite(heights)])
    return float(max(np.nanmax(finite) * 0.025, 0.1)) if len(finite) else 0.1


def colorscale_for_metric(metric: str) -> str:
    lower = metric.lower()
    for key, scale in METRIC_COLOR_SCALES.items():
        if key in lower:
            return scale
    return "Viridis"


def metric_unit_exponent(metric: object) -> int:
    name = str(metric).lower()
    if name.endswith("_area_px") or name.endswith("area_px"):
        return 2
    if name.endswith("_px"):
        spatial_terms = ("length", "axis_major", "axis_minor", "perimeter", "diameter", "width", "height")
        if any(term in name for term in spatial_terms):
            return 1
    return 0


def convert_metric_series(metric: object, values: pd.Series, unit_mode: str, mm_per_px: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    exponent = metric_unit_exponent(metric)
    if unit_mode == "metric" and exponent > 0:
        return numeric * (float(mm_per_px) ** exponent)
    return numeric


def convert_dataframe_units(df: pd.DataFrame, unit_mode: str, mm_per_px: float) -> pd.DataFrame:
    if unit_mode != "metric":
        return df
    out = df.copy()
    for col in out.columns:
        if metric_unit_exponent(col) > 0 and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = convert_metric_series(col, out[col], unit_mode=unit_mode, mm_per_px=mm_per_px)
    return out


def display_name(column: object, unit_mode: str = "pixels") -> str:
    text = str(column)
    exponent = metric_unit_exponent(text)
    label = DISPLAY_NAMES.get(text)
    if label is None:
        label = text.replace("qr_", "QR ").replace("_", " ").title()
        if exponent == 2:
            label = label.removesuffix(" Px") + " (px²)"
        elif exponent == 1:
            label = label.removesuffix(" Px") + " (px)"
    if unit_mode == "metric":
        if exponent == 2:
            label = label.replace("(px²)", "(mm²)").replace("(px^2)", "(mm²)")
        elif exponent == 1:
            label = label.replace("(px)", "(mm)")
    return label


def rename_for_display(df: pd.DataFrame, unit_mode: str = "pixels") -> pd.DataFrame:
    """Rename columns to human-readable labels and keep Arrow/Streamlit-safe uniqueness.

    Streamlit sends dataframes through PyArrow. PyArrow rejects duplicate column
    names, which can happen after display renaming; for example, when global
    plant-weight normalization is enabled both `count` and
    `count_per_kg_plant_weight` can naturally display as larvae/kg. We keep the
    nice labels but append the original source column when needed.
    """
    labels = [display_name(col, unit_mode) for col in df.columns]
    unique_labels = make_unique_display_columns(labels, [str(c) for c in df.columns])
    out = df.copy()
    out.columns = unique_labels
    return out


def make_unique_display_columns(labels: Iterable[str], source_columns: Iterable[str] | None = None) -> list[str]:
    """Return display labels with no duplicates, preserving order.

    Duplicate display labels are disambiguated with the original source column
    name when available, otherwise with a numeric suffix. This prevents
    `ValueError: Duplicate column names found` in `st.dataframe`.
    """
    labels_list = [str(x) for x in labels]
    source_list = [str(x) for x in source_columns] if source_columns is not None else [""] * len(labels_list)

    counts: dict[str, int] = {}
    for label in labels_list:
        counts[label] = counts.get(label, 0) + 1

    seen: dict[str, int] = {}
    used: set[str] = set()
    out: list[str] = []
    for label, source in zip(labels_list, source_list):
        seen[label] = seen.get(label, 0) + 1
        if counts[label] == 1 and label not in used:
            candidate = label
        else:
            source_suffix = source if source else str(seen[label])
            candidate = f"{label} [{source_suffix}]"
        if candidate in used:
            base = candidate
            idx = 2
            while f"{base}.{idx}" in used:
                idx += 1
            candidate = f"{base}.{idx}"
        out.append(candidate)
        used.add(candidate)
    return out


def unique_existing_columns(candidates: list[str], existing_columns: Iterable[str]) -> list[str]:
    existing = set(existing_columns)
    seen: set[str] = set()
    out: list[str] = []
    for col in candidates:
        if col in existing and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def short_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:.2g}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(annotations=[dict(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")], height=500)
    return fig



def metadata_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric parcel/metadata columns that are meaningful as maps/trends."""
    preferred = [
        "plant_weight_kg",
        "plant_weight_g",
        "n_plants",
        "plant_weight_per_plant_g",
        "count_per_kg_plant_weight",
        "count_per_plant",
        "parcel_weight_match",
    ]
    candidates: list[str] = []
    for col in preferred:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    metadata_keywords = (
        "plant_weight", "n_plants", "parcel_weight", "manual", "tray", "plot_trap",
        "pixel_scale", "qr_bbox_width", "qr_bbox_height", "qr_side_mean",
    )
    blocked = {"qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s"}
    for col in df.columns:
        lower = str(col).lower()
        if col in blocked or col in candidates:
            continue
        if any(key in lower for key in metadata_keywords) and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def render_metadata_maps(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    unit_mode: str,
    mm_per_px: float,
    z_scale: str,
    z_height_fraction: float,
) -> None:
    st.subheader("Metadata maps")
    st.caption(
        "Map and trend parcel metadata, especially plant weight, across the row/column grid. "
        "These variables can also be selected in the normal Trend analysis and Clustering tabs."
    )
    if df.empty:
        st.warning("No rows remain after filters.")
        return

    meta_cols = metadata_metric_columns(df)
    if not meta_cols:
        st.info("No numeric parcel metadata columns are available. Add or check `parcel_metadata.csv` in the app root.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        default_meta = "plant_weight_kg" if "plant_weight_kg" in meta_cols else meta_cols[0]
        meta_metric = st.selectbox(
            "Metadata variable",
            meta_cols,
            index=index_or_zero(meta_cols, default_meta),
            format_func=lambda c: display_name(c, unit_mode),
            key="metadata_map_metric",
        )
    with c2:
        agg = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max", "count"], index=0, key="metadata_map_agg")
    with c3:
        use_qr_grid = st.checkbox("Force QR row/column axes", value=True, key="metadata_map_force_qr")

    map_x = "qr_reihe" if use_qr_grid and "qr_reihe" in df.columns else x_col
    map_y = "qr_spalte" if use_qr_grid and "qr_spalte" in df.columns else y_col
    grid = make_grid(df, x_col=map_x, y_col=map_y, metric=meta_metric, agg=agg, unit_mode=unit_mode, mm_per_px=mm_per_px)
    metric_label = display_name(meta_metric, unit_mode)

    finite_values = pd.to_numeric(df.get(meta_metric), errors="coerce")
    finite_values = finite_values[np.isfinite(finite_values)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows with value", f"{int(finite_values.notna().sum()):,}")
    k2.metric("Mean", f"{float(finite_values.mean()):.3g}" if len(finite_values) else "n/a")
    k3.metric("Median", f"{float(finite_values.median()):.3g}" if len(finite_values) else "n/a")
    k4.metric("Occupied cells", f"{len(grid):,}")

    t_heat, t_3d, t_table, t_trends = st.tabs(["2D metadata map", "3D metadata bars", "Metadata grid table", "Metadata trends"])
    with t_heat:
        st.plotly_chart(
            make_heatmap_figure(grid, x_col=map_x, y_col=map_y, metric=meta_metric, metric_label=metric_label),
            use_container_width=True,
        )
    with t_3d:
        st.plotly_chart(
            make_3d_bar_figure(
                grid,
                x_col=map_x,
                y_col=map_y,
                metric=meta_metric,
                metric_label=metric_label,
                z_scale=z_scale,
                z_height_fraction=z_height_fraction,
            ),
            use_container_width=True,
        )
    with t_table:
        pivot = pivot_grid(grid)
        st.dataframe(pivot, use_container_width=True)
        st.download_button(
            "Download metadata grid as CSV",
            pivot.to_csv().encode("utf-8"),
            file_name=f"metadata_grid_{meta_metric}_by_{map_x}_{map_y}.csv",
            mime="text/csv",
        )
    with t_trends:
        render_metadata_trend_pair(df, meta_metric=meta_metric, x_col=map_x, y_col=map_y, unit_mode=unit_mode, mm_per_px=mm_per_px)


def render_metadata_trend_pair(df: pd.DataFrame, meta_metric: str, x_col: str, y_col: str, unit_mode: str, mm_per_px: float) -> None:
    axes = [c for c in [x_col, y_col] if c in df.columns]
    if not axes:
        st.info("No grid axes are available for metadata trends.")
        return
    metric_label = display_name(meta_metric, unit_mode)
    fig = go.Figure()
    trend_tables: list[pd.DataFrame] = []
    for axis in axes:
        work = df[[axis, meta_metric]].copy()
        work["axis_value"] = pd.to_numeric(work[axis], errors="coerce")
        work["metric_value"] = convert_metric_series(meta_metric, work[meta_metric], unit_mode=unit_mode, mm_per_px=mm_per_px)
        work = work.dropna(subset=["axis_value", "metric_value"])
        if work.empty:
            continue
        trend = work.groupby("axis_value", dropna=True)["metric_value"].mean().reset_index(name="value").sort_values("axis_value")
        trend["axis"] = display_name(axis)
        trend_tables.append(trend)
        fig.add_trace(go.Scatter(
            x=trend["axis_value"],
            y=trend["value"],
            mode="lines+markers",
            name=f"Mean by {display_name(axis)}",
        ))
    if not trend_tables:
        st.warning("No finite metadata trend values are available.")
        return
    fig.update_layout(
        title=f"{metric_label} trends across grid axes",
        xaxis_title="Grid coordinate",
        yaxis_title=metric_label,
        height=480,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    trend_all = pd.concat(trend_tables, ignore_index=True)
    stats = trend_regression_table(trend_all.rename(columns={"axis": "trend_axis"}), group_col="trend_axis")
    st.markdown("#### Metadata trend diagnostics")
    st.dataframe(stats, use_container_width=True)
    st.download_button(
        "Download metadata trend values as CSV",
        trend_all.to_csv(index=False).encode("utf-8"),
        file_name=f"metadata_trends_{meta_metric}.csv",
        mime="text/csv",
    )


def render_trend_analysis(df: pd.DataFrame, metric_cols: list[str], default_metric: str, unit_mode: str, mm_per_px: float) -> None:
    st.subheader("Trend analysis")
    st.caption("Aggregate any selected larva metric or parcel metadata variable along a row/column/sample axis and fit simple linear trends. This can be used for larva counts, larvae/kg, plant weight, plant number, and other numeric parcel metadata.")
    if df.empty:
        st.warning("No rows remain after filters.")
        return

    axis_candidates = [c for c in ["qr_reihe", "qr_spalte", "qr_plot", "qr_sample_id", "parcel_reihe", "parcel_spalte", "parcel_plot", "parcel_r4s"] if c in df.columns]
    axis_candidates += [c for c in df.columns if (c.startswith("qr_") or c.startswith("parcel_")) and c not in axis_candidates and pd.api.types.is_numeric_dtype(df[c])]
    if not axis_candidates or not metric_cols:
        st.info("No suitable numeric QR axis or metric is available for trend analysis.")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        trend_axis = st.selectbox("Trend axis", axis_candidates, index=index_or_zero(axis_candidates, "qr_reihe"), format_func=display_name, key="trend_axis")
    with c2:
        trend_metric = st.selectbox("Trend metric", metric_cols, index=index_or_zero(metric_cols, default_metric), format_func=lambda c: display_name(c, unit_mode), key="trend_metric")
    group_options = ["<none>"] + [c for c in ["qr_condition", "qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "parcel_spalte", "parcel_reihe", "parcel_plot", "n_plants"] if c in df.columns and c != trend_axis]
    with c3:
        group_col = st.selectbox("Separate lines by", group_options, index=0, format_func=lambda c: "None" if c == "<none>" else display_name(c), key="trend_group")
    with c4:
        agg = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max", "count"], index=0, key="trend_agg")

    work_cols = [trend_axis, trend_metric] + ([] if group_col == "<none>" else [group_col])
    work = df[work_cols].copy()
    work["axis_value"] = pd.to_numeric(work[trend_axis], errors="coerce")
    work["metric_value"] = convert_metric_series(trend_metric, work[trend_metric], unit_mode=unit_mode, mm_per_px=mm_per_px)
    work = work.dropna(subset=["axis_value", "metric_value"])
    if work.empty:
        st.warning("No finite values are available for this trend selection.")
        return

    group_keys = ["axis_value"] if group_col == "<none>" else [group_col, "axis_value"]
    grouped = work.groupby(group_keys, dropna=False)["metric_value"]
    if agg == "count":
        trend = grouped.count().reset_index(name="value")
    else:
        trend = getattr(grouped, agg)().reset_index(name="value")
    trend = trend.sort_values(group_keys)

    fig = go.Figure()
    if group_col == "<none>":
        fig.add_trace(go.Scatter(x=trend["axis_value"], y=trend["value"], mode="lines+markers", name=display_name(trend_metric, unit_mode)))
    else:
        groups = trend[group_col].astype(str).drop_duplicates().tolist()
        max_groups = 16
        if len(groups) > max_groups:
            st.info(f"Showing the first {max_groups} groups in the plot. The statistics table still includes all groups.")
            groups = groups[:max_groups]
        for group in groups:
            part = trend[trend[group_col].astype(str) == group]
            fig.add_trace(go.Scatter(x=part["axis_value"], y=part["value"], mode="lines+markers", name=str(group)))

    fig.update_layout(
        title=f"{display_name(trend_metric, unit_mode)} trend along {display_name(trend_axis)}",
        xaxis_title=display_name(trend_axis),
        yaxis_title=display_name(trend_metric, unit_mode),
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    stats = trend_regression_table(trend, group_col=None if group_col == "<none>" else group_col)
    st.markdown("#### Linear trend diagnostics")
    st.dataframe(stats, use_container_width=True)
    st.download_button(
        "Download trend table as CSV",
        trend.to_csv(index=False).encode("utf-8"),
        file_name=f"trend_{trend_metric}_by_{trend_axis}.csv",
        mime="text/csv",
    )


def trend_regression_table(trend: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[object, pd.DataFrame]]
    if group_col is None:
        groups = [("all", trend)]
    else:
        groups = [(name, part) for name, part in trend.groupby(group_col, dropna=False)]
    for name, part in groups:
        x = pd.to_numeric(part["axis_value"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(part["value"], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]; y = y[finite]
        row = {"group": name, "n_points": int(len(x))}
        if len(x) >= 2 and float(np.nanstd(x)) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = slope * x + intercept
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            row.update({"slope_per_axis_unit": float(slope), "intercept": float(intercept), "r2": float(r2), "mean_value": float(np.mean(y))})
        else:
            row.update({"slope_per_axis_unit": np.nan, "intercept": np.nan, "r2": np.nan, "mean_value": float(np.mean(y)) if len(y) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")


def render_clustering_analysis(df: pd.DataFrame, metric_cols: list[str], x_col: str, y_col: str, unit_mode: str, mm_per_px: float) -> None:
    st.subheader("Clustering analysis")
    st.caption("Cluster parcels/images by selected numeric metrics. This uses a lightweight k-means implementation in the app, with z-scored features and a PCA preview for interpretation.")
    if df.empty:
        st.warning("No rows remain after filters.")
        return
    if not metric_cols:
        st.info("No numeric metrics are available for clustering.")
        return

    default_features = [c for c in ["count", "count_per_kg_plant_weight", "plant_weight_kg", "n_plants", "plant_weight_per_plant_g", "mean_skeleton_length_px", "mean_area_px", "mean_aspect_ratio", "n_rejected_masks", "valid_region_fraction"] if c in metric_cols]
    if not default_features:
        default_features = metric_cols[: min(4, len(metric_cols))]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        features = st.multiselect("Features", metric_cols, default=default_features, format_func=lambda c: display_name(c, unit_mode), key="cluster_features")
    with c2:
        k = st.slider("Number of clusters", min_value=2, max_value=8, value=3, step=1, key="cluster_k")
    with c3:
        cluster_level = st.selectbox("Cluster level", ["grid cells", "images"], index=0, help="Grid-cell mode averages duplicate row/column positions first.", key="cluster_level")

    if not features:
        st.info("Select at least one feature.")
        return

    work = df.copy()
    for feature in features:
        work[feature] = convert_metric_series(feature, work[feature], unit_mode=unit_mode, mm_per_px=mm_per_px)

    if cluster_level == "grid cells":
        if x_col not in work.columns or y_col not in work.columns:
            st.warning("Selected grid axes are not available.")
            return
        work = work.dropna(subset=[x_col, y_col])
        if work.empty:
            st.warning("No grid-assigned rows are available for clustering.")
            return
        agg = work.groupby([y_col, x_col], dropna=True)[features].mean().reset_index()
        label_cols = [y_col, x_col]
    else:
        id_cols = unique_existing_columns(["original_filename", "output_basename", "qr_plot", "qr_spalte", "qr_reihe", "qr_condition", "qr_sample_id"], work.columns)
        agg = work[id_cols + features].copy()
        label_cols = id_cols

    feature_matrix = agg[features].apply(pd.to_numeric, errors="coerce")
    usable_mask = feature_matrix.notna().any(axis=1)
    agg = agg.loc[usable_mask].reset_index(drop=True)
    feature_matrix = feature_matrix.loc[usable_mask].reset_index(drop=True)
    if len(agg) < 2:
        st.warning("Not enough rows with numeric feature values for clustering.")
        return

    X = feature_matrix.to_numpy(dtype=float)
    X = fill_nan_with_column_median(X)
    Z, means, scales = zscore_matrix(X)
    k_eff = int(min(k, len(Z)))
    labels, centers, inertia = kmeans_numpy(Z, k=k_eff, random_state=7, n_init=20)
    agg["cluster"] = labels + 1

    st.metric("Clustered rows", f"{len(agg):,}")
    st.metric("Within-cluster inertia", f"{inertia:.3g}")

    if cluster_level == "grid cells":
        st.plotly_chart(make_cluster_map(agg, x_col=x_col, y_col=y_col), use_container_width=True)

    pca = pca_scores(Z, n_components=2)
    if pca is not None:
        fig = go.Figure()
        for cluster_id in sorted(agg["cluster"].unique()):
            mask = agg["cluster"] == cluster_id
            fig.add_trace(go.Scatter(
                x=pca[mask.to_numpy(), 0],
                y=pca[mask.to_numpy(), 1],
                mode="markers",
                name=f"Cluster {cluster_id}",
                text=cluster_hover_text(agg.loc[mask], label_cols),
                hovertemplate="%{text}<br>PC1=%{x:.3g}<br>PC2=%{y:.3g}<extra></extra>",
            ))
        fig.update_layout(title="PCA view of clustered feature space", xaxis_title="PC1", yaxis_title="PC2", height=500)
        st.plotly_chart(fig, use_container_width=True)

    profile = agg.groupby("cluster")[features].agg(["count", "mean", "median", "std"])
    st.markdown("#### Cluster profiles")
    st.dataframe(profile, use_container_width=True)

    st.markdown("#### Cluster assignments")
    display = rename_for_display(agg.copy(), unit_mode=unit_mode)
    st.dataframe(display, use_container_width=True, height=420)
    st.download_button(
        "Download cluster assignments as CSV",
        agg.to_csv(index=False).encode("utf-8"),
        file_name="cluster_assignments.csv",
        mime="text/csv",
    )


def fill_nan_with_column_median(X: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        out[:, j] = col
    return out


def zscore_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(X, axis=0)
    scales = np.nanstd(X, axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return (X - means) / scales, means, scales


def kmeans_numpy(Z: np.ndarray, k: int, random_state: int = 0, n_init: int = 10, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    k = max(1, min(int(k), n))
    best_labels = np.zeros(n, dtype=int)
    best_centers = Z[:k].copy()
    best_inertia = float("inf")
    for _ in range(max(1, int(n_init))):
        indices = rng.choice(n, size=k, replace=False)
        centers = Z[indices].copy()
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            distances = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for c in range(k):
                members = Z[labels == c]
                if len(members):
                    centers[c] = members.mean(axis=0)
                else:
                    centers[c] = Z[rng.integers(0, n)]
        inertia = float(((Z - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def pca_scores(Z: np.ndarray, n_components: int = 2) -> np.ndarray | None:
    if Z.shape[0] < 2 or Z.shape[1] < 1:
        return None
    centered = Z - np.mean(Z, axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    comps = min(n_components, U.shape[1])
    scores = U[:, :comps] * S[:comps]
    if comps == 1:
        scores = np.column_stack([scores[:, 0], np.zeros(scores.shape[0])])
    return scores


def make_cluster_map(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    work = df.copy()
    work["x_num"] = axis_to_numeric(work[x_col])
    work["y_num"] = axis_to_numeric(work[y_col])
    fig = go.Figure(go.Scatter(
        x=work["x_num"],
        y=work["y_num"],
        mode="markers+text",
        text=work["cluster"].astype(str),
        textposition="middle center",
        marker=dict(symbol="square", size=24, color=work["cluster"].astype(float), colorscale="Turbo", showscale=True, colorbar=dict(title="Cluster")),
        customdata=np.column_stack([work[x_col].astype(str), work[y_col].astype(str), work["cluster"].astype(str)]),
        hovertemplate=f"{display_name(x_col)}=%{{customdata[0]}}<br>{display_name(y_col)}=%{{customdata[1]}}<br>Cluster=%{{customdata[2]}}<extra></extra>",
    ))
    fig.update_layout(
        title="Cluster map",
        xaxis_title=display_name(x_col),
        yaxis_title=display_name(y_col),
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
        height=620,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def cluster_hover_text(df: pd.DataFrame, label_cols: list[str]) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        parts = [f"{display_name(c)}={format_axis_label(row[c])}" for c in label_cols if c in row.index]
        parts.append(f"Cluster={row.get('cluster', '')}")
        texts.append("<br>".join(parts))
    return texts


def render_qc_analysis(summary_df: pd.DataFrame, filtered: pd.DataFrame, images_df: pd.DataFrame | None, worms_df: pd.DataFrame | None, parcel_df: pd.DataFrame | None, x_col: str, y_col: str) -> None:
    st.subheader("QC / missing fields")
    st.caption("Find unreadable QR codes, incomplete QR fields, duplicate parcels and missing grid cells for the current filter selection.")

    qc_df = summary_df.copy()
    required = [c for c in ["qr_plot", "qr_spalte", "qr_reihe", "qr_condition", "qr_sample_id"] if c in qc_df.columns]
    qr_text_col = "qr_text" if "qr_text" in qc_df.columns else None
    qr_text_ok = qc_df[qr_text_col].notna() & (qc_df[qr_text_col].astype(str).str.strip() != "") if qr_text_col else pd.Series(False, index=qc_df.index)
    qr_detected_ok = qc_df.get("qr_detected", pd.Series(False, index=qc_df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    required_ok = qc_df[required].notna().all(axis=1) if required else pd.Series(False, index=qc_df.index)
    readable = qr_text_ok & qr_detected_ok & required_ok

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Images", f"{len(qc_df):,}")
    k2.metric("Readable QR + fields", f"{int(readable.sum()):,}")
    k3.metric("Unreadable/incomplete QR", f"{int((~readable).sum()):,}")
    if "count" in qc_df.columns:
        k4.metric("Total larvae", f"{int(pd.to_numeric(qc_df['count'], errors='coerce').fillna(0).sum()):,}")
    else:
        k4.metric("Total larvae", "n/a")

    tab_qr, tab_fields, tab_missing, tab_dupes, tab_weights, tab_tables = st.tabs(["Unreadable QR", "Missing QR fields", "Missing parcels", "Duplicate parcels", "Plant weights", "Table inventory"])

    with tab_qr:
        bad = qc_df.loc[~readable].copy()
        st.markdown("#### Images without a readable/complete QR code")
        st.caption("A row is listed when QR text is missing, QR detection is false, or any required QR field is missing after relaxed parsing.")
        show_cols = unique_existing_columns(["original_filename", "output_basename", "input_path", "qr_detected", "qr_text", "qr_method", "qr_plot", "qr_spalte", "qr_reihe", "qr_condition", "qr_sample_id", "overlay_png"], bad.columns)
        st.dataframe(bad[show_cols] if show_cols else bad, use_container_width=True, height=420)
        st.download_button("Download unreadable QR list", bad.to_csv(index=False).encode("utf-8"), file_name="unreadable_or_incomplete_qr_images.csv", mime="text/csv")

    with tab_fields:
        rows = []
        for col in required + ["qr_text", "qr_detected", "qr_method"]:
            if col in qc_df.columns:
                missing = qc_df[col].isna() | (qc_df[col].astype(str).str.strip() == "")
                rows.append({"field": col, "display_name": display_name(col), "missing_count": int(missing.sum()), "missing_fraction": float(missing.mean())})
        field_df = pd.DataFrame(rows)
        st.dataframe(field_df, use_container_width=True)
        st.download_button("Download missing-field summary", field_df.to_csv(index=False).encode("utf-8"), file_name="missing_qr_field_summary.csv", mime="text/csv")

    with tab_missing:
        st.markdown("#### Missing parcels for current filters and selected grid axes")
        missing = missing_grid_cells(filtered, x_col=x_col, y_col=y_col)
        if missing.empty:
            st.success("No missing integer grid cells were found inside the current selected axis ranges, or the axes are not numeric/integer-like.")
        else:
            st.write(f"Missing cells: **{len(missing):,}**")
            st.dataframe(rename_for_display(missing, unit_mode="pixels"), use_container_width=True, height=420)
            st.download_button("Download missing parcels", missing.to_csv(index=False).encode("utf-8"), file_name=f"missing_parcels_by_{x_col}_{y_col}.csv", mime="text/csv")

    with tab_dupes:
        st.markdown("#### Duplicate parcel assignments")
        dupes = duplicate_grid_cells(filtered, x_col=x_col, y_col=y_col)
        if dupes.empty:
            st.success("No duplicate grid cells for the current selected axes.")
        else:
            st.dataframe(rename_for_display(dupes, unit_mode="pixels"), use_container_width=True, height=420)
            st.download_button("Download duplicate parcels", dupes.to_csv(index=False).encode("utf-8"), file_name=f"duplicate_parcels_by_{x_col}_{y_col}.csv", mime="text/csv")

    with tab_weights:
        st.markdown("#### Plant-weight metadata matching")
        if "plant_weight_kg" not in qc_df.columns:
            st.info("No plant-weight columns are available. Add `parcel_metadata.csv` to the app root.")
        else:
            weight_kg = pd.to_numeric(qc_df["plant_weight_kg"], errors="coerce")
            matched = weight_kg.gt(0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows with plant weight", f"{int(matched.sum()):,}")
            c2.metric("Rows without plant weight", f"{int((~matched).sum()):,}")
            c3.metric("Mean plant weight", f"{float(weight_kg[matched].mean()):.3g} kg" if matched.any() else "n/a")
            c4.metric("Median plant weight", f"{float(weight_kg[matched].median()):.3g} kg" if matched.any() else "n/a")
            missing_weight = qc_df.loc[~matched].copy()
            show_cols = unique_existing_columns(["original_filename", "output_basename", "qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id", "qr_text"], missing_weight.columns)
            st.markdown("##### Analyzed images without matched plant weight")
            st.dataframe(missing_weight[show_cols] if show_cols else missing_weight, use_container_width=True, height=280)
            st.download_button("Download images without plant weight", missing_weight.to_csv(index=False).encode("utf-8"), file_name="images_without_plant_weight.csv", mime="text/csv")

            if parcel_df is not None and not parcel_df.empty:
                missing_expected = expected_parcels_without_image(summary_df, parcel_df)
                st.markdown("##### Expected parcel rows without analyzed image")
                if missing_expected.empty:
                    st.success("Every parcel metadata row appears to have at least one matching analyzed image row.")
                else:
                    st.write(f"Missing expected parcel rows: **{len(missing_expected):,}**")
                    st.dataframe(missing_expected, use_container_width=True, height=320)
                    st.download_button("Download expected parcels without analyzed image", missing_expected.to_csv(index=False).encode("utf-8"), file_name="expected_parcels_without_analyzed_image.csv", mime="text/csv")

    with tab_tables:
        rows = [{"table": "image_summary", "available": True, "rows": len(summary_df), "columns": len(summary_df.columns)}]
        rows.append({"table": "images", "available": images_df is not None, "rows": 0 if images_df is None else len(images_df), "columns": 0 if images_df is None else len(images_df.columns)})
        rows.append({"table": "worms", "available": worms_df is not None, "rows": 0 if worms_df is None else len(worms_df), "columns": 0 if worms_df is None else len(worms_df.columns)})
        rows.append({"table": "parcel_metadata", "available": parcel_df is not None, "rows": 0 if parcel_df is None else len(parcel_df), "columns": 0 if parcel_df is None else len(parcel_df.columns)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if images_df is not None:
            with st.expander("Images table preview", expanded=False):
                st.dataframe(images_df.head(200), use_container_width=True)
        if worms_df is not None:
            with st.expander("Worms table preview", expanded=False):
                st.dataframe(worms_df.head(200), use_container_width=True)
        if parcel_df is not None:
            with st.expander("Parcel metadata preview", expanded=False):
                st.dataframe(parcel_df.head(300), use_container_width=True)


def expected_parcels_without_image(summary_df: pd.DataFrame, parcel_df: pd.DataFrame) -> pd.DataFrame:
    if parcel_df is None or parcel_df.empty:
        return pd.DataFrame()
    left = parcel_df.copy()
    right = summary_df.copy()
    left_keys = ["parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s"]
    right_keys = ["qr_plot", "qr_spalte", "qr_reihe", "qr_sample_id"]
    if not all(c in left.columns for c in left_keys) or not all(c in right.columns for c in right_keys):
        return pd.DataFrame()
    for c in left_keys:
        left[c] = pd.to_numeric(left[c], errors="coerce")
    for c in right_keys:
        right[c] = pd.to_numeric(right[c], errors="coerce")
    found = right[right_keys].dropna().drop_duplicates()
    found = found.rename(columns=dict(zip(right_keys, left_keys)))
    merged = left.merge(found.assign(_found=True), on=left_keys, how="left")
    missing = merged[merged["_found"].isna()].drop(columns=["_found"])
    show_cols = [c for c in ["barcode", "parcel_plot", "parcel_spalte", "parcel_reihe", "parcel_r4s", "plant_weight_g", "plant_weight_kg", "n_plants", "observations"] if c in missing.columns]
    return missing[show_cols].sort_values([c for c in ["parcel_spalte", "parcel_reihe", "parcel_plot"] if c in show_cols])


def missing_grid_cells(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    xy = df[[x_col, y_col]].copy()
    xy[x_col] = pd.to_numeric(xy[x_col], errors="coerce")
    xy[y_col] = pd.to_numeric(xy[y_col], errors="coerce")
    xy = xy.dropna()
    if xy.empty:
        return pd.DataFrame()
    if not is_integer_like(xy[x_col]) or not is_integer_like(xy[y_col]):
        return pd.DataFrame()
    x_vals = xy[x_col].round().astype(int)
    y_vals = xy[y_col].round().astype(int)
    x_full = range(int(x_vals.min()), int(x_vals.max()) + 1)
    y_full = range(int(y_vals.min()), int(y_vals.max()) + 1)
    occupied = set(zip(x_vals, y_vals))
    missing = [{x_col: x, y_col: y} for y in y_full for x in x_full if (x, y) not in occupied]
    return pd.DataFrame(missing)


def duplicate_grid_cells(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return pd.DataFrame()
    group = work.groupby([y_col, x_col], dropna=False)
    counts = group.size().reset_index(name="n_images")
    dupes = counts[counts["n_images"] > 1].copy()
    if dupes.empty:
        return dupes
    details = group.agg(
        original_filenames=("original_filename", lambda s: "; ".join(map(str, s.dropna().head(10)))) if "original_filename" in work.columns else (work.columns[0], "count"),
        output_basenames=("output_basename", lambda s: "; ".join(map(str, s.dropna().head(10)))) if "output_basename" in work.columns else (work.columns[0], "count"),
    ).reset_index()
    return dupes.merge(details, on=[y_col, x_col], how="left").sort_values("n_images", ascending=False)


def index_or_zero(options: Iterable[object], value: object) -> int:
    options = list(options)
    try:
        return options.index(value)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
