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
        "repairs QR row/column fields from the decoded QR text, and averages duplicate grid positions by default."
    )

    try:
        summary_df, source_label = load_image_summary()
    except Exception as exc:
        st.error(f"Could not load image_summary.parquet or image_summary.csv from repository root: {exc}")
        st.stop()

    summary_df = repair_qr_metadata(summary_df)

    if summary_df.empty:
        st.warning("The image summary table is empty.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        st.caption(f"Loaded: `{source_label}`")
        if (APP_DIR / "manifest.json").exists():
            with st.expander("Manifest", expanded=False):
                st.json(load_manifest())

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

        metric_cols = metric_columns(filtered, exclude={x_col, y_col})
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

    grid_assigned_mask = coordinates_available(filtered, x_col, y_col)
    grid = make_grid(filtered, x_col=x_col, y_col=y_col, metric=metric, agg=agg, unit_mode=unit_mode, mm_per_px=mm_per_px)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total images", f"{len(filtered):,}")
    kpi2.metric("Grid-assigned images", f"{int(grid_assigned_mask.sum()):,}")
    kpi3.metric("Occupied grid cells", f"{len(grid):,}")
    if "count" in filtered.columns:
        kpi4.metric("Total larvae", f"{int(pd.to_numeric(filtered['count'], errors='coerce').fillna(0).sum()):,}")
    else:
        kpi4.metric("Total larvae", "n/a")

    missing_grid = int((~grid_assigned_mask).sum())
    if missing_grid:
        st.info(
            f"{missing_grid} image(s) are not assigned to a grid cell for the selected axes. "
            "They remain in the total image and larva counts, but are omitted from the grid plots."
        )

    tab_map, tab_3d, tab_table, tab_rows = st.tabs(["2D map", "3D bars", "Counter grid", "Rows"])

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
            [y_col, x_col, metric, "count", "qr_plot", "qr_condition", "qr_sample_id", "qr_extra_suffix", "qr_text", "original_filename", "output_basename"],
            filtered.columns,
        )
        extra = [c for c in filtered.columns if c not in columns_to_show]
        rows_df = filtered[columns_to_show + extra[:25]].copy()
        rows_df = convert_dataframe_units(rows_df, unit_mode=unit_mode, mm_per_px=mm_per_px)
        rows_df = rename_for_display(rows_df, unit_mode=unit_mode)
        st.dataframe(rows_df, use_container_width=True, height=500)
        st.download_button(
            "Download filtered rows as CSV",
            rows_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_image_summary.csv",
            mime="text/csv",
        )

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
    return df.rename(columns={col: display_name(col, unit_mode) for col in df.columns})


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


def index_or_zero(options: Iterable[object], value: object) -> int:
    options = list(options)
    try:
        return options.index(value)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
