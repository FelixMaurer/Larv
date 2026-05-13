from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Larvae QR Grid Explorer", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MM_PER_PX = 0.14
DEFAULT_X = "qr_reihe"   # Row on horizontal axis
DEFAULT_Y = "qr_spalte"  # Column on vertical axis

DISPLAY_NAMES = {
    "qr_reihe": "Row",
    "qr_spalte": "Column",
    "qr_plot": "Plot",
    "qr_condition": "Condition",
    "qr_sample_id": "Sample ID",
    "qr_text": "QR text",
    "original_filename": "Original filename",
    "output_basename": "Output basename",
    "count": "Larva count",
    "mean_skeleton_length_px": "Mean skeleton length (px)",
    "median_skeleton_length_px": "Median skeleton length (px)",
    "mean_axis_major_px": "Mean major axis length (px)",
    "median_axis_major_px": "Median major axis length (px)",
    "mean_axis_minor_px": "Mean minor axis length (px)",
    "median_axis_minor_px": "Median minor axis length (px)",
    "mean_area_px": "Mean area (px²)",
    "median_area_px": "Median area (px²)",
    "mean_aspect_ratio": "Mean aspect ratio",
    "median_aspect_ratio": "Median aspect ratio",
    "mean_eccentricity": "Mean eccentricity",
    "mean_solidity": "Mean solidity",
    "mean_perimeter_px": "Mean perimeter (px)",
    "median_perimeter_px": "Median perimeter (px)",
    "n_raw_masks": "Raw masks",
    "n_rejected_masks": "Rejected masks",
    "valid_region_fraction": "Valid region fraction",
}

PREFERRED_METRICS = [
    "count",
    "mean_skeleton_length_px",
    "median_skeleton_length_px",
    "mean_axis_major_px",
    "median_axis_major_px",
    "mean_axis_minor_px",
    "mean_area_px",
    "median_area_px",
    "mean_aspect_ratio",
    "mean_eccentricity",
    "mean_solidity",
    "mean_perimeter_px",
    "n_raw_masks",
    "n_rejected_masks",
    "valid_region_fraction",
]

METRIC_COLOR_SCALES = {
    "count": "Viridis",
    "length": "Cividis",
    "skeleton": "Cividis",
    "axis": "Cividis",
    "area": "YlOrBr",
    "aspect": "Plasma",
    "eccentricity": "Turbo",
    "solidity": "Greens",
    "perimeter": "Blues",
    "rejected": "Reds",
    "raw": "Magma",
}

ID_LIKE_COLUMNS = {
    "qr_reihe", "qr_spalte", "qr_plot", "qr_sample_id", "qr_detected",
    "original_image_height_px", "original_image_width_px",
    "working_image_height_px", "working_image_width_px",
    "roi_y0", "roi_y1", "roi_x0", "roi_x1",
    "crop_height_px", "crop_width_px", "coordinate_scale", "scale_factor",
}


def main() -> None:
    st.title("Larvae QR Grid Explorer")
    st.caption(
        "Flat GitHub/Streamlit version. Loads image_summary.parquet or image_summary.csv "
        "from the repository root, plots QR grid data, and averages duplicate grid positions by default."
    )

    summary_df = load_required_table("image_summary")
    images_df = load_optional_table("images")
    larvae_df = load_optional_table("worms")  # existing export filename; UI still calls these larvae

    if summary_df.empty:
        st.error("image_summary is empty.")
        st.stop()

    summary_df = clean_columns(summary_df)
    summary_df = normalize_common_columns(summary_df)

    required = {DEFAULT_X, DEFAULT_Y}
    missing = sorted(required - set(summary_df.columns))
    if missing:
        st.error(f"Missing required QR grid columns: {', '.join(missing)}")
        st.write("Available columns:", list(summary_df.columns))
        st.stop()

    with st.sidebar:
        st.header("Filters")
        filtered = apply_filters(summary_df)

        st.header("Grid")
        grid_cols = available_grid_columns(filtered)
        x_col = st.selectbox(
            "X axis",
            grid_cols,
            index=index_or_zero(grid_cols, DEFAULT_X),
            format_func=display_name,
        )
        y_col = st.selectbox(
            "Y axis",
            grid_cols,
            index=index_or_zero(grid_cols, DEFAULT_Y),
            format_func=display_name,
        )

        st.header("Units")
        unit_mode = st.radio(
            "Measurement units",
            ["metric", "pixels"],
            index=0,
            format_func=lambda v: "Metric dimensions" if v == "metric" else "Pixels",
        )
        mm_per_px = st.number_input(
            "Scale factor (mm/px)",
            min_value=0.0001,
            max_value=100.0,
            value=DEFAULT_MM_PER_PX,
            step=0.01,
            format="%.4f",
        )

        st.header("Metric")
        metrics = metric_columns(filtered, exclude={x_col, y_col})
        if not metrics:
            st.error("No numeric metric columns found.")
            st.stop()
        default_metric = "count" if "count" in metrics else metrics[0]
        metric = st.selectbox(
            "Parameter",
            metrics,
            index=index_or_zero(metrics, default_metric),
            format_func=lambda c: display_name(c, unit_mode=unit_mode),
        )
        metric_label = display_name(metric, unit_mode=unit_mode)

        agg = st.selectbox(
            "Duplicate grid positions",
            ["mean", "median", "sum", "min", "max", "count"],
            index=0,
            help="Mean is the default. This also averages duplicated larva counts at the same grid position.",
        )

        st.header("3D display")
        z_transform = st.selectbox("Height transform", ["linear", "sqrt", "log1p"], index=0)
        max_height = st.slider(
            "Maximum 3D height",
            min_value=0.05,
            max_value=1.50,
            value=0.35,
            step=0.05,
            help="Visual bar-height compression. Colors still represent the selected metric values.",
        )

    if filtered.empty:
        st.warning("No rows remain after filtering.")
        st.stop()

    grid = make_grid(
        filtered,
        x_col=x_col,
        y_col=y_col,
        metric=metric,
        agg=agg,
        unit_mode=unit_mode,
        mm_per_px=mm_per_px,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Images", f"{len(filtered):,}")
    kpi2.metric("Grid positions", f"{len(grid):,}")
    kpi3.metric("Non-missing values", f"{grid['value'].notna().sum():,}")
    if "count" in filtered.columns:
        kpi4.metric("Total larvae", f"{pd.to_numeric(filtered['count'], errors='coerce').fillna(0).sum():.0f}")
    elif not larvae_df.empty:
        kpi4.metric("Larva rows", f"{len(larvae_df):,}")
    else:
        kpi4.metric("Larva rows", "n/a")

    tab_2d, tab_3d, tab_values, tab_rows = st.tabs(["2D map", "3D bars", "Counter grid", "Rows"])

    with tab_2d:
        st.plotly_chart(
            make_heatmap(grid, x_col=x_col, y_col=y_col, metric=metric, metric_label=metric_label),
            use_container_width=True,
        )

    with tab_3d:
        st.plotly_chart(
            make_3d_bars(
                grid,
                x_col=x_col,
                y_col=y_col,
                metric=metric,
                metric_label=metric_label,
                z_transform=z_transform,
                max_height=max_height,
            ),
            use_container_width=True,
        )

    with tab_values:
        pivot = pivot_grid(grid)
        st.dataframe(pivot, use_container_width=True)
        st.download_button(
            "Download current grid as CSV",
            pivot.to_csv().encode("utf-8"),
            file_name=f"grid_{metric}_by_{x_col}_{y_col}.csv",
            mime="text/csv",
        )

    with tab_rows:
        show_cols = unique_existing_columns(
            [y_col, x_col, metric, "count", "qr_plot", "qr_condition", "qr_sample_id", "original_filename", "output_basename", "qr_text"],
            filtered.columns,
        )
        extra = [c for c in filtered.columns if c not in show_cols][:30]
        rows = filtered[show_cols + extra].copy()
        rows = convert_dataframe_units(rows, unit_mode=unit_mode, mm_per_px=mm_per_px)
        rows = rename_for_display(rows, unit_mode=unit_mode)
        st.dataframe(rows, use_container_width=True, height=520)


def load_required_table(stem: str) -> pd.DataFrame:
    df = load_optional_table(stem)
    if df.empty:
        st.error(
            f"Could not find {stem}.parquet or {stem}.csv next to streamlit_app.py. "
            "For the flat GitHub version, place image_summary.parquet in the repository root."
        )
        st.stop()
    return df


def load_optional_table(stem: str) -> pd.DataFrame:
    parquet_path = APP_DIR / f"{stem}.parquet"
    csv_path = APP_DIR / f"{stem}.csv"
    try:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not load {stem}: {exc}")
        st.stop()
    return pd.DataFrame()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remove exact duplicates and CSV-mangled duplicates such as count.1 when present.
    df = df.loc[:, ~df.columns.duplicated()].copy()
    keep = []
    seen = set()
    for col in df.columns:
        base = col.rsplit(".", 1)[0] if col.rsplit(".", 1)[-1].isdigit() else col
        if base in seen:
            continue
        keep.append(col)
        seen.add(base)
    df = df[keep].copy()
    df.columns = [c.rsplit(".", 1)[0] if c.rsplit(".", 1)[-1].isdigit() else c for c in df.columns]
    return df


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["qr_reihe", "qr_spalte", "qr_plot", "qr_sample_id", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "qr_detected" in out.columns:
        only_detected = st.checkbox("Only rows with decoded QR", value=False)
        if only_detected:
            s = out["qr_detected"]
            if s.dtype == bool:
                out = out[s]
            else:
                out = out[s.astype(str).str.lower().isin(["true", "1", "yes"])]

    for col in ["qr_plot", "qr_condition", "qr_sample_id"]:
        if col not in out.columns:
            continue
        options = sorted([x for x in out[col].dropna().unique().tolist()], key=lambda v: str(v))
        if not options:
            continue
        selected = st.multiselect(display_name(col), options, default=options)
        out = out[out[col].isin(selected)]

    return out


def available_grid_columns(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in ["qr_reihe", "qr_spalte", "qr_plot", "qr_sample_id"] if c in df.columns]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in preferred]
    return preferred + numeric


def metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    preferred = [c for c in PREFERRED_METRICS if c in numeric]
    rest = [c for c in numeric if c not in preferred and c not in ID_LIKE_COLUMNS]
    return preferred + rest


def make_grid(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    metric: str,
    agg: str,
    unit_mode: str,
    mm_per_px: float,
) -> pd.DataFrame:
    tmp = df[[x_col, y_col, metric]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])
    tmp["value"] = convert_series_units(tmp[metric], metric, unit_mode=unit_mode, mm_per_px=mm_per_px)

    grouped = tmp.groupby([x_col, y_col], dropna=True)["value"].agg(agg).reset_index()
    grouped = grouped.sort_values([y_col, x_col]).reset_index(drop=True)

    # Include missing integer grid positions as empty cells, so increments stay square.
    if is_integerish(grouped[x_col]) and is_integerish(grouped[y_col]) and not grouped.empty:
        xs = np.arange(int(np.nanmin(grouped[x_col])), int(np.nanmax(grouped[x_col])) + 1)
        ys = np.arange(int(np.nanmin(grouped[y_col])), int(np.nanmax(grouped[y_col])) + 1)
        full = pd.MultiIndex.from_product([xs, ys], names=[x_col, y_col]).to_frame(index=False)
        grouped = full.merge(grouped, on=[x_col, y_col], how="left")

    return grouped


def pivot_grid(grid: pd.DataFrame) -> pd.DataFrame:
    x_col, y_col = grid.columns[0], grid.columns[1]
    return grid.pivot(index=y_col, columns=x_col, values="value").sort_index(ascending=True)


def make_heatmap(grid: pd.DataFrame, *, x_col: str, y_col: str, metric: str, metric_label: str) -> go.Figure:
    pivot = pivot_grid(grid)
    color_scale = color_scale_for(metric)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=color_scale,
            colorbar={"title": metric_label},
            hovertemplate=f"{display_name(x_col)}=%{{x}}<br>{display_name(y_col)}=%{{y}}<br>{metric_label}=%{{z:.4g}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{metric_label} by {display_name(x_col)} and {display_name(y_col)}",
        xaxis_title=display_name(x_col),
        yaxis_title=display_name(y_col),
        height=760,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(dtick=1, constrain="domain")
    fig.update_yaxes(dtick=1, scaleanchor="x", scaleratio=1, autorange="reversed")
    return fig


def make_3d_bars(
    grid: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    metric: str,
    metric_label: str,
    z_transform: str,
    max_height: float,
) -> go.Figure:
    data = grid.dropna(subset=["value"]).copy()
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title="No values to plot")
        return fig

    xs = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(data[y_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(data["value"], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    xs, ys, values = xs[finite], ys[finite], values[finite]
    if len(values) == 0:
        fig.update_layout(title="No finite values to plot")
        return fig

    z_vis = transform_values_for_height(values, z_transform)
    z_vis = np.clip(z_vis, 0, None)
    z_span = float(np.nanmax(z_vis)) if np.nanmax(z_vis) > 0 else 1.0

    x_extent = max(float(np.nanmax(xs) - np.nanmin(xs) + 1), 1.0)
    y_extent = max(float(np.nanmax(ys) - np.nanmin(ys) + 1), 1.0)
    grid_footprint = max(x_extent, y_extent)
    heights = z_vis / z_span * (max_height * grid_footprint)

    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    cscale = color_scale_for(metric)

    for x, y, value, height in zip(xs, ys, values, heights):
        color_value = 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
        color = sample_plotly_color(cscale, color_value)
        add_box(fig, x=x, y=y, z0=0.0, dz=float(height), color=color, opacity=0.92)

    # Add invisible anchors to force full centered view.
    xmin, xmax = float(np.nanmin(xs) - 0.75), float(np.nanmax(xs) + 0.75)
    ymin, ymax = float(np.nanmin(ys) - 0.75), float(np.nanmax(ys) + 0.75)
    zmax = max(float(np.nanmax(heights)) * 1.10, 1.0)
    fig.add_trace(go.Scatter3d(
        x=[xmin, xmax], y=[ymin, ymax], z=[0, zmax], mode="markers",
        marker=dict(size=1, opacity=0), showlegend=False, hoverinfo="skip"
    ))

    max_xy = max(x_extent, y_extent, 1.0)
    fig.update_layout(
        title=f"{metric_label} by {display_name(x_col)} and {display_name(y_col)}",
        height=820,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(title=display_name(x_col), range=[xmin, xmax], dtick=1),
            yaxis=dict(title=display_name(y_col), range=[ymin, ymax], dtick=1),
            zaxis=dict(title="Visual height", range=[0, zmax]),
            aspectmode="manual",
            aspectratio=dict(x=x_extent / max_xy, y=y_extent / max_xy, z=max_height),
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=0.0, y=0.0, z=2.8),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=1.0, z=0.0),
            ),
        ),
    )
    return fig


def add_box(fig: go.Figure, *, x: float, y: float, z0: float, dz: float, color: str, opacity: float) -> None:
    half = 0.42
    z1 = max(z0 + dz, z0 + 0.01)
    vx = [x-half, x+half, x+half, x-half, x-half, x+half, x+half, x-half]
    vy = [y-half, y-half, y+half, y+half, y-half, y-half, y+half, y+half]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    faces_i = [0, 0, 0, 4, 4, 4, 0, 1, 2, 3, 0, 3]
    faces_j = [1, 2, 3, 5, 6, 7, 4, 5, 6, 7, 1, 2]
    faces_k = [2, 3, 0, 6, 7, 4, 5, 6, 7, 4, 5, 6]
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=faces_i, j=faces_j, k=faces_k,
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        showscale=False,
    ))


def transform_values_for_height(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(float)
    min_value = np.nanmin(values)
    shifted = values - min(0.0, min_value)
    if mode == "sqrt":
        return np.sqrt(np.clip(shifted, 0, None))
    if mode == "log1p":
        return np.log1p(np.clip(shifted, 0, None))
    return shifted


def color_scale_for(metric: str) -> str:
    lower = metric.lower()
    for key, scale in METRIC_COLOR_SCALES.items():
        if key in lower:
            return scale
    return "Viridis"


def sample_plotly_color(colorscale: str, t: float) -> str:
    # Small dependency-free approximation of Plotly scale sampling.
    # It is sufficient for per-bar colors in this lightweight app.
    palettes = {
        "Viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        "Cividis": [(0, 34, 78), (40, 99, 129), (115, 137, 117), (188, 174, 97), (253, 234, 69)],
        "YlOrBr": [(255, 255, 212), (254, 217, 142), (254, 153, 41), (217, 95, 14), (153, 52, 4)],
        "Plasma": [(13, 8, 135), (126, 3, 168), (203, 71, 119), (248, 149, 64), (240, 249, 33)],
        "Turbo": [(48, 18, 59), (43, 117, 231), (56, 204, 92), (255, 196, 56), (122, 4, 3)],
        "Greens": [(237, 248, 233), (186, 228, 179), (116, 196, 118), (49, 163, 84), (0, 109, 44)],
        "Blues": [(239, 243, 255), (189, 215, 231), (107, 174, 214), (33, 113, 181), (8, 48, 107)],
        "Magma": [(0, 0, 4), (80, 18, 123), (182, 55, 121), (251, 136, 97), (252, 253, 191)],
        "Reds": [(254, 229, 217), (252, 174, 145), (251, 106, 74), (222, 45, 38), (165, 15, 21)],
    }
    pts = palettes.get(colorscale, palettes["Viridis"])
    t = float(np.clip(t, 0, 1))
    pos = t * (len(pts) - 1)
    i = int(np.floor(pos))
    j = min(i + 1, len(pts) - 1)
    f = pos - i
    rgb = tuple(int(round((1 - f) * pts[i][k] + f * pts[j][k])) for k in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def convert_series_units(s: pd.Series, metric: str, *, unit_mode: str, mm_per_px: float) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    if unit_mode != "metric":
        return values
    exponent = metric_unit_exponent(metric)
    if exponent == 1:
        return values * mm_per_px
    if exponent == 2:
        return values * (mm_per_px ** 2)
    return values


def convert_dataframe_units(df: pd.DataFrame, *, unit_mode: str, mm_per_px: float) -> pd.DataFrame:
    out = df.copy()
    if unit_mode != "metric":
        return out
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = convert_series_units(out[col], col, unit_mode=unit_mode, mm_per_px=mm_per_px)
    return out


def metric_unit_exponent(metric: str) -> int:
    lower = metric.lower()
    if "area" in lower and ("px" in lower or lower.endswith("area")):
        return 2
    length_terms = ["length", "perimeter", "axis", "diameter", "radius", "width", "height"]
    if any(term in lower for term in length_terms) and "px" in lower:
        return 1
    return 0


def display_name(col: str, unit_mode: str | None = None) -> str:
    name = DISPLAY_NAMES.get(col, col.replace("_", " ").strip().title())
    if unit_mode == "metric":
        exponent = metric_unit_exponent(col)
        if exponent == 1:
            return name.replace("(px)", "(mm)")
        if exponent == 2:
            return name.replace("(px²)", "(mm²)")
    return name


def rename_for_display(df: pd.DataFrame, *, unit_mode: str) -> pd.DataFrame:
    return df.rename(columns={c: display_name(c, unit_mode=unit_mode) for c in df.columns})


def index_or_zero(items: list[str], wanted: str) -> int:
    try:
        return items.index(wanted)
    except ValueError:
        return 0


def unique_existing_columns(wanted: Iterable[str], columns: Iterable[str]) -> list[str]:
    colset = set(columns)
    out = []
    for col in wanted:
        if col in colset and col not in out:
            out.append(col)
    return out


def is_integerish(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return False
    return bool(np.all(np.isclose(vals, np.round(vals))))


if __name__ == "__main__":
    main()
