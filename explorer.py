"""Interactive analysis lab for the Larvae Explorer.

Adds the exploratory half of the app, kept separate from ``streamlit_app.py`` so the
validated field/genotype analyses there stay untouched:

  * **Sample labels**  - rename / annotate / exclude samples; overrides travel with
    the session file and are applied to every view and export.
  * **Plot studio**    - build any scatter/box/violin/histogram/line from the data,
    restyle it, and export SVG / PNG / HTML.
  * **Gating**         - lasso or box-select a region of a 2-D point cloud, save it as
    a named gate, combine gates, and export the larvae inside them. Gates are stored
    as coordinates (not row indices) so they survive a reload and apply to the full
    table even when drawn on a subsample.
  * **GMM lab**        - refit the size mixture for any k on any feature set, inspect
    it in 3-D, and export per-larva assignments.
  * **Statistics**     - group-by summaries, group comparisons (Kruskal-Wallis /
    Mann-Whitney / ANOVA / t-test) and correlation+regression, at larva, image or
    sample level.
  * **Session file**   - download everything above as one JSON and restore it later.

Everything is driven from ``st.session_state['lab']`` so a single dict round-trips.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

LAB = "lab"
SESSION_VERSION = 1
MAX_POINTS_DEFAULT = 30_000


# --------------------------------------------------------------------------- state
def lab_state() -> dict:
    """The one dict that holds every explorer setting (and thus the session file)."""
    if LAB not in st.session_state:
        st.session_state[LAB] = {"gates": [], "labels": {}, "excluded": [], "notes": {}}
    s = st.session_state[LAB]
    for k, v in (("gates", []), ("labels", {}), ("excluded", []), ("notes", {})):
        s.setdefault(k, v)
    return s


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))


def grouping_columns(df: pd.DataFrame, max_levels: int = 400) -> list[str]:
    """Columns usable as a grouping/colour key: few enough distinct values to be useful."""
    out = []
    for c in df.columns:
        try:
            n = df[c].nunique(dropna=True)
        except TypeError:
            continue
        if 1 < n <= max_levels:
            out.append(c)
    return sorted(out)


def subsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if n and len(df) > n else df


# ------------------------------------------------------------------- figure export
def figure_exports(fig: go.Figure, basename: str, key: str) -> None:
    """SVG / PNG / HTML download buttons + an SVG-by-default modebar camera."""
    c1, c2, c3 = st.columns(3)
    try:
        svg = fig.to_image(format="svg")
        c1.download_button("Download SVG", svg, file_name=f"{basename}.svg",
                           mime="image/svg+xml", key=f"{key}_svg")
    except Exception as exc:                                   # kaleido missing/broken
        c1.caption(f"SVG export unavailable ({type(exc).__name__}); "
                   "use the camera icon on the chart.")
    try:
        png = fig.to_image(format="png", scale=2)
        c2.download_button("Download PNG", png, file_name=f"{basename}.png",
                           mime="image/png", key=f"{key}_png")
    except Exception:
        pass
    c3.download_button("Download interactive HTML",
                       fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                       file_name=f"{basename}.html", mime="text/html", key=f"{key}_html")


def show_figure(fig: go.Figure, basename: str, key: str, **kwargs) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"toImageButtonOptions": {"format": "svg", "filename": basename},
                            "displaylogo": False,
                            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"]},
                    **kwargs)
    figure_exports(fig, basename, key)


def style_controls(prefix: str) -> dict:
    """Shared look-and-feel controls so any figure can be tuned before export."""
    with st.expander("Style & axes", expanded=False):
        c1, c2, c3 = st.columns(3)
        s = {
            "title": c1.text_input("Title", "", key=f"{prefix}_title"),
            "template": c1.selectbox("Theme", ["plotly_white", "simple_white", "plotly",
                                               "plotly_dark", "ggplot2", "seaborn"],
                                     key=f"{prefix}_template"),
            "xlabel": c2.text_input("X label (blank = auto)", "", key=f"{prefix}_xlab"),
            "ylabel": c2.text_input("Y label (blank = auto)", "", key=f"{prefix}_ylab"),
            "width": c3.number_input("Width (px)", 400, 3000, 1000, 50, key=f"{prefix}_w"),
            "height": c3.number_input("Height (px)", 300, 2000, 600, 50, key=f"{prefix}_h"),
        }
        c4, c5, c6 = st.columns(3)
        s["marker"] = c4.slider("Marker size", 1, 20, 5, key=f"{prefix}_ms")
        s["opacity"] = c5.slider("Opacity", 0.05, 1.0, 0.7, 0.05, key=f"{prefix}_op")
        s["font"] = c6.slider("Font size", 8, 28, 14, key=f"{prefix}_fs")
        c7, c8 = st.columns(2)
        s["logx"] = c7.checkbox("Log X", key=f"{prefix}_logx")
        s["logy"] = c8.checkbox("Log Y", key=f"{prefix}_logy")
        s["xrange"] = c7.text_input("X range (min,max)", "", key=f"{prefix}_xr")
        s["yrange"] = c8.text_input("Y range (min,max)", "", key=f"{prefix}_yr")
    return s


def _parse_range(txt: str):
    try:
        lo, hi = (float(v) for v in txt.split(","))
        return [lo, hi]
    except Exception:
        return None


def apply_style(fig: go.Figure, s: dict) -> go.Figure:
    fig.update_layout(template=s["template"], width=int(s["width"]), height=int(s["height"]),
                      font=dict(size=int(s["font"])), title=s["title"] or None)
    if s["xlabel"]:
        fig.update_xaxes(title_text=s["xlabel"])
    if s["ylabel"]:
        fig.update_yaxes(title_text=s["ylabel"])
    if s["logx"]:
        fig.update_xaxes(type="log")
    if s["logy"]:
        fig.update_yaxes(type="log")
    if (r := _parse_range(s["xrange"])):
        fig.update_xaxes(range=r)
    if (r := _parse_range(s["yrange"])):
        fig.update_yaxes(range=r)
    fig.update_traces(marker=dict(size=int(s["marker"]), opacity=float(s["opacity"])),
                      selector=dict(type="scattergl"))
    fig.update_traces(marker=dict(size=int(s["marker"]), opacity=float(s["opacity"])),
                      selector=dict(type="scatter"))
    return fig


def download_table(df: pd.DataFrame, basename: str, key: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button(f"Download CSV ({len(df):,} rows)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{basename}.csv", mime="text/csv", key=f"{key}_csv")
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
        c2.download_button("Download Parquet", buf.getvalue(),
                           file_name=f"{basename}.parquet", key=f"{key}_pq")
    except Exception:
        pass


# ------------------------------------------------------------------ sample labels
def apply_label_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Attach user-edited sample labels; also drops samples the user excluded."""
    s = lab_state()
    if "qr_sample_id" not in df.columns:
        return df
    out = df.copy()
    sid = pd.to_numeric(out["qr_sample_id"], errors="coerce")
    if s["labels"]:
        mapping = {float(k): v for k, v in s["labels"].items()}
        out["sample_label"] = sid.map(mapping)
    else:
        out["sample_label"] = np.nan
    base = out["genotype"] if "genotype" in out.columns else sid.astype("string")
    out["sample_label"] = out["sample_label"].fillna(base).fillna(sid.astype("string"))
    if s["notes"]:
        out["sample_note"] = sid.map({float(k): v for k, v in s["notes"].items()})
    if s["excluded"]:
        out = out[~sid.isin([float(x) for x in s["excluded"]])]
    return out


def render_label_editor(summary_df: pd.DataFrame, worms_df: pd.DataFrame | None,
                        parcel_df: pd.DataFrame | None) -> None:
    st.subheader("Sample labels")
    st.caption("Rename samples, add notes, or exclude them. Edits apply to every tab and "
               "every export in this session, and are saved in the session file. The "
               "underlying data files are never modified.")
    if "qr_sample_id" not in summary_df.columns:
        st.info("No sample id column in this project.")
        return
    s = lab_state()

    g = summary_df.copy()
    g["qr_sample_id"] = pd.to_numeric(g["qr_sample_id"], errors="coerce")
    agg = {"images": ("qr_sample_id", "size")}
    if "count" in g.columns:
        agg["larvae"] = ("count", "sum")
    if "genotype" in g.columns:
        agg["genotype"] = ("genotype", "first")
    if "qr_plot" in g.columns:
        agg["plot"] = ("qr_plot", "first")
    tbl = g.dropna(subset=["qr_sample_id"]).groupby("qr_sample_id").agg(**agg).reset_index()
    # NB: force string dtype - with no genotype column these would be all-NaN floats,
    # which st.data_editor refuses to render as an editable text column.
    tbl["label"] = tbl["qr_sample_id"].map({float(k): v for k, v in s["labels"].items()})
    if "genotype" in tbl.columns:
        tbl["label"] = tbl["label"].fillna(tbl["genotype"])
    tbl["label"] = tbl["label"].astype("object").where(tbl["label"].notna(), "").astype(str)
    tbl["note"] = (tbl["qr_sample_id"].map({float(k): v for k, v in s["notes"].items()})
                   .astype("object").where(lambda x: x.notna(), "").astype(str))
    tbl["exclude"] = tbl["qr_sample_id"].isin([float(x) for x in s["excluded"]])

    q = st.text_input("Filter samples (id / genotype / label contains)", "", key="lab_lbl_q")
    view = tbl
    if q:
        m = pd.Series(False, index=tbl.index)
        for c in ("qr_sample_id", "genotype", "label"):
            if c in tbl.columns:
                m |= tbl[c].astype(str).str.contains(q, case=False, na=False)
        view = tbl[m]

    edited = st.data_editor(
        view, key="lab_label_editor", use_container_width=True, hide_index=True,
        column_config={
            "qr_sample_id": st.column_config.NumberColumn("R4S", disabled=True, format="%d"),
            "genotype": st.column_config.TextColumn("Genotype (original)", disabled=True),
            "images": st.column_config.NumberColumn("Images", disabled=True),
            "larvae": st.column_config.NumberColumn("Larvae", disabled=True),
            "plot": st.column_config.NumberColumn("Plot", disabled=True, format="%d"),
            "label": st.column_config.TextColumn("Label (editable)"),
            "note": st.column_config.TextColumn("Note (editable)"),
            "exclude": st.column_config.CheckboxColumn("Exclude"),
        })

    c1, c2, c3 = st.columns(3)
    if c1.button("Apply edits", key="lab_lbl_apply", type="primary"):
        for _, r in edited.iterrows():
            sid = float(r["qr_sample_id"])
            orig = r.get("genotype")
            lbl = (r.get("label") or "").strip()
            if lbl and lbl != (str(orig) if pd.notna(orig) else ""):
                s["labels"][str(sid)] = lbl
            else:
                s["labels"].pop(str(sid), None)
            note = (r.get("note") or "").strip()
            if note:
                s["notes"][str(sid)] = note
            else:
                s["notes"].pop(str(sid), None)
            if bool(r.get("exclude")):
                if str(sid) not in s["excluded"]:
                    s["excluded"].append(str(sid))
            elif str(sid) in s["excluded"]:
                s["excluded"].remove(str(sid))
        st.success(f"{len(s['labels'])} renamed, {len(s['notes'])} notes, "
                   f"{len(s['excluded'])} excluded.")
        st.rerun()
    if c2.button("Clear all label edits", key="lab_lbl_clear"):
        s["labels"], s["notes"], s["excluded"] = {}, {}, []
        st.rerun()
    c3.download_button("Download label overrides (CSV)",
                       pd.DataFrame({"qr_sample_id": list(s["labels"].keys()),
                                     "label": list(s["labels"].values())}).to_csv(index=False).encode(),
                       file_name="sample_label_overrides.csv", key="lab_lbl_dl")
    if s["labels"] or s["excluded"]:
        st.info(f"Active: {len(s['labels'])} renamed · {len(s['excluded'])} excluded.")


# --------------------------------------------------------------------- plot studio
def render_plot_studio(summary_df: pd.DataFrame, worms_df: pd.DataFrame | None) -> None:
    st.subheader("Plot studio")
    st.caption("Build any figure from the data, restyle it, and export it as SVG (vector, "
               "for figures), PNG or an interactive HTML file.")
    sources = {"Per image (summary)": summary_df}
    if worms_df is not None and not worms_df.empty:
        sources["Per larva"] = worms_df
    src = st.selectbox("Data source", list(sources), key="lab_ps_src")
    df = sources[src]
    if df is None or df.empty:
        st.info("No data.")
        return

    kind = st.selectbox("Chart type",
                        ["Scatter", "Histogram", "Box", "Violin", "Line", "Bar", "Density heatmap"],
                        key="lab_ps_kind")
    num, grp = numeric_columns(df), grouping_columns(df)
    if not num:
        st.info("No numeric columns.")
        return

    # lead with columns people actually plot, not whatever sorts first alphabetically
    PREFERRED = ["count", "area_mm2", "axis_major_mm", "skeleton_length_mm", "aspect_ratio",
                 "plant_weight_g", "larvae_per_kg", "qr_reihe", "qr_spalte", "gmm_pc1", "gmm_pc2"]
    xopts = num + grp
    pref_x = next((c for c in PREFERRED if c in xopts), xopts[0])
    pref_y = next((c for c in PREFERRED if c in num and c != pref_x), num[0] if num else None)
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("X", xopts, index=xopts.index(pref_x), key="lab_ps_x")
    yopts = ["(none)"] + num
    y = c2.selectbox("Y", yopts, index=yopts.index(pref_y) if pref_y in yopts else 0,
                     key="lab_ps_y")
    color = c3.selectbox("Colour", ["(none)"] + grp + num, key="lab_ps_color")
    facet = c4.selectbox("Facet", ["(none)"] + grp, key="lab_ps_facet")
    y = None if y == "(none)" else y
    color = None if color == "(none)" else color
    facet = None if facet == "(none)" else facet

    maxpts = st.slider("Max points plotted (subsample for speed; exports use the same view)",
                       1000, 200_000, min(MAX_POINTS_DEFAULT, max(1000, len(df))), 1000,
                       key="lab_ps_max")
    d = subsample(df, maxpts)
    s = style_controls("lab_ps")

    try:
        common = dict(color=color, facet_col=facet, template=s["template"])
        if kind == "Scatter":
            fig = px.scatter(d, x=x, y=y, render_mode="webgl", **common)
        elif kind == "Histogram":
            fig = px.histogram(d, x=x, **common)
        elif kind == "Box":
            fig = px.box(d, x=x if x in grp else None, y=y or x, **common)
        elif kind == "Violin":
            fig = px.violin(d, x=x if x in grp else None, y=y or x, box=True, **common)
        elif kind == "Line":
            fig = px.line(d.sort_values(x), x=x, y=y, **common)
        elif kind == "Bar":
            fig = px.bar(d, x=x, y=y, **common)
        else:
            fig = px.density_heatmap(d, x=x, y=y, template=s["template"], facet_col=facet)
    except Exception as exc:
        st.error(f"Could not build that chart: {exc}")
        return

    show_figure(apply_style(fig, s), "plot_studio", "lab_ps_fig")
    with st.expander("Data behind this figure"):
        cols = [c for c in {x, y, color, facet} if c]
        download_table(d[cols] if cols else d, "plot_studio_data", "lab_ps_tbl")


# --------------------------------------------------------------------------- gating
def gate_mask(df: pd.DataFrame, gate: dict) -> pd.Series:
    """Point-in-gate test, evaluated from stored coordinates (never row indices)."""
    # a gate restored from a session file may reference columns this project lacks
    if gate.get("x") not in df.columns or gate.get("y") not in df.columns:
        return pd.Series(False, index=df.index)
    x = pd.to_numeric(df[gate["x"]], errors="coerce")
    y = pd.to_numeric(df[gate["y"]], errors="coerce")
    if gate["kind"] == "rect":
        x0, x1, y0, y1 = gate["bounds"]
        return x.between(min(x0, x1), max(x0, x1)) & y.between(min(y0, y1), max(y0, y1))
    from matplotlib.path import Path as MplPath
    pts = np.column_stack([x.to_numpy(float), y.to_numpy(float)])
    ok = np.isfinite(pts).all(axis=1)
    inside = np.zeros(len(df), bool)
    if ok.any():
        inside[ok] = MplPath(np.asarray(gate["path"], float)).contains_points(pts[ok])
    return pd.Series(inside, index=df.index)


def combined_gate_mask(df: pd.DataFrame, names: list[str], how: str) -> pd.Series:
    gates = [g for g in lab_state()["gates"] if g["name"] in names]
    if not gates:
        return pd.Series(True, index=df.index)
    masks = [gate_mask(df, g) for g in gates]
    out = masks[0]
    for m in masks[1:]:
        out = (out & m) if how == "AND (intersection)" else (out | m)
    return out


def render_gating(worms_df: pd.DataFrame | None) -> None:
    st.subheader("Gating - select subpopulations in parameter space")
    st.caption("Draw a **box** or **lasso** on the point cloud (toolbar, top-right of the "
               "chart), then save the selection as a named gate. Gates are stored as "
               "coordinates, so they apply to every larva - not just the plotted subsample "
               "- and survive a session reload.")
    if worms_df is None or worms_df.empty:
        st.info("No per-larva table in this project.")
        return
    s = lab_state()
    num = numeric_columns(worms_df)
    if len(num) < 2:
        st.info("Need at least two numeric columns.")
        return

    defaults = [c for c in ("area_mm2", "axis_major_mm", "gmm_pc1", "gmm_pc2") if c in num]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", num, index=num.index(defaults[0]) if defaults else 0, key="lab_g_x")
    y = c2.selectbox("Y", num,
                     index=num.index(defaults[1]) if len(defaults) > 1 else min(1, len(num) - 1),
                     key="lab_g_y")
    grp = grouping_columns(worms_df)
    color = c3.selectbox("Colour", ["(none)"] + grp, key="lab_g_color")
    maxpts = st.slider("Max points shown", 2000, 120_000,
                       min(40_000, len(worms_df)), 2000, key="lab_g_max")

    d = subsample(worms_df, maxpts).reset_index(drop=True)
    fig = px.scatter(d, x=x, y=y, color=None if color == "(none)" else color,
                     render_mode="webgl", template="plotly_white", opacity=0.55)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=620, dragmode="lasso",
                      title=f"{len(d):,} of {len(worms_df):,} larvae shown")

    ev = st.plotly_chart(fig, use_container_width=True, key="lab_gate_scatter",
                         on_select="rerun",
                         config={"displaylogo": False,
                                 "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                                 "toImageButtonOptions": {"format": "svg"}})

    sel = (ev or {}).get("selection", {}) if isinstance(ev, dict) else {}
    pts = sel.get("points", []) or []
    st.write(f"**{len(pts):,}** points currently selected on the chart.")

    name = st.text_input("Gate name", f"gate_{len(s['gates']) + 1}", key="lab_g_name")
    c4, c5 = st.columns(2)
    if c4.button("Save selection as gate", key="lab_g_save", type="primary", disabled=not pts):
        gate = None
        # Prefer the exact drawn geometry when Streamlit reports it ...
        for box in (sel.get("box") or []):
            gate = {"kind": "rect",
                    "bounds": [box["x"][0], box["x"][1], box["y"][0], box["y"][1]]}
        for lasso in (sel.get("lasso") or []):
            gate = {"kind": "poly",
                    "path": list(zip(lasso["x"], lasso["y"]))}
        if gate is None:                       # ... otherwise wrap the selected points
            xs = [p.get("x") for p in pts if p.get("x") is not None]
            ys = [p.get("y") for p in pts if p.get("y") is not None]
            if xs and ys:
                gate = {"kind": "rect", "bounds": [min(xs), max(xs), min(ys), max(ys)]}
        if gate:
            gate.update(name=name or f"gate_{len(s['gates']) + 1}", x=x, y=y,
                        n_drawn=len(pts), created=time.strftime("%Y-%m-%d %H:%M"))
            s["gates"] = [g for g in s["gates"] if g["name"] != gate["name"]] + [gate]
            st.success(f"Saved gate '{gate['name']}' ({gate['kind']}) on {x} vs {y}.")
            st.rerun()
    if c5.button("Delete all gates", key="lab_g_clear", disabled=not s["gates"]):
        s["gates"] = []
        st.rerun()

    if not s["gates"]:
        return
    st.markdown("### Saved gates")
    rows = []
    for g in s["gates"]:
        m = gate_mask(worms_df, g)
        rows.append({"gate": g["name"], "kind": g["kind"], "x": g["x"], "y": g["y"],
                     "larvae in gate": int(m.sum()),
                     "% of all": round(100 * m.mean(), 2), "created": g.get("created", "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    names = st.multiselect("Combine gates", [g["name"] for g in s["gates"]],
                           default=[s["gates"][-1]["name"]], key="lab_g_pick")
    how = st.radio("Combine with", ["OR (union)", "AND (intersection)"],
                   horizontal=True, key="lab_g_how")
    if names:
        m = combined_gate_mask(worms_df, names, how)
        sub = worms_df[m]
        st.success(f"**{len(sub):,}** larvae in the combined gate "
                   f"({100 * m.mean():.2f}% of {len(worms_df):,}).")
        if not sub.empty:
            stat_cols = [c for c in ("area_mm2", "axis_major_mm", "skeleton_length_mm",
                                     "aspect_ratio", "solidity") if c in sub.columns]
            if stat_cols:
                comp = pd.DataFrame({
                    "in gate": sub[stat_cols].median(numeric_only=True),
                    "all larvae": worms_df[stat_cols].median(numeric_only=True)})
                comp["ratio"] = (comp["in gate"] / comp["all larvae"]).round(3)
                st.dataframe(comp.round(3), use_container_width=True)
            if "qr_sample_id" in sub.columns:
                per = (sub.groupby("qr_sample_id").size().rename("larvae_in_gate")
                       .reset_index().sort_values("larvae_in_gate", ascending=False))
                with st.expander(f"Per-sample counts inside the gate ({len(per)} samples)"):
                    st.dataframe(per, use_container_width=True, hide_index=True)
                    download_table(per, "gate_per_sample", "lab_g_per")
            download_table(sub, "gated_larvae", "lab_g_dl")


# -------------------------------------------------------------------------- GMM lab
def _diag_gmm(X: np.ndarray, k: int, seed: int = 7, n_init: int = 4, iters: int = 150):
    """Diagonal-covariance EM - same model the deployed size classes use."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        if init == 0:
            centers = np.column_stack([np.quantile(X[:, j], np.linspace(.12, .88, k))
                                       for j in range(d)])
        else:
            centers = X[rng.choice(n, size=k, replace=False)].copy()
        var = np.tile(X.var(axis=0) + 1e-6, (k, 1))
        w = np.full(k, 1.0 / k)
        ll_old = -np.inf
        for _ in range(iters):
            lp = np.stack([-0.5 * (np.sum(np.log(2 * np.pi * var[c]))
                                   + np.sum((X - centers[c]) ** 2 / var[c], axis=1))
                           for c in range(k)], axis=1) + np.log(np.maximum(w, 1e-12))
            mx = lp.max(axis=1, keepdims=True)
            lse = (mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True))).ravel()
            resp = np.exp(lp - lse[:, None])
            ll = float(lse.sum())
            nk = np.maximum(resp.sum(axis=0), 1e-9)
            w = nk / n
            centers = (resp.T @ X) / nk[:, None]
            for c in range(k):
                dif = X - centers[c]
                var[c] = (resp[:, c][:, None] * dif * dif).sum(axis=0) / nk[c] + 1e-6
            if abs(ll - ll_old) < 1e-5 * max(1.0, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, w.copy(), centers.copy(), var.copy(), resp.copy())
    ll, w, centers, var, resp = best
    return resp.argmax(axis=1), resp, {"loglik": ll, "weights": w, "means": centers, "vars": var}


def render_gmm_lab(worms_df: pd.DataFrame | None, gmm_info: dict | None) -> None:
    st.subheader("GMM lab")
    st.caption("Refit the size mixture on any feature set and any number of classes, view it "
               "in 3-D, and export per-larva assignments. The deployed 2-class result stays "
               "untouched - this is a sandbox.")
    if worms_df is None or worms_df.empty:
        st.info("No per-larva table in this project.")
        return
    num = numeric_columns(worms_df)
    pcs = [c for c in ("gmm_pc1", "gmm_pc2", "gmm_pc3") if c in num]

    default_feats = [c for c in ("area_mm2", "axis_major_mm", "skeleton_length_mm",
                                 "aspect_ratio", "solidity", "mean_gray") if c in num] or num[:4]
    feats = st.multiselect("Features", num, default=default_feats, key="lab_gmm_feats")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Classes (k)", 1, 8, 2, key="lab_gmm_k")
    maxpts = c2.slider("Max larvae used for the fit", 2000, 120_000,
                       min(40_000, len(worms_df)), 2000, key="lab_gmm_max")
    only_reliable = c3.checkbox("Only size-reliable larvae", value="size_reliable" in worms_df.columns,
                                key="lab_gmm_rel",
                                help="Excludes merged head-to-tail clumps and rim artifacts, "
                                     "which otherwise inflate the large-size class.")
    if len(feats) < 2:
        st.info("Pick at least two features.")
        return

    base = worms_df
    if only_reliable and "size_reliable" in base.columns:
        base = base[base["size_reliable"].astype(bool)]
    d = subsample(base, maxpts).copy()
    X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d[X.notna().all(axis=1)]
    X = X.loc[d.index].to_numpy(float)
    if len(X) < 50:
        st.info("Not enough complete rows.")
        return
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))

    if st.button("Fit GMM", key="lab_gmm_fit", type="primary"):
        labels, resp, model = _diag_gmm(Z, k)
        # order classes by the first feature so "Class 1" is always the smallest
        order = np.argsort([d[feats[0]].to_numpy()[labels == c].mean() if (labels == c).any()
                            else np.inf for c in range(k)])
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[l] for l in labels])
        st.session_state["lab_gmm_res"] = {
            # index-aligned so a stale fit can never be pasted onto different rows
            "labels": pd.Series(labels, index=d.index),
            "conf": pd.Series(resp.max(axis=1), index=d.index),
            "model": model, "feats": list(feats), "k": k, "project_rows": len(worms_df)}
    res = st.session_state.get("lab_gmm_res")
    if not res:
        st.info("Press **Fit GMM** to run the mixture on the current selection.")
        return
    idx = d.index.intersection(res["labels"].index)
    if len(idx) < 0.5 * len(d):
        st.warning("The stored fit does not match the current data or settings - press "
                   "**Fit GMM** again.")
        return
    d = d.loc[idx]
    d["gmm_class"] = res["labels"].loc[idx].map(lambda i: f"Class {i + 1}")
    d["gmm_confidence"] = res["conf"].loc[idx]

    # model selection sweep (opt-in: it refits the mixture eight times)
    with st.expander("How many classes? (BIC / AIC sweep)"):
        if st.button("Run k = 1..8 sweep", key="lab_gmm_sweep"):
            rows = []
            for kk in range(1, 9):
                _l, _r, m = _diag_gmm(Z, kk, seed=11 + kk, n_init=2, iters=80)
                p = (kk - 1) + 2 * kk * Z.shape[1]
                rows.append({"k": kk, "params": p,
                             "BIC": p * np.log(len(Z)) - 2 * m["loglik"],
                             "AIC": 2 * p - 2 * m["loglik"]})
            st.session_state["lab_gmm_sweep_res"] = pd.DataFrame(rows)
        ms = st.session_state.get("lab_gmm_sweep_res")
        if ms is not None:
            ms = ms.copy()
            ms["dBIC"] = ms.BIC - ms.BIC.min()
            f = go.Figure()
            f.add_scatter(x=ms.k, y=ms["dBIC"], mode="lines+markers", name="dBIC")
            f.update_layout(template="plotly_white", height=340,
                            xaxis_title="classes (k)", yaxis_title="dBIC (lower = better)")
            show_figure(f, "gmm_model_selection", "lab_gmm_ms")
            st.dataframe(ms.round(1), use_container_width=True, hide_index=True)

    counts = d["gmm_class"].value_counts().sort_index()
    st.write("**Class sizes**: " + " · ".join(f"{c}: {n:,} ({100*n/len(d):.1f}%)"
                                              for c, n in counts.items()))

    view = st.radio("View", ["3D", "2D"], horizontal=True, key="lab_gmm_view")
    axis_pool = (pcs + res["feats"]) if pcs else res["feats"]
    if view == "3D":
        if len(axis_pool) < 3:
            st.info("Need three numeric axes for the 3-D view.")
            return
        c1, c2, c3 = st.columns(3)
        ax = c1.selectbox("X", axis_pool, index=0, key="lab_gmm_3x")
        ay = c2.selectbox("Y", axis_pool, index=1, key="lab_gmm_3y")
        az = c3.selectbox("Z", axis_pool, index=2, key="lab_gmm_3z")
        fig = px.scatter_3d(d, x=ax, y=ay, z=az, color="gmm_class",
                            opacity=0.6, template="plotly_white")
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(height=740, legend_title="GMM class")
        show_figure(fig, "gmm_3d", "lab_gmm_fig3d")
    else:
        c1, c2 = st.columns(2)
        ax = c1.selectbox("X", axis_pool, index=0, key="lab_gmm_2x")
        ay = c2.selectbox("Y", axis_pool, index=min(1, len(axis_pool) - 1), key="lab_gmm_2y")
        fig = px.scatter(d, x=ax, y=ay, color="gmm_class", render_mode="webgl",
                         opacity=0.6, template="plotly_white")
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=640)
        show_figure(fig, "gmm_2d", "lab_gmm_fig2d")

    prof_feats = [c for c in res["feats"] if c in d.columns]
    prof = d.groupby("gmm_class")[prof_feats].median().round(3)
    st.markdown("**Class profiles (median)**")
    st.dataframe(prof, use_container_width=True)
    keep = [c for c in ("original_filename", "worm_id", "qr_sample_id", "gmm_class",
                        "gmm_confidence", *prof_feats) if c in d.columns]
    download_table(d[keep], "gmm_assignments", "lab_gmm_dl")


# ----------------------------------------------------------------------- statistics
def render_flex_stats(summary_df: pd.DataFrame, worms_df: pd.DataFrame | None) -> None:
    st.subheader("Statistics")
    st.caption("Group-by summaries, group comparisons and correlation/regression at larva, "
               "image or sample level. Every table is exportable.")
    levels = {"Per image": summary_df}
    if worms_df is not None and not worms_df.empty:
        levels["Per larva"] = worms_df
    if "qr_sample_id" in summary_df.columns:
        # exclude the group key from the value columns, else reset_index() collides
        vals = [c for c in numeric_columns(summary_df) if c != "qr_sample_id"]
        if vals:
            levels["Per sample (image means)"] = (summary_df.groupby("qr_sample_id")[vals]
                                                  .mean().reset_index())
    lvl = st.selectbox("Level", list(levels), key="lab_st_lvl")
    df = levels[lvl]
    if df is None or df.empty:
        st.info("No data.")
        return

    t_sum, t_cmp, t_reg = st.tabs(["Group summary", "Compare groups", "Correlation & regression"])
    num, grp = numeric_columns(df), grouping_columns(df)

    with t_sum:
        if not grp or not num:
            st.info("Need a grouping column and a numeric column.")
        else:
            g = st.selectbox("Group by", grp, key="lab_st_g")
            vs = st.multiselect("Values", num,
                                default=[c for c in ("count", "area_mm2", "axis_major_mm")
                                         if c in num][:2] or num[:1], key="lab_st_v")
            if vs:
                out = df.groupby(g)[vs].agg(["count", "mean", "std", "median",
                                             lambda s: s.quantile(.25),
                                             lambda s: s.quantile(.75)])
                out.columns = [f"{a}_{'q25' if '<lambda_0>' in b else 'q75' if '<lambda_1>' in b else b}"
                               for a, b in out.columns]
                out = out.reset_index().round(3)
                st.dataframe(out, use_container_width=True, hide_index=True)
                download_table(out, "group_summary", "lab_st_sum")

    with t_cmp:
        if not grp or not num:
            st.info("Need a grouping column and a numeric column.")
        else:
            from scipy import stats as sps
            g = st.selectbox("Group by", grp, key="lab_st_cg")
            v = st.selectbox("Value", num, key="lab_st_cv")
            lv = df[g].dropna().unique().tolist()
            pick = st.multiselect("Groups to compare (2+)", lv, default=lv[:2], key="lab_st_cl")
            test = st.selectbox("Test", ["Kruskal-Wallis (non-parametric, 2+)",
                                         "Mann-Whitney U (non-parametric, 2)",
                                         "One-way ANOVA (parametric, 2+)",
                                         "t-test (parametric, 2)"], key="lab_st_ct")
            if len(pick) >= 2:
                samples = [pd.to_numeric(df.loc[df[g] == l, v], errors="coerce").dropna().to_numpy()
                           for l in pick]
                samples = [s for s in samples if len(s) > 1]
                if len(samples) >= 2:
                    try:
                        if test.startswith("Kruskal"):
                            stat, p = sps.kruskal(*samples)
                        elif test.startswith("Mann"):
                            stat, p = sps.mannwhitneyu(samples[0], samples[1])
                        elif test.startswith("One-way"):
                            stat, p = sps.f_oneway(*samples)
                        else:
                            stat, p = sps.ttest_ind(samples[0], samples[1], equal_var=False)
                        c1, c2 = st.columns(2)
                        c1.metric("statistic", f"{stat:.4g}")
                        c2.metric("p-value", f"{p:.3g}",
                                  delta="significant (p<0.05)" if p < 0.05 else "n.s.")
                    except Exception as exc:
                        st.warning(f"Test failed: {exc}")
                    desc = pd.DataFrame({"group": pick,
                                         "n": [len(s) for s in samples],
                                         "mean": [s.mean() for s in samples],
                                         "median": [np.median(s) for s in samples],
                                         "sd": [s.std(ddof=1) for s in samples]}).round(4)
                    st.dataframe(desc, use_container_width=True, hide_index=True)
                    sub = df[df[g].isin(pick)]
                    fig = px.box(sub, x=g, y=v, points="outliers", template="plotly_white")
                    fig.update_layout(height=480)
                    show_figure(fig, "group_comparison", "lab_st_fig")
                    download_table(desc, "group_comparison", "lab_st_cmp")

    with t_reg:
        if len(num) < 2:
            st.info("Need two numeric columns.")
        else:
            from scipy import stats as sps
            c1, c2 = st.columns(2)
            xv = c1.selectbox("X", num, key="lab_st_rx")
            yv = c2.selectbox("Y", num, index=min(1, len(num) - 1), key="lab_st_ry")
            cg = st.selectbox("Colour by", ["(none)"] + grp, key="lab_st_rc")
            dd = df[[xv, yv] + ([cg] if cg != "(none)" else [])].apply(
                lambda s: pd.to_numeric(s, errors="ignore")).dropna(subset=[xv, yv])
            xs = pd.to_numeric(dd[xv], errors="coerce") if len(dd) else pd.Series(dtype=float)
            ys = pd.to_numeric(dd[yv], errors="coerce") if len(dd) else pd.Series(dtype=float)
            ok = xs.notna() & ys.notna() if len(dd) else pd.Series(dtype=bool)
            if int(ok.sum()) < 3:
                st.info("Not enough complete pairs for a regression.")
            elif xs[ok].nunique() < 2 or ys[ok].nunique() < 2:
                st.info(f"'{xv}' or '{yv}' is constant over the current selection - "
                        "pick columns that vary.")
            else:
                lr = sps.linregress(xs[ok], ys[ok])
                rho, prho = sps.spearmanr(xs[ok], ys[ok])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pearson r", f"{lr.rvalue:.3f}")
                c2.metric("p (linear)", f"{lr.pvalue:.3g}")
                c3.metric("Spearman ρ", f"{rho:.3f}")
                c4.metric("slope", f"{lr.slope:.4g}")
                fig = px.scatter(dd, x=xv, y=yv, color=None if cg == "(none)" else cg,
                                 render_mode="webgl", template="plotly_white", opacity=0.6)
                # fit line drawn from the linregress result (avoids a statsmodels dependency)
                xl = np.array([xs[ok].min(), xs[ok].max()], float)
                fig.add_scatter(x=xl, y=lr.intercept + lr.slope * xl, mode="lines",
                                name="OLS fit", line=dict(color="black", width=2))
                fig.update_layout(height=560,
                                  title=f"{yv} = {lr.slope:.4g}·{xv} + {lr.intercept:.4g}  "
                                        f"(R²={lr.rvalue**2:.3f}, n={int(ok.sum()):,})")
                show_figure(fig, "regression", "lab_st_regfig")
                download_table(dd, "regression_data", "lab_st_reg")


# --------------------------------------------------------------------- session file
SNAPSHOT_PREFIXES = ("lab_",)


def render_session_io(project: str) -> None:
    st.subheader("Session file")
    st.caption("Download everything you have set up here - gates, sample labels, plot and "
               "analysis settings - as a single JSON file, then upload it later (or send it "
               "to a colleague) to restore the exact same view.")
    s = lab_state()
    settings = {k: v for k, v in st.session_state.items()
                if isinstance(k, str) and k.startswith(SNAPSHOT_PREFIXES)
                and isinstance(v, (str, int, float, bool, list, type(None)))}
    blob = {"version": SESSION_VERSION, "saved_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project": project, "gates": s["gates"], "labels": s["labels"],
            "notes": s["notes"], "excluded": s["excluded"], "settings": settings}
    c1, c2 = st.columns(2)
    c1.download_button("Download session file",
                       json.dumps(blob, indent=1, default=str).encode("utf-8"),
                       file_name=f"larvae_session_{project}_{time.strftime('%Y%m%d_%H%M')}.json",
                       mime="application/json", key="lab_sess_dl", type="primary")
    c1.caption(f"{len(s['gates'])} gates · {len(s['labels'])} labels · "
               f"{len(s['excluded'])} excluded · {len(settings)} settings")

    up = c2.file_uploader("Restore a session file", type=["json"], key="lab_sess_up")
    if up is not None and c2.button("Restore", key="lab_sess_restore"):
        try:
            data = json.load(up)
        except Exception as exc:
            st.error(f"Not a valid session file: {exc}")
            return
        if data.get("project") and data["project"] != project:
            st.warning(f"This file was saved for project '{data['project']}' - restoring it "
                       f"onto '{project}'. Gates referring to missing columns are ignored.")
        s["gates"] = data.get("gates", [])
        s["labels"] = data.get("labels", {})
        s["notes"] = data.get("notes", {})
        s["excluded"] = data.get("excluded", [])
        for k, v in (data.get("settings") or {}).items():
            if isinstance(k, str) and k.startswith(SNAPSHOT_PREFIXES):
                st.session_state[k] = v
        st.success(f"Restored {len(s['gates'])} gates and {len(s['labels'])} label edits "
                   f"(saved {data.get('saved_utc', '?')}).")
        st.rerun()

    if s["gates"]:
        with st.expander("Gate definitions (JSON)"):
            st.code(json.dumps(s["gates"], indent=1, default=str), language="json")


# ------------------------------------------------------------------------ entrypoint
def render_lab(summary_df: pd.DataFrame, worms_df: pd.DataFrame | None,
               parcel_df: pd.DataFrame | None, gmm_info: dict | None, project: str) -> None:
    """Render the whole lab as one tab group."""
    t_plot, t_gate, t_gmm, t_stats, t_lbl, t_sess = st.tabs(
        ["Plot studio", "Gating", "GMM lab", "Statistics", "Sample labels", "Session file"])
    with t_plot:
        render_plot_studio(summary_df, worms_df)
    with t_gate:
        render_gating(worms_df)
    with t_gmm:
        render_gmm_lab(worms_df, gmm_info)
    with t_stats:
        render_flex_stats(summary_df, worms_df)
    with t_lbl:
        render_label_editor(summary_df, worms_df, parcel_df)
    with t_sess:
        render_session_io(project)
