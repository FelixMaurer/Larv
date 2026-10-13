#!/usr/bin/env python3
"""Precompute the GMM size states exactly as in the manuscript and store them for the app.

The manuscript fits the GMM with `fit_gmm` in paper_elsevier_updated_20260629/figure_scripts/
generate_figures_4_and_6.py (Asendorf: all larvae; Malchow: Malchow photos, valid and size-reliable larvae,
as in make_malchow_si_figures.py):
  13 per-larva features; larvae with all features present and inside the 0.5th to 99.5th percentile of
  every feature; standardised; PCA (6 components, for the PC scores); two-component diagonal Gaussian
  mixture (scikit-learn, n_init 3, random_state 3) on the standardised features; Size 1 = class with the
  smaller mean area.
This script repeats that procedure (same code, same seeds) and writes into data_<project>/:
  worms.parquet  gmm_size_class, gmm_size1_probability, gmm_size2_probability, gmm_pc1..3 (empty for
                 larvae outside the fit)
  gmm_info.json  the summary the app shows (same keys as the app's own compute_larva_gmm_classes), plus
                 "source". The model-selection sweep (k = 1 to 12) is fitted in PC1 to PC3 space.
Expected (manuscript): Asendorf 26,398 larvae (15,603 Size 1, 10,795 Size 2); Malchow 155,094
(66,240 / 88,854).

    python scripts/precompute_paper_gmm.py        (needs scikit-learn; the app itself does not)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["area_mm2", "skeleton_length_mm", "axis_major_mm", "axis_minor_mm", "perimeter_mm",
            "equivalent_diameter_area_mm", "aspect_ratio", "eccentricity", "solidity", "mean_skeleton_width_px",
            "mean_gray", "local_dark_contrast_gray", "candidate_classifier_score"]
MM_FALLBACK = 0.14
EXPECTED = {"asendorf": (26398, 15603, 10795), "malchow": (155094, 66240, 88854)}


def add_units(w: pd.DataFrame, img: pd.DataFrame) -> pd.DataFrame:
    """generate_figures_4_and_6.attach_scale_info + add_units."""
    w = w.copy()
    if "pixel_scale_mm_per_px_working" not in w.columns or w["pixel_scale_mm_per_px_working"].isna().all():
        s = img[["output_basename", "pixel_scale_mm_per_px_working"]].drop_duplicates()
        w = w.merge(s, on="output_basename", how="left", suffixes=("", "_img"))
        if "pixel_scale_mm_per_px_working_img" in w.columns:
            w["pixel_scale_mm_per_px_working"] = w["pixel_scale_mm_per_px_working"].fillna(
                w.pop("pixel_scale_mm_per_px_working_img"))
    sc = pd.to_numeric(w["pixel_scale_mm_per_px_working"], errors="coerce").astype(float)
    sc = sc.where(np.isfinite(sc) & (sc > 0), MM_FALLBACK)
    px = lambda c: pd.to_numeric(w[c], errors="coerce").astype(float)  # noqa: E731
    w["area_mm2"] = px("area_px") * sc ** 2
    for src, dst in (("skeleton_length_px", "skeleton_length_mm"), ("axis_major_px", "axis_major_mm"),
                     ("axis_minor_px", "axis_minor_mm"), ("perimeter_px", "perimeter_mm"),
                     ("equivalent_diameter_area_px", "equivalent_diameter_area_mm")):
        w[dst] = px(src) * sc
    w["_qr_scaled"] = np.isfinite(pd.to_numeric(w["pixel_scale_mm_per_px_working"], errors="coerce"))
    return w


def fit(w: pd.DataFrame):
    """generate_figures_4_and_6.fit_gmm (classification part)."""
    X0 = w[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    mask = X0.notna().all(axis=1)
    for c in FEATURES:
        lo, hi = np.nanpercentile(X0.loc[mask, c], [0.5, 99.5])
        mask &= X0[c].between(lo, hi)
    used = w.loc[mask]
    X = used[FEATURES].to_numpy(float)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=6, random_state=3)
    S = pca.fit_transform(Z)
    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=3, max_iter=200, random_state=3)
    lab0 = gm.fit_predict(Z)
    prob0 = gm.predict_proba(Z)
    order = list(pd.Series(used["area_mm2"].to_numpy()).groupby(lab0).mean().sort_values().index)
    lab = np.where(lab0 == order[0], 1, 2)
    prob = prob0[:, order]                       # columns: Size 1, Size 2
    return mask, Z, S, pca, lab, prob


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def process(project: str) -> None:
    d = ROOT / f"data_{project}"
    worms = pd.read_parquet(d / "worms.parquet")
    img = pd.read_parquet(d / "image_summary.parquet")
    w = add_units(worms, img)
    pool = pd.Series(True, index=w.index)
    if project == "malchow":            # as make_malchow_si_figures.py
        pool = (w["is_malchow_sample"].fillna(True).astype(bool) & w["is_valid_larva"].fillna(False).astype(bool)
                & w["size_reliable"].fillna(False).astype(bool))
    mask_pool, Z, S, pca, lab, prob = fit(w[pool])
    idx = w[pool].index[mask_pool.to_numpy()]

    out = worms.copy()
    out["gmm_size_class"] = pd.Series(np.nan, index=out.index, dtype=object)
    out.loc[idx, "gmm_size_class"] = np.where(lab == 1, "GMM Size 1", "GMM Size 2")
    for col, vals in (("gmm_size1_probability", prob[:, 0]), ("gmm_size2_probability", prob[:, 1]),
                      ("gmm_pc1", S[:, 0]), ("gmm_pc2", S[:, 1]), ("gmm_pc3", S[:, 2])):
        out[col] = np.nan
        out.loc[idx, col] = vals.astype("float32")
        out[col] = out[col].astype("float32")
    out.to_parquet(d / "worms.parquet", index=False)

    # model selection in PC1 to PC3 space (as reported in the app), k = 1 to 12
    P3 = S[:, :3]
    sel = []
    for k in range(1, 13):
        m = GaussianMixture(n_components=k, covariance_type="full", n_init=1, max_iter=200, random_state=0).fit(P3)
        sel.append({"k": k, "bic": float(m.bic(P3)), "aic": float(m.aic(P3)),
                    "loglik": float(m.score(P3) * len(P3))})
    best = int(min(sel, key=lambda r: r["bic"])["k"])
    zdf = pd.DataFrame(Z, columns=FEATURES)
    diffs = []
    for c in FEATURES:
        z1, z2 = zdf.loc[lab == 1, c].mean(), zdf.loc[lab == 2, c].mean()
        diffs.append({"feature": c, "mean_z_size1": z1, "mean_z_size2": z2, "delta_z_size2_minus_size1": z2 - z1})
    diffs.sort(key=lambda r: abs(r["delta_z_size2_minus_size1"]), reverse=True)
    area = w.loc[idx, "area_mm2"].to_numpy(float)
    evr = pca.explained_variance_ratio_
    maxp = prob.max(axis=1)
    qr = w.loc[idx, "_qr_scaled"].to_numpy(bool)
    info = {
        "available": True,
        "message": "GMM size classes as in the manuscript (scripts/precompute_paper_gmm.py)",
        "source": "manuscript fit_gmm: 13 features, 0.5 to 99.5 percentile filter, diagonal 2-component GMM",
        "features": FEATURES,
        "n_analyzed_larvae": int(len(idx)),
        "n_input_larvae": int(pool.sum()),
        "n_qr_scaled_larvae": int(qr.sum()),
        "n_fallback_scaled_larvae": int((~qr).sum()),
        "size1_count": int((lab == 1).sum()),
        "size2_count": int((lab == 2).sum()),
        "mean_area_size1_mm2": float(np.nanmean(area[lab == 1])),
        "mean_area_size2_mm2": float(np.nanmean(area[lab == 2])),
        "pca_explained_variance_ratio": [float(x) for x in evr[:5]],
        "pca_cumulative_first3": float(np.sum(evr[:3])),
        "model_selection": sel,
        "best_bic_k_pc123": best,
        "ambiguous_04_06_fraction": float(np.mean((prob[:, 1] > 0.4) & (prob[:, 1] < 0.6))),
        "intermediate_025_075_fraction": float(np.mean((prob[:, 1] > 0.25) & (prob[:, 1] < 0.75))),
        "high_confidence_090_fraction": float(np.mean(maxp > 0.90)),
        "feature_differences": diffs,
        "loadings": {FEATURES[i]: {f"PC{j + 1}": float(pca.components_[j, i]) for j in range(5)}
                     for i in range(len(FEATURES))},
    }
    (d / "gmm_info.json").write_text(json.dumps(jsonable(info), indent=1), encoding="utf-8")
    exp = EXPECTED.get(project)
    got = (info["n_analyzed_larvae"], info["size1_count"], info["size2_count"])
    print(f"{project}: analysed {got[0]:,} (Size 1 {got[1]:,}, Size 2 {got[2]:,}); manuscript {exp}; "
          f"{'MATCH' if got == exp else 'DIFFERENT'}; PC1 {100 * evr[0]:.2f}%, PC1-3 {100 * evr[:3].sum():.2f}%, "
          f"max posterior > 0.9 {100 * info['high_confidence_090_fraction']:.2f}%, "
          f"ambiguous {100 * info['ambiguous_04_06_fraction']:.2f}%")


if __name__ == "__main__":
    for p in ("asendorf", "malchow"):
        process(p)
