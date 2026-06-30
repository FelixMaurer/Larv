# Larvae Explorer (multi-project)

Streamlit app for GitHub / Streamlit Community Cloud. It now hosts **two datasets**
selected from a sidebar "Project / Dataset" control:

- **Asendorf — QR-linked** (in `data_asendorf/`) — full plant-weight + genotype features.
- **Malchow — text-labelled / OCR** (in `data_malchow/`) — field/x/y/R4S grid; pixel units.

**See [`UPLOAD_INSTRUCTIONS.md`](UPLOAD_INSTRUCTIONS.md) for what to push to GitHub and how to deploy.**
The GMM tab now also reports the **number of size classes** (BIC elbow + AIC/BIC minima
over k=1..12, with an interactive re-fit). The notes below describe the original
single-project Asendorf data.

---

Required files per project folder (`data_asendorf/`, `data_malchow/`):

```text
streamlit_app.py
requirements.txt
image_summary.parquet   # or image_summary.csv
```

Recommended auxiliary files:

```text
images.parquet          # or images.csv
worms.parquet           # or worms.csv
manifest.json
```

The app repairs QR metadata from `qr_text` / `qr_raw`. In Res4StRes QR strings, `R4S` is the project code and the following number is the unique genotype identifier:

```text
Plot72_Spalte2_Reihe12_R4S_197_S
```

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Main tabs

- **2D map**: square heatmap over selected grid axes.
- **3D bars**: orthographic 3D bar view with adjustable height transform.
- **Counter grid**: pivot table for the selected metric.
- **Rows**: filtered image-summary table with CSV export.
- **Genotypes**: one-row-per-R4S genotype phenotypes, paired ascending rankings, selectable traits, contrasting-end highlights, and CSV export.
- **Trend analysis**: aggregate a metric along row, column, plot, or sample axis; plot trend lines and compute slope / intercept / R².
- **Clustering**: multivariate two-state larval GMM with field maps, PCA diagnostics, class profiles, and assignment export.
- **QC / missing**: shows images without readable or complete QR metadata, missing QR fields, missing grid parcels, duplicate parcel assignments, and auxiliary table inventory.

## Notes

The clustering implementation is intentionally dependency-light and uses NumPy only; no scikit-learn dependency is required.

## QR-derived physical scaling

Metric dimensions use `pixel_scale_mm_per_px_working` for each image and larval row. If that field is unavailable or invalid, the app uses the documented 0.14 mm/working-pixel fallback. The Rows, Genotypes, and GMM views report the scale source or the number of fallback-scaled observations. Area metrics use the squared per-image scale.

## Plant-weight normalization

This version includes parcel-level plant metadata in `parcel_metadata.csv`, extracted from `Asendorf plant weight and number.xlsm`.
The app joins the metadata to analyzed image rows using:

```text
qr_plot      -> parcel_plot
qr_spalte    -> parcel_spalte
qr_reihe     -> parcel_reihe
R4S genotype (`qr_sample_id`) -> `parcel_r4s`
```

The most important field is `plant_weight_kg`, derived from the source `weight` column in grams.
In the Streamlit sidebar, enable **Normalize larva counts by plant weight** to make the global `count` metric behave as larvae per kg plant weight for maps, trend analysis and clustering. Absolute counts are retained as `count_absolute`, and explicit derived columns such as `count_per_kg_plant_weight` and `count_per_plant` remain available.

The QC / missing tab contains a **Plant weights** section showing analyzed images without matched plant weights and expected parcel metadata rows without a matching analyzed image.


## Fix: plant-weight normalization table display

This version includes a safety guard for Streamlit/PyArrow dataframe rendering.
When plant-weight normalization is enabled, several count-like columns can have
similar human-readable labels. The app now keeps display column names unique,
so the Rows/QC tables no longer crash with `Duplicate column names found`.

---
## Data correction applied (2026-06-28)

The `worms` / `image_summary` tables here are **corrected** (originals in `_backup_precorrection_20260628/`).

- **610 images; total larvae 32,443 → 30,596 (−5.7%).** (2 images — IMG_0737/0738, Plot234 — added 2026-06-29, segmented with the exact production config + same corrections; the original 608 reproduce identically.) Per-image counts in `image_summary.count` are the corrected counts. **This is a central estimate with an irreducible ~±5% margin** — faint/partial larvae on the textured transparent tray, plus some Cellpose missed-detections, are genuinely ambiguous; the pipeline's over- and under-counting roughly cancel, so the true total is ~32,200. Detection accuracy was explored exhaustively (fine-tune, appearance CNN AUC 0.88, feature classifiers) and this is at the achievable floor for this data.
- **Corrections:** (1) debris/QR-fragment false positives removed via the review-router (shape classifier, P(non-larva) ≥ 0.40); (2) over-segmented "split" larvae (one larva detected as two) merged via the geometric split-pair detector (high-confidence auto-tier + VLM-confirmed pairs on the 51 reviewed trays); (3) **transparent container artifacts** — the see-through tray's triangle stamp and thin scratches detected as larvae — removed via `local_dark_contrast_mean_gray < 8 OR local_dark_contrast_gray < 3` (larvae are dark on the backlit tray; transparent features are bright like the background, and stamp arrows have a transparent interior with a dark edge that the median condition catches). Merged objects were **re-measured** from the label-mask union, so area / length / morphometrics — and the GMM size-states computed live in this app — are consistent with the new counts.
- **NOT applied:** touching-larva *merge*-splitting (its heuristic is ~15% precision and the merge error is only ~0.6%).
- **Validation:** pipeline precision **0.971** / recall **0.988** / count error **+4.4%** on a 12-tray VLM-labeled set (before correction). The transparent-FP contrast filter was tuned on a **96-detection blind VLM test set** and achieves **91% removal precision** at the chosen threshold (≈283 transparent artifacts removed, ~24 faint larvae lost).
- **Caveats:** corrected counts are a *central* estimate — the debris router is ~88% precise and the contrast filter ~91% precise, so a small number of real (faint/pale) larvae are removed (mild under-count); ambiguous high-angle splits are left uncorrected. The size distribution shifts (debris/artifacts removed are small/faint, merged objects are larger), so the GMM size-state cutoffs move accordingly.
