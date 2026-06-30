# Deploying the multi-project Larvae Explorer to GitHub + Streamlit Cloud

The app now hosts **two datasets** with a global selector in the sidebar:

- **Asendorf — QR-linked** (oilseed rape; plant-weight + genotype features)
- **Malchow — text-labelled** (OCR field/x/y/R4S; no plant weight)

Pick the project from the **"Project / Dataset"** selector at the top of the sidebar.

## What to upload to GitHub

Upload **everything in this `streamlit_upload/` folder**, preserving the structure:

```
<your-repo>/
├── streamlit_app.py            # the app (entry point)
├── requirements.txt            # dependencies
├── README.md
├── UPLOAD_INSTRUCTIONS.md      # this file
├── data_asendorf/              # project 1 data
│   ├── image_summary.parquet   (+ .csv)
│   ├── images.parquet          (+ .csv)
│   ├── worms.parquet
│   ├── parcel_metadata.csv     # plant weights
│   ├── genotype_names.csv      # optional R4S -> name lookup
│   └── manifest.json
└── data_malchow/               # project 2 data
    ├── image_summary.parquet   (+ .csv)
    ├── worms.parquet
    └── manifest.json
```

**Minimum required per project:** `image_summary.parquet` (or `.csv`) and
`worms.parquet`. Everything else is optional and the app degrades gracefully
(e.g. Malchow has no `parcel_metadata.csv`, so plant-weight views are simply
empty for it).

### File sizes
`worms.parquet` is the large file (Asendorf ~7 MB, Malchow ~3 MB) — both are well
under GitHub's 100 MB limit, so a normal `git add` works (no Git LFS needed).

## Steps

1. Create a new GitHub repo (or a subfolder of one).
2. Copy the contents of `streamlit_upload/` into the repo root and push:
   ```bash
   git add streamlit_app.py requirements.txt README.md UPLOAD_INSTRUCTIONS.md data_asendorf data_malchow
   git commit -m "Multi-project larvae explorer (Asendorf + Malchow)"
   git push
   ```
3. On https://share.streamlit.io → **New app** → pick the repo/branch and set
   **Main file path** to `streamlit_app.py`. Deploy.

The app auto-discovers any `data_*` project folder listed in `PROJECTS` inside
`streamlit_app.py`. To add another dataset later, drop a new `data_<name>/`
folder with the same files and add an entry to the `PROJECTS` dict.

## Adding a new project (for reference)
`PROJECTS` in `streamlit_app.py` maps a key → `{label, dir, title, axes, has_weight}`.
The `axes` dict relabels the grid fields (`qr_plot`/`qr_spalte`/`qr_reihe`) — e.g.
Malchow shows them as **Feld / x / y**. New data must reuse those `qr_*` column
names (map your own fields onto them, as `build_streamlit_projects.py` does).

## Notes
- **Malchow measurements are in pixels** (no QR scale bar yet), so its GMM size
  classes are *operational* (PCA/standardised), not millimetre instars.
- **GMM "number of classes":** the clustering tab fits k = 1..12 components in
  PC1–PC3 space and reports the **BIC elbow** (suggested classes), the BIC and AIC
  minima, the full curve, and an interactive *"re-fit with k classes"* control.
  For both datasets the elbow is ~5, but BIC keeps drifting down — the size
  structure is a continuum, so the default 2-class (Size 1 / Size 2) split is an
  operational simplification, not a sharp biological boundary.
