# Larvae QR Grid Explorer

Flat Streamlit app for GitHub / Streamlit Community Cloud.

Required files in the repository root:

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
