# Worm QR Segmenter

Batch analysis pipeline for worm images that contain an optional QR code with sample metadata.

The pipeline does five things per image:

1. loads the raw image and immediately creates the configurable working-scale image,
2. decodes the QR code before any tray ROI crop, first on the full raw image and then, if needed, on the complete working-scale image,
3. detects the tray ROI and excludes non-tray dark or grey regions,
4. segments worms only in the valid tray region,
5. writes a label image and a flat per-worm statistics table.

By default, `scale_factor = 0.5`. All segmentation masks, overlays, ROI coordinates, and worm measurements are therefore in half-scale working pixels. No physical calibration is applied yet.

## What is saved

For an input image `image_001.png` with a QR code such as

```text
Plot203_Spalte4_Reihe23_R4S_448_
```

output filenames contain a sanitized QR slug, for example

```text
labels/image-001__qr-Plot203-Spalte4-Reihe23-R4S-448__a1b2c3d4__labels.tif
overlays/image-001__qr-Plot203-Spalte4-Reihe23-R4S-448__a1b2c3d4__labeled.png
stats/image-001__qr-Plot203-Spalte4-Reihe23-R4S-448__a1b2c3d4__worms.csv
```

The original input files are never renamed. Use `--copy_originals` if you also want copies of the originals with QR metadata in the copied filename.

The output tree is:

```text
out/
  labels/                    integer TIFF label masks, 0 = background, 1..N = worm_id
  overlays/                  main labeled output images with colored worm masks and IDs
  stats/
    <image>__worms.csv       one row per worm for one image
    all_worms.csv            one row per worm across the whole batch
  metadata/
    images_metadata.csv      one row per image, including QR content and output paths
    images_metadata.jsonl    same metadata in JSON-lines form
    <image>__qr.txt          decoded QR text only, empty if no QR was readable
    <image>__qr.json         full QR result, including parser output and bbox
    run_config.json          exact run configuration
  debug/                     only created with --debug
```

Debug output includes the valid-region overlay, initial raw mask, post-exclusion raw mask, dark-worm rescue masks, final raw mask, rejected masks, rejected overlays, ROI diagnostics, QR search diagnostics and histograms. It is intentionally not written during normal production runs. For QR problems, inspect `debug/<image>__qr_search_debug_original.png` and `debug/<image>__qr_search_debug_working.png`; orange boxes are candidate crops and the green polygon is the decoded QR if found.

## Installation

Recommended in a fresh conda/mamba environment:

```bash
mamba create -n worms-qr python=3.10 -y
mamba activate worms-qr
pip install -e .[cellpose]
```

If CUDA is not detected by PyTorch on Windows/WSL, install the matching PyTorch wheel before installing/running Cellpose.

For QR-only testing without Cellpose:

```bash
pip install -e .
```

Optional QR fallback:

```bash
pip install -e .[qr-extra]
```

`pyzbar` may require the native `zbar` library on some systems, so OpenCV QR decoding is the default and primary method.

## Usage

Process a folder with the default half-scale workflow:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out
```

Process recursively:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --recursive
```

Run QR metadata extraction only:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --metadata_only
```

This is the fastest way to verify that the QR content is readable before running Cellpose. The console now prints the decoded QR text when successful. The same text is written to `metadata/<image>__qr.txt`, and parsed fields are written to `images_metadata.csv`.

Run with debug outputs:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --debug
```

Force CPU:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --no_gpu
```

Use full resolution instead of half scale:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --scale_factor 1.0 --diameter 30
```

Let Cellpose estimate the diameter:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --diameter 0
```

Tune the primary Cellpose-SAM detection thresholds:

```bash
worm-qr-segment --image image.png --out out_debug --debug \
  --diameter 10 \
  --flow_threshold 0.4 \
  --cellprob_threshold -2.0 \
  --no_dark_worm_rescue
```

`--flow_threshold` controls how strict Cellpose is about mask flow consistency. Lower values are more permissive and can keep more irregular worm masks. `--cellprob_threshold` controls how permissive the mask-probability cutoff is. Lower values can recover fainter worms, but may also increase false positives. Check `debug/<image>__raw_mask_initial_overlay.png` to judge only the primary Cellpose-SAM output before the rescue pass and morphology filtering.

Manual ROI override in working-image coordinates, meaning after `scale_factor` has been applied:

```bash
worm-qr-segment --image image.png --out out --roi 60 900 40 700
```

Lightweight smoke test without Cellpose:

```bash
worm-qr-segment --image image.png --out out --backend threshold --debug
```

The threshold backend is only for checking that the repository and output code work. Use `cellpose_sam` for real analysis.

## Dark-region exclusion

The default workflow removes the main failure mode seen in the first test output, where the QR code and other dark areas were segmented as worms.

It does this by computing a valid segmentation domain from a strongly blurred luminance image. Large dark or grey structures such as the QR card, black background, tray frame, ruler and outside-table areas are replaced by median tray background before segmentation. Small dark worms stay detectable because the valid domain is computed after strong blurring, not from the raw dark pixels directly.

Relevant parameters:

```bash
--no_dark_region_exclusion
--min_valid_background_luma 125
--valid_region_blur_px 41
--valid_region_close_px 25
--valid_region_erode_px 2
--valid_region_min_inside_fraction 0.80
```

For problematic images, first run:

```bash
worm-qr-segment --image image.png --out out_debug --debug
```

Then inspect:

```text
debug/<image>__valid_region_debug.png
debug/<image>__raw_mask_initial_overlay.png
debug/<image>__raw_mask_after_exclusion_overlay.png
debug/<image>__dark_rescue_overlay.png
debug/<image>__raw_mask_final_overlay.png
```

If true worms near the tray edge are removed, reduce the valid-region threshold or erosion, for example:

```bash
worm-qr-segment --image image.png --out out_debug --debug \
  --min_valid_background_luma 95 \
  --valid_region_erode_px 0
```

If the QR card or dark background still leaks into the segmentation region, increase the threshold:

```bash
worm-qr-segment --image image.png --out out_debug --debug \
  --min_valid_background_luma 115
```


## Missed-worm rescue pass

If a visible worm does not appear in the rejected-mask overlay, it was usually not filtered out. It was probably never returned by the primary segmentation backend. The current version therefore includes a conservative rescue pass that searches the valid tray region for dark elongated objects that overlap only weakly with the primary masks. These rescue candidates are appended to the raw label mask and then pass through the same morphology filter as all other objects.

The rescue pass is enabled by default. Disable it with:

```bash
worm-qr-segment --image image.png --out out --no_dark_worm_rescue
```

Useful rescue parameters, all in working pixels:

```bash
--dark_rescue_min_contrast 10
--dark_rescue_bg_blur_px 61
--dark_rescue_blackhat_px 31
--dark_rescue_min_area_px 18
--dark_rescue_min_skeleton_length_px 14
--dark_rescue_min_aspect_ratio 1.35
--dark_rescue_max_minor_axis_px 22
--dark_rescue_max_existing_overlap 0.20
```

If visible worms are still missed, lower the contrast threshold slightly:

```bash
worm-qr-segment --image image.png --out out_debug --debug \
  --dark_rescue_min_contrast 7
```

If small dirt or scratches are added as worms, make the rescue stricter:

```bash
worm-qr-segment --image image.png --out out_debug --debug \
  --dark_rescue_min_contrast 14 \
  --dark_rescue_min_skeleton_length_px 20 \
  --dark_rescue_min_aspect_ratio 1.8
```

In debug mode, inspect these files in this order:

```text
debug/<image>__raw_mask_initial_overlay.png
debug/<image>__raw_mask_after_exclusion_overlay.png
debug/<image>__dark_rescue_overlay.png
debug/<image>__raw_mask_final_overlay.png
debug/<image>__rejected_overlay.png
```

Interpretation: if the worm is absent from `raw_mask_initial_overlay.png`, Cellpose missed it. If it appears in `dark_rescue_overlay.png`, the rescue pass recovered it. If it appears in `raw_mask_final_overlay.png` but not in the final labeled overlay, it was rejected by the morphology filters and should appear in `rejected_overlay.png` or `rejected.csv`.

## QR detection and parsing

QR detection is done before any tray ROI crop. The code first tries the full raw image. If that fails, it tries the complete already-rescaled working image, so the QR cannot be cut off by tray detection or segmentation cropping. Both passes use OpenCV on the whole image, broad right-card fallback windows that deliberately avoid cutting the QR near the image border, likely high-contrast QR candidate crops, contrast normalization / thresholding and stronger upscaling. Disable this robust mode only for speed tests:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --metadata_only --no_qr_try_harder
```

To disable the additional complete working-image QR retry, use:

```bash
worm-qr-segment --image_dir path/to/images --out path/to/out --metadata_only --no_qr_working_image_search
```

The parser recognizes the current QR format:

```text
Plot203_Spalte4_Reihe23_R4S_448_
```

and writes these fields to `images_metadata.csv`:

```text
qr_raw
qr_format
qr_plot
qr_spalte
qr_reihe
qr_condition
qr_sample_id
```

If another QR text format appears, the raw text is still saved and a generic tokenization is written. The parser can be extended in `src/worm_qr_segmenter/qr.py` without changing the segmentation pipeline.

For each image, the QR result is saved three ways:

```text
metadata/images_metadata.csv      flat batch table with qr_text and qr_* parsed columns
metadata/<image>__qr.txt          decoded raw text only
metadata/<image>__qr.json         full detected flag, method, bbox, points and parsed fields
```

If `qr_detected` is false but the code is visible, rerun with `--debug` and inspect both `debug/<image>__qr_search_debug_original.png` and `debug/<image>__qr_search_debug_working.png`. If the QR is physically cut off at the camera frame or strongly blurred, it may still be unreadable even though the printed pattern is partly visible.

## Statistics schema

Each per-image worm CSV and `stats/all_worms.csv` contain one row per worm. The label mask pixel value equals `worm_id`.

Core columns include:

```text
original_filename
output_basename
scale_factor
coordinate_scale
qr_detected
qr_text
qr_* parsed metadata fields
worm_id
raw_label
area_px
skeleton_length_px
axis_major_px
axis_minor_px
aspect_ratio
eccentricity
solidity
perimeter_px
orientation_rad
equivalent_diameter_area_px
centroid_y_crop
centroid_x_crop
centroid_y_image
centroid_x_image
bbox_*_crop
bbox_*_image
```

All of these geometry fields are working-pixel values. With the default `--scale_factor 0.5`, they refer to the half-scale image.

## Notes on QR exclusion

The QR code is decoded before ROI cropping. If it is detected on the raw image, its spatial coordinates are mapped to working pixels. If it is detected only on the working image, those coordinates are used directly for segmentation exclusion and mapped back to raw-image coordinates for metadata. If the QR overlaps the tray crop, its pixels are replaced by median background before segmentation and any raw label overlapping the QR area is removed. If QR decoding fails, the valid-region mask still prevents the QR card from being counted as worms in the usual imaging layout.

## Parquet database and Streamlit app

After running segmentation, build a compact analysis database from the output folder:

```bash
pip install -e ".[cellpose,app]"

worm-qr-build-db \
  --segmentation_out path/to/segmentation_output \
  --db_dir data/worm_database
```

This writes both parquet tables and CSV mirrors:

```text
data/worm_database/images.parquet         one row per processed image
data/worm_database/worms.parquet          one row per worm
data/worm_database/image_summary.parquet  one row per image with count and aggregate metrics
data/worm_database/manifest.json          source/output summary
```

The `image_summary` table is the main table for grid plots. It keeps zero-worm and missing/unfinished image positions instead of inventing values. The Streamlit app simply leaves missing QR row/column positions empty.

Run the app from the repository root:

```bash
streamlit run app/streamlit_app.py
```

or specify the database explicitly:

```bash
WORM_QR_DB=data/worm_database streamlit run app/streamlit_app.py
```

The app supports:

```text
filters by plot, condition and sample_id
selectable grid axes, defaulting to qr_spalte and qr_reihe
selectable metric, including count and all numeric image-level aggregate metrics
unit display switch between working pixels and metric dimensions
interactive scale factor for conversion, defaulting to 0.14 mm/px
rotatable Plotly 3D bar plot, initialized as a top-down 2D-like view
2D heatmap
counter/value grid table
CSV export of the current grid
```

Useful metrics generated from worm-level raw measurements include:

```text
count
mean_skeleton_length_px
median_skeleton_length_px
mean_axis_major_px
mean_area_px
mean_aspect_ratio
mean_eccentricity
mean_solidity
mean_perimeter_px
```

Additional numeric columns from the segmentation metadata, such as `n_raw_masks`, `n_rejected_masks`, `valid_region_fraction` and rescue diagnostics, are also available in the app.

The app does not change the parquet database when switching units. Pixel mode shows the raw working-pixel values. Metric mode converts length-like columns ending in `_px`, such as skeleton length, major-axis length and perimeter, by `mm_per_px`; area-like columns such as `mean_area_px` are converted by `mm_per_px²`. Counts, ratios and fractions are left unchanged.

For GitHub, commit the code and small parquet database files if they are reasonably sized. Keep raw images, debug overlays and large label masks out of normal Git. Use Git LFS only if you intentionally want versioned binary outputs.
