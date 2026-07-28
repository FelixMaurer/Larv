# Analysis lab — user guide

The **🔬 Analysis lab** tab is the interactive half of the app. The nine tabs to its
left are unchanged; nothing you do in the lab modifies the underlying data files.

---

## Plot studio
Build any figure from the data and export it for a manuscript.

* **Data source** — per image (one row per tray photo) or per larva (one row per larva).
* **Chart type** — scatter, histogram, box, violin, line, bar, density heatmap.
* **X / Y / Colour / Facet** — any column; *Facet* splits into small multiples.
* **Style & axes** — title, theme, axis labels, ranges, log axes, marker size, opacity,
  font size, figure size.
* **Export** — **SVG** (vector — use this for figures you will edit in Illustrator or
  Inkscape), **PNG** (2× resolution) or **interactive HTML**. The camera icon on the
  chart also saves SVG.
* Large tables are subsampled for responsiveness; the slider controls how many points
  are drawn, and the exported figure matches what you see.

## Gating — selecting subpopulations
1. Pick the two parameters for the point cloud (e.g. `area_mm2` vs `axis_major_mm`,
   or `gmm_pc1` vs `gmm_pc2`).
2. Use the **lasso** (default) or **box select** tool in the chart toolbar and draw
   around the population you want.
3. Give the gate a name and press **Save selection as gate**.

Gates are stored as *coordinates*, not row numbers, which means they

* apply to **every** larva, not only the subsample that was drawn on,
* survive a reload, and
* can be shared with a colleague via the session file.

Saved gates can be combined with **OR (union)** or **AND (intersection)**. For the
combined selection you get the larva count, a median comparison against all larvae,
per-sample counts, and a **CSV / Parquet export of exactly those larvae**.

## GMM lab
A sandbox for the size mixture — the deployed 2-class result is not affected.

* Choose any **features** and any number of **classes (k)**.
* *Only size-reliable larvae* (on by default) excludes merged head-to-tail clumps and
  tray-rim artifacts, which otherwise inflate the largest class.
* **Fit GMM** runs the same diagonal-covariance model used in the pipeline.
* **k = 1..8 sweep** shows ΔBIC so you can judge the number of classes.
* View the result in **3D** (rotate/zoom) or 2D, on principal components or raw features.
* Export per-larva class assignments with confidences.

## Statistics
Works at **per larva**, **per image** or **per sample** level.

* **Group summary** — n, mean, sd, median, quartiles for any value by any grouping.
* **Compare groups** — Kruskal-Wallis, Mann-Whitney U, one-way ANOVA or Welch t-test,
  with a box plot and a descriptive table.
* **Correlation & regression** — Pearson r, Spearman ρ, slope, R², p-value, and a
  scatter with the fitted line.

## Sample labels
Rename samples, add notes, or exclude them. Type in the table, then press
**Apply edits**. Changes flow through every tab and every export in the session and
are stored in the session file — the data files on disk are never touched.

## Session file
**Download session file** saves gates, label edits and all lab settings as one JSON.
Upload it later (or send it to a colleague) and press **Restore** to get the same view
back. Restoring a file saved for a different project still works; gates that reference
columns the project does not have are ignored.

---

### Notes
* Use **`size_reliable`** rather than `is_valid_larva` whenever you analyse *size*:
  merged head-to-tail larvae are counted correctly but measure as one long object,
  which inflates the large-size tail (~14 % at p99). Counts are unaffected.
* SVG export needs `kaleido` (in `requirements.txt`). Without it the download button
  disappears but the chart's camera icon still exports SVG client-side.
