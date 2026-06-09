# Streamlit app update: GMM larval size classes and plant-weight normalization

This version replaces the former lightweight parcel KMeans clustering tab with the multivariate larval GMM analysis developed for the manuscript.

## Main additions

- Loads `worms.parquet` when available.
- Computes larval physical features using the default 0.14 mm/px scale.
- Fits a two-component diagonal Gaussian mixture model in standardized multivariate larval feature space.
- Aligns classes as:
  - `GMM Size 1`: lower-size morphometric class
  - `GMM Size 2`: higher-size morphometric class
- Adds larval posterior probabilities:
  - `gmm_size1_probability`
  - `gmm_size2_probability`
- Aggregates GMM class counts to image/sample level:
  - `gmm_total_count`
  - `gmm_size1_count`
  - `gmm_size2_count`
  - `gmm_size2_fraction`
  - `gmm_mean_size2_probability`
- Adds plant-weight-normalized metrics where `plant_weight_kg` is available:
  - `gmm_total_count_per_kg_plant_weight`
  - `gmm_size1_count_per_kg_plant_weight`
  - `gmm_size2_count_per_kg_plant_weight`
- Keeps absolute counts as separate columns.
- Adds a downloadable current-data GMM infographic report from inside the Streamlit app.

## Dependencies added

- `matplotlib`
- `reportlab`

The GMM/PCA implementation itself uses NumPy only, so scikit-learn is not required.
