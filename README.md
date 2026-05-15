# Larvae QR Grid Explorer

Minimal flat Streamlit app for GitHub/Streamlit Community Cloud.

Files required in the repository root:

```
streamlit_app.py
requirements.txt
image_summary.parquet
images.parquet
worms.parquet
manifest.json
```

The app automatically repairs QR metadata from `qr_text` or `qr_raw`, including QR strings with optional suffixes after the numeric sample ID, for example `Plot72_Spalte2_Reihe12_R4S_197_S`.

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
