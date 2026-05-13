# Larvae QR Grid Explorer

Minimal flat Streamlit app for GitHub / Streamlit Community Cloud.

## Required repository layout

Put these files in the root of the GitHub repository:

```text
streamlit_app.py
requirements.txt
image_summary.parquet   # preferred
images.parquet          # optional
worms.parquet           # optional, kept for compatibility with existing export name
manifest.json           # optional
```

CSV fallbacks are supported if parquet files are absent:

```text
image_summary.csv
images.csv
worms.csv
```

## Deploy on Streamlit Community Cloud

1. Push the files to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Use `streamlit_app.py` as the entrypoint file.
5. Deploy.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
