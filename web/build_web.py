#!/usr/bin/env python3
"""Build the static, server-free version of the Larvae Explorer (stlite: Streamlit in the browser via Pyodide).

    python web/build_web.py            -> web/dist/

The result is plain files (index.html, the two .py files, the data folders, an icon), so any static web host
can serve it; Python runs in the visitor's browser. Hosted at https://felixmilanmaurer.com/larvae/ behind
Basic Auth (web/htaccess, see web/README.md). Re-run after every change to the app or the data, then upload.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = WEB / "dist"
STLITE = "1.9.2"            # @stlite/browser on jsdelivr (Streamlit 1.62, Pyodide / Python 3.13)
REQUIREMENTS = ["plotly", "pyarrow", "scipy", "matplotlib", "reportlab"]   # reportlab: PDF report
CODE = ["streamlit_app.py", "explorer.py"]
# Parquet tables plus the small metadata files; the CSV copies of the parquet tables are not needed.
DATA = {
    "data_asendorf": ["image_summary.parquet", "images.parquet", "worms.parquet", "parcel_metadata.csv",
                      "genotype_names.csv", "manifest.json", "gmm_info.json"],
    "data_malchow": ["image_summary.parquet", "worms.parquet", "parcel_metadata.csv", "genotype_names.csv",
                     "manifest.json", "gmm_info.json", "trap_harvest_manual_larvae.csv"],
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Larvae Explorer</title>
<link rel="icon" href="icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.css">
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; background: #0e1117; }
  #boot { font: 15px/1.5 system-ui, sans-serif; color: #c9d1d9; padding: 40px 16px; text-align: center; }
</style>
</head>
<body>
<div id="root"><div id="boot">Loading the Larvae Explorer. Python starts in your browser; the first visit
downloads about 60&nbsp;MB and takes a minute, later visits are faster.</div></div>
<script type="module">
import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@__STLITE__/build/stlite.js";
const v = "__VERSION__";
const files = __FILES__;
const withVersion = Object.fromEntries(Object.entries(files).map(([k, u]) => [k, { url: u + "?v=" + v }]));
mount({ requirements: __REQUIREMENTS__, entrypoint: "streamlit_app.py", files: withVersion,
        streamlitConfig: { "client.toolbarMode": "viewer" } },
      document.getElementById("root"));
</script>
</body>
</html>
"""


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    files = {}
    for name in CODE:
        shutil.copy2(ROOT / name, DIST / name)
        files[name] = f"./{name}"
    for folder, names in DATA.items():
        (DIST / folder).mkdir()
        for name in names:
            src = ROOT / folder / name
            if src.exists():
                shutil.copy2(src, DIST / folder / name)
                files[f"{folder}/{name}"] = f"./{folder}/{name}"
    shutil.copy2(WEB / "icon.svg", DIST / "icon.svg")
    # cache-busting token: changes whenever code or data change
    stamp = max(int((DIST / k).stat().st_mtime) for k in files)
    page = (PAGE.replace("__STLITE__", STLITE).replace("__VERSION__", str(stamp))
            .replace("__FILES__", json.dumps(files, indent=1)).replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS)))
    (DIST / "index.html").write_text(page, encoding="utf-8")
    shutil.copy2(WEB / "htaccess", DIST / ".htaccess")
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) / 1e6
    print(f"web/dist: {len(files)} app files, {size:.1f} MB")


if __name__ == "__main__":
    main()
