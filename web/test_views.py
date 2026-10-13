#!/usr/bin/env python3
"""Run every view of the app headless (streamlit.testing AppTest) and report exceptions.

    python web/test_views.py
Selects each top-level view (and every sub-view of the metadata maps, QC and Analysis lab views) for both
projects; a view passes when the script run raises no exception.
"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # the app imports explorer.py from its own folder
APP = str(ROOT / "streamlit_app.py")
MAIN = ['2D map', '3D bars', 'Grid table', 'Data table', 'Genotypes', 'Metadata', 'Trends', 'GMM size states', 'QC', 'Analysis lab', 'Report']
SUB = {"Metadata": ("view_metadata", ["2D metadata map", "3D metadata bars", "Metadata grid table",
                                          "Metadata trends"]),
       "QC": ("view_qc", ["Unreadable QR", "Missing QR fields", "Missing parcels", "Duplicate parcels",
                                    "Plant weights", "Table inventory"]),
       "Analysis lab": ("view_lab", ["Plot studio", "Gating", "GMM lab", "Statistics", "Sample labels",
                                        "Session file"])}


def run(state: dict) -> list[str]:
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return [e.value for e in at.exception]


def main() -> int:
    failures = 0
    for proj in ("asendorf", "malchow"):
        for view in MAIN:
            cases = [{"view_main": view}]
            if view in SUB:
                key, subs = SUB[view]
                cases = [{"view_main": view, key: sub} for sub in subs]
            for state in cases:
                exc = run({"project_select": proj, **state})
                label = " / ".join(state.values())
                print(f"{'FAIL' if exc else 'ok  '} {proj:9} {label}" + (f"\n      {exc[0][:400]}" if exc else ""),
                      flush=True)
                failures += bool(exc)
    print(f"{failures} failing views")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
