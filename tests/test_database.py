from pathlib import Path

import pandas as pd

from worm_qr_segmenter.database import make_image_summary, read_segmentation_tables


def test_make_image_summary_keeps_missing_grid_positions(tmp_path: Path):
    images = pd.DataFrame(
        [
            {
                "original_filename": "a.png",
                "output_basename": "a",
                "qr_plot": 203,
                "qr_spalte": 4,
                "qr_reihe": 23,
                "qr_condition": "R4S",
                "qr_sample_id": 448,
                "n_kept_worms": 2,
            },
            {
                "original_filename": "b.png",
                "output_basename": "b",
                "qr_plot": 203,
                "qr_spalte": 5,
                "qr_reihe": 23,
                "qr_condition": "R4S",
                "qr_sample_id": 449,
                "n_kept_worms": 0,
            },
        ]
    )
    worms = pd.DataFrame(
        [
            {"output_basename": "a", "worm_id": 1, "skeleton_length_px": 10.0, "area_px": 40},
            {"output_basename": "a", "worm_id": 2, "skeleton_length_px": 20.0, "area_px": 80},
        ]
    )

    summary = make_image_summary(images, worms)

    assert list(summary["count"]) == [2, 0]
    assert summary.loc[summary["output_basename"] == "a", "mean_skeleton_length_px"].iloc[0] == 15.0
    assert pd.isna(summary.loc[summary["output_basename"] == "b", "mean_skeleton_length_px"].iloc[0])


def test_read_segmentation_tables_from_standard_output(tmp_path: Path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "stats").mkdir()
    pd.DataFrame(
        [
            {
                "original_filename": "a.png",
                "output_basename": "a",
                "qr_plot": "203",
                "qr_spalte": "4",
                "qr_reihe": "23",
                "qr_condition": "R4S",
                "qr_sample_id": "448",
                "n_kept_worms": "1",
            }
        ]
    ).to_csv(tmp_path / "metadata" / "images_metadata.csv", index=False)
    pd.DataFrame(
        [{"output_basename": "a", "worm_id": 1, "skeleton_length_px": 11.0}]
    ).to_csv(tmp_path / "stats" / "all_worms.csv", index=False)

    images, worms = read_segmentation_tables(tmp_path)

    assert images["qr_spalte"].iloc[0] == 4
    assert "qr_spalte" in worms.columns
    assert worms["qr_spalte"].iloc[0] == 4
