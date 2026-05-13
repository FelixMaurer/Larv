from worm_qr_segmenter.qr import parse_qr_text
from worm_qr_segmenter.io_utils import safe_slug


def test_parse_known_qr_format():
    parsed = parse_qr_text("Plot203_Spalte4_Reihe23_R4S_448_")
    assert parsed["format"] == "plot_spalte_reihe_condition_sample"
    assert parsed["plot"] == 203
    assert parsed["spalte"] == 4
    assert parsed["reihe"] == 23
    assert parsed["condition"] == "R4S"
    assert parsed["sample_id"] == 448


def test_safe_slug():
    assert safe_slug("Plot203_Spalte4_Reihe23_R4S_448_") == "Plot203-Spalte4-Reihe23-R4S-448"
