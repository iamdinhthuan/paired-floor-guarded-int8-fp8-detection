"""The native graphical abstract must distinguish recorded clean controls."""
from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from build_paper_artifacts import generate_method_graphical_abstract


def test_graphical_abstract_renders_recorded_control_semantics_and_native_formats(tmp_path, monkeypatch):
    # Keep the real rendered figure alive solely to inspect its actual text artists.
    close = plt.close
    monkeypatch.setattr(plt, 'close', lambda *_: None)
    try:
        generate_method_graphical_abstract(tmp_path)
        figure = plt.gcf()
        text = '\n'.join(artist.get_text() for artist in figure.texts + figure.axes[0].texts)
        assert 'exploratory JPEG-95 / final-holdout original' in text
        assert 'recorded clean control' in text
        assert r'_{clean}' in text and r'_{J95}' not in text
        assert 'shared terminal JPEG-95 codec' not in text and 'J95 clean' not in text
        assert 'Identical bytes across formats in each row;' in text
        assert 'Gap contraction near a shared accuracy floor\nis not retained robustness.' in text
        assert tuple(figure.get_size_inches()) == (10., 4.)
        with Image.open(tmp_path / 'graphical_abstract.png') as png, Image.open(tmp_path / 'graphical_abstract.tif') as tif:
            assert png.size == tif.size == (3000, 1200)
            assert tif.mode == 'RGB' and tif.info['dpi'] == (300., 300.)
            assert png.convert('RGB').tobytes() == tif.tobytes()
        assert (tmp_path / 'graphical_abstract.pdf').read_bytes().startswith(b'%PDF-')
    finally:
        close('all')


def test_v4_evidence_includes_native_graphical_generator_and_test():
    import pytest
    from build_reviewer_evidence import v4_selected_files
    root = Path(__file__).resolve().parents[1]
    if not (root / 'outputs/analysis/cviu_v4/clean_control/complete.json').is_file():
        pytest.skip('completed V4 evidence unavailable')
    inventory = root / 'EVIDENCE_SHA256.json'
    names = set(json.loads(inventory.read_text())) if inventory.exists() else v4_selected_files(root)
    assert {'analysis/build_paper_artifacts.py', 'tests/test_v4_graphical_abstract.py'} <= names
