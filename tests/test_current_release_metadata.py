"""Current release metadata is distinct from historical paper/ evidence."""
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOI = '10.5281/zenodo.22664869'
TITLE = 'A Paired Evaluation of Clean Accuracy and Corruption Sensitivity in Quantized Object Detection'


def test_current_metadata_matches_active_manuscript():
    cff = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
    zenodo = json.loads((ROOT / '.zenodo.json').read_text())
    main = (ROOT / 'submission_package/source/main.tex').read_text()
    assert TITLE in main
    assert ' '.join(cff['title'].split()) == zenodo['title'] == TITLE + ': Reproducibility Package'
    assert cff['doi'] == DOI
    assert cff['version'] == zenodo['version'] == '2.2.0'
    assert DOI in main and DOI in (ROOT / 'submission_package/source/supplement.tex').read_text()
    assert len(cff['authors']) == len(zenodo['creators']) == 6
    assert not any('claude' in json.dumps(p).lower() for p in cff['authors'] + zenodo['creators'])


def test_data_license_is_not_silently_replaced_by_mit():
    readme = (ROOT / 'submission_package/REVIEWER_EVIDENCE_README.md').read_text()
    assert 'Attribution-NonCommercial-ShareAlike 3.0' in readme
    assert 'data/annotations.json' in readme
    assert 'Geiger' in readme and '1,197-image' in readme
    assert 'does **not** apply to third-party' in readme


def test_tokens_and_large_release_archives_are_ignored():
    names = ['git_token.txt', 'zenodo_key.txt', 'submission_package/CVIU_Reviewer_Evidence.zip']
    result = subprocess.run(['git', 'check-ignore', '--no-index', *names], cwd=ROOT,
                            text=True, capture_output=True, check=True)
    assert set(result.stdout.splitlines()) == set(names)


def test_readme_points_to_current_not_historical_editable_sources():
    readme = (ROOT / 'README.md').read_text()
    assert DOI in readme and 'Version **2.2.0**' in readme
    assert 'not the current editable manuscript' in readme
    assert 'original-source clean' in readme
