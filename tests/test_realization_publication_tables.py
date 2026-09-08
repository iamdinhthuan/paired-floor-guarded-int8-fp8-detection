"""Area-based and height-based realization endpoints must never be pooled."""
import json

import pytest

from build_cviu_revision_artifacts import realization_tables


def report_fixture(tmp_path):
    cells = [dict(dataset=dataset, corruption=corruption, severity=severity,
                  realization_seed=seed, delta_e=.01,
                  delta_psi={'voc': .01, 'kitti': .03, 'tt100k': .09}[dataset])
             for seed in (1, 2, 3) for dataset in ('voc', 'kitti', 'tt100k')
             for corruption in ('gaussian_noise', 'motion_blur', 'fog', 'jpeg')
             for severity in (1, 3, 5)]
    report = dict(realization_cells=cells, conditions=[{} for _ in range(36)])
    source = tmp_path/'outputs/reports/corruption_realization_analysis_v1.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(report))
    return source, report


def test_seed_table_separates_area_and_height_and_writes_only_requested_table(tmp_path):
    source, _ = report_fixture(tmp_path)
    before = source.read_bytes()
    output = tmp_path/'tables'
    realization_tables(tmp_path, output, seed_only=True)
    table = (output/'corruption_realization_seed_summary.tex').read_text()
    assert 'Area' in table and 'Height' in table
    assert '1 & 36 & +1.00 & +2.00 & +9.00' in table
    assert '+2.00; SD 0.00' in table and '+9.00; SD 0.00' in table
    assert '+4.33' not in table  # The forbidden mixed-endpoint mean.
    assert list(output.iterdir()) == [output/'corruption_realization_seed_summary.tex']
    assert source.read_bytes() == before


@pytest.mark.parametrize('mutation', ['dataset', 'duplicate', 'seed', 'nonfinite'])
def test_seed_table_rejects_malformed_realization_scope(tmp_path, mutation):
    source, report = report_fixture(tmp_path)
    if mutation == 'dataset': report['realization_cells'][0]['dataset'] = 'coco'
    if mutation == 'duplicate': report['realization_cells'][0] = report['realization_cells'][1]
    if mutation == 'seed': report['realization_cells'][0]['realization_seed'] = 4
    if mutation == 'nonfinite': report['realization_cells'][0]['delta_psi'] = float('nan')
    source.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        realization_tables(tmp_path, tmp_path/'tables', seed_only=True)
