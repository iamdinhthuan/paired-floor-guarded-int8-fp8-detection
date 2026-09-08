import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'analysis/package_v4_holdout_example.py'
    assert path.exists(), 'real holdout example packager missing'
    spec = importlib.util.spec_from_file_location('v4_pack', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def audit_fixture():
    rows = []
    for precision in ('int8-entropy', 'fp8'):
        for condition, severity in (('clean', 0), ('fog', 1)):
            rows.append({'status': 'verified', 'run_record': {'dataset': 'kitti', 'model': 'yolo11m',
                         'n_images': 1197, 'precision': precision, 'corruption': condition, 'severity': severity},
                         'metric_record': {'stats': {'AP': .4 if precision == 'int8-entropy' else .5}}})
    return {'fixed_example': {'dataset': 'kitti', 'model': 'yolo11m', 'corruption': 'fog', 'severity': 1, 'runs': rows},
            'dataset_inputs': {'kitti': {'expected_images': 1197,
                'original_control_semantics': {'control': 'original_source', 'records': 1197,
                                               'original_source_identity_records': 1197}}}}


def test_fixed_example_is_original_source_and_all_four_arms_required():
    m = module()
    audit = audit_fixture()
    selected = m.select_example(audit)
    assert set(selected) == {'int8_clean', 'fp8_clean', 'int8_corrupt', 'fp8_corrupt'}
    audit['fixed_example']['runs'].append(copy.deepcopy(audit['fixed_example']['runs'][0]))
    with pytest.raises(ValueError):
        m.select_example(audit)


@pytest.mark.parametrize('change', [
    lambda x: x['dataset_inputs']['kitti']['original_control_semantics'].update(control='JPEG95'),
    lambda x: x['fixed_example']['runs'][0]['run_record'].update(n_images=1496),
    lambda x: x['fixed_example']['runs'][0].update(status='partial'),
])
def test_packager_rejects_wrong_scope_or_incomplete_lineage(change):
    audit = audit_fixture()
    change(audit)
    with pytest.raises(ValueError):
        module().select_example(audit)


def test_bound_copy_rejects_changed_bytes_and_never_overwrites(tmp_path):
    m = module()
    source = tmp_path / 'source'
    source.write_bytes(b'original')
    out = tmp_path / 'package'
    out.mkdir()
    ref = m.bound_copy(source, out, 'data/payload', m.sha(source))
    assert (out / ref['path']).read_bytes() == b'original'
    source.write_bytes(b'mutated')
    with pytest.raises(ValueError):
        m.bound_copy(source, out, 'data/new', ref['sha256'])
    with pytest.raises(FileExistsError):
        m.bound_copy(source, out, 'data/payload', m.sha(source))
    assert (out / 'data/payload').read_bytes() == b'original'


def test_report_comparison_cannot_pass_by_reusing_expected_as_result():
    m = module()
    expected = {'point_ap': {'int8_clean': 40., 'fp8_clean': 50., 'int8_corrupt': 20., 'fp8_corrupt': 25.},
                'point_delta_e': -5., 'delta_e_interval': [-8., -5., -2.]}
    report = {'point': {key: {'all': value} for key, value in expected['point_ap'].items()},
              'intervals': {'delta_e': {'all': [-8., -5., -2.]}}}
    report['point']['delta_e'] = {'all': -5.}
    assert m.compare_report(report, expected)['status'] == 'PASS'
    report['point']['int8_clean']['all'] = 40.001
    with pytest.raises(ValueError):
        m.compare_report(report, expected)
