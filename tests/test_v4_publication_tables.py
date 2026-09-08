"""Publication tables must preserve paired covariance, scope, and source binding."""
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('build_v4_publication_tables'), 'publication generator missing'
    return importlib.import_module('build_v4_publication_tables')


PRIMARY = [f'{d}_yolo11{s}' for d in ('voc', 'kitti') for s in ('n', 'm', 'x')]
ARMS = ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')


def clean_fixture():
    blocks, points, draws, jobs = [], {}, {}, []
    for i, block in enumerate(PRIMARY + ['tt100k_yolo11n', 'tt100k_yolo11x']):
        values = [.4, .5, .3, .35] if i < 6 else [.3, .8, .3, .3]
        blocks.append(dict(block=block, group='primary' if i < 6 else 'diagnostic',
                           ap_points=dict(zip(ARMS, np.array(values)*100)),
                           S=5 if i < 6 else 50, interval=[5, 5, 5] if i < 6 else [50, 50, 50]))
        for arm, value in zip(ARMS, values):
            name = block + '__' + arm
            points[name] = value
            draws[name] = np.full(2000, value)
            jobs.append(dict(name=name, point=value, n_boot=2000, image_ids=[1, 2],
                             schedule_sha256=block.split('_')[0], annotation_sha256='a'))
    return dict(blocks=blocks, units='AP points', n_boot=2000,
                primary_macro=dict(S=5, interval=[5, 5, 5])), points, draws, jobs


def test_clean_recomputes_four_arm_shift_and_excludes_diagnostics():
    result = module().summarize_clean(*clean_fixture())
    assert result['primary_macro']['S'] == pytest.approx(5)
    assert result['primary_macro']['interval'] == pytest.approx([5, 5, 5])
    assert result['blocks'][-1]['S'] == pytest.approx(50)


def test_clean_macro_averages_common_draws_before_percentiles():
    summary, points, draws, jobs = clean_fixture()
    x = np.linspace(-.01, .01, 2000)
    for i, block in enumerate(PRIMARY):
        draws[block+'__fp8_original'] += (-1 if i % 2 else 1)*x
        summary['blocks'][i]['interval'] = [4.05, 5, 5.95]
    result = module().summarize_clean(summary, points, draws, jobs)
    assert result['primary_macro']['interval'] == pytest.approx([5, 5, 5])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'diagnostic_pool', 'wrong_arm', 'wrong_schedule', 'wrong_order', 'wrong_point', 'nonfinite'])
def test_clean_rejects_malformed_scope_pairing_and_values(mutation):
    summary, points, draws, jobs = clean_fixture()
    if mutation == 'missing': summary['blocks'].pop()
    if mutation == 'duplicate': summary['blocks'][-1] = summary['blocks'][0]
    if mutation == 'diagnostic_pool': summary['blocks'][-1]['group'] = 'primary'
    if mutation == 'wrong_arm': draws['voc_yolo11n__fp8_original'] = draws.pop('voc_yolo11n__int8_original')
    if mutation == 'wrong_schedule': jobs[0]['schedule_sha256'] = 'other'
    if mutation == 'wrong_order': jobs[0]['image_ids'] = [2, 1]
    if mutation == 'wrong_point': points[jobs[0]['name']] += .1
    if mutation == 'nonfinite': draws[jobs[0]['name']][0] = np.nan
    with pytest.raises(ValueError): module().summarize_clean(summary, points, draws, jobs)


def holdout_fixture():
    blocks = [dict(dataset=b.split('_')[0], model=b.split('_')[1], conditions=12,
                   clean_control='original_source', int8_clean_ap_points=40., fp8_clean_ap_points=50.,
                   int8_corrupted_mean_ap_points=20., fp8_corrupted_mean_ap_points=25.,
                   clean_gap_ap_points=10., corrupted_gap_ap_points=5., delta_e_ap_points=-5.,
                   delta_e_percentile95_ap_points=[-5., -5., -5.],
                   corrupted_gap_percentile95_ap_points=[5., 5., 5.]) for b in PRIMARY]
    summary = dict(blocks=blocks, units='AP points', n_boot=2000,
                   primary_macro=dict(blocks=6, delta_e_ap_points=-5., corrupted_gap_ap_points=5.,
                                      delta_e_percentile95_ap_points=[-5., -5., -5.],
                                      corrupted_gap_percentile95_ap_points=[5., 5., 5.]))
    return summary, np.full((6, 2000), -.05), np.full((6, 2000), .05), PRIMARY.copy()


def test_holdout_four_ap_identity_and_paired_macro():
    result = module().summarize_holdout(*holdout_fixture())
    assert result['primary_macro']['delta_e_ap_points'] == pytest.approx(-5)
    assert result['primary_macro']['corrupted_gap_percentile95_ap_points'] == pytest.approx([5, 5, 5])


@pytest.mark.parametrize('mutation', ['labels', 'missing', 'duplicate', 'codec', 'conditions', 'identity', 'nan', 'short'])
def test_holdout_rejects_wrong_scope_labels_or_arithmetic(mutation):
    summary, delta, gap, labels = holdout_fixture()
    if mutation == 'labels': labels.reverse()
    if mutation == 'missing': summary['blocks'].pop()
    if mutation == 'duplicate': summary['blocks'][-1] = summary['blocks'][0]
    if mutation == 'codec': summary['blocks'][0]['clean_control'] = 'j95'
    if mutation == 'conditions': summary['blocks'][0]['conditions'] = 11
    if mutation == 'identity': summary['blocks'][0]['clean_gap_ap_points'] = 9
    if mutation == 'nan': gap[0, 0] = np.nan
    if mutation == 'short': delta = delta[:, :-1]
    with pytest.raises(ValueError): module().summarize_holdout(summary, delta, gap, labels)


def test_completion_rejects_changed_or_unbound_inputs(tmp_path):
    m = module()
    source = tmp_path/'summary.json'
    source.write_text('{}')
    marker = tmp_path/'complete.json'
    marker.write_text(json.dumps({'summary.json': m.sha256_file(source)}))
    m.verify_completion(marker, {'summary.json'})
    with pytest.raises(ValueError): m.verify_completion(marker, {'points.json'})
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError): m.verify_completion(marker, {'summary.json'})


def test_completed_artifacts_reproduce_publication_numbers_and_tables():
    m = module()
    root = Path(__file__).resolve().parents[1]
    base = root/'outputs/analysis/cviu_v4'
    if not (base/'clean_control/complete.json').is_file():
        pytest.skip('completed V4 artifacts unavailable')
    clean = m.load_clean(base/'clean_control')
    holdout = m.load_holdout(base/'holdout_synthesis/recovered')
    assert clean['primary_macro']['S'] == pytest.approx(.057356594037777375)
    assert clean['primary_macro']['interval'][::2] == pytest.approx([-.18392563489576894, .25176608064085154])
    assert holdout['primary_macro']['corrupted_gap_ap_points'] == pytest.approx(1.0502, abs=.00005)
    assert holdout['primary_macro']['delta_e_ap_points'] == pytest.approx(-.5498, abs=.00005)
    clean_tex, holdout_tex = m.render_clean(clean), m.render_holdout(holdout)
    assert '+0.0574' in clean_tex and '[-0.1839, 0.2518]' in clean_tex
    assert '[0.9105, 1.1554]' in holdout_tex and '[-0.7843, -0.2908]' in holdout_tex
    assert 'Diagnostic' in clean_tex and 'not pooled' in clean_tex
    assert r'\begin{table*}' in holdout_tex
    assert r'\resizebox' not in holdout_tex and r'\tiny' not in holdout_tex
