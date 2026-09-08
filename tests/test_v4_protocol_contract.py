"""Prediction-level contract regressions; all AP expectations use real COCOeval."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'analysis/reproduce_v4_four_arm.py'
    assert path.exists(), 'V4 prediction-level protocol entry point is missing'
    spec = importlib.util.spec_from_file_location('v4_four_arm', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(root, name, data):
    path = root / name
    path.write_text(json.dumps(data))
    return {'path': name, 'sha256': digest(path)}


def fixture(root, *, equal_treatments=False, equal_conditions=False):
    root.mkdir(parents=True, exist_ok=True)
    ids = [20, 10]  # Deliberately not COCO's internally sorted order.
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    annotations = {'info': {}, 'images': [{'id': x, 'width': 100, 'height': 100} for x in ids],
                   'categories': [{'id': 1, 'name': 'object'}, {'id': 2, 'name': 'absent'}],
                   'annotations': [{'id': i + 1, 'image_id': x, 'category_id': 1,
                                    'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0}
                                   for i, x in enumerate(ids)]}
    ann = write(root, 'annotations.json', annotations)
    pred = lambda x: {'image_id': x, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .9}
    values = {'int8_clean': [pred(20)], 'fp8_clean': [pred(20), pred(10)],
              'int8_corrupt': [], 'fp8_corrupt': [pred(10)]}
    if equal_treatments:
        values['fp8_clean'] = values['int8_clean']
        values['fp8_corrupt'] = values['int8_corrupt']
    if equal_conditions:
        values['int8_corrupt'] = values['int8_clean']
        values['fp8_corrupt'] = values['fp8_clean']
    evaluator = {'endpoint': 'area', 'iou_type': 'bbox', 'max_dets': 100, 'use_categories': True}
    inference = {'imgsz': 640, 'conf': .001, 'nms_iou': .7, 'max_det': 300}
    arms = {}
    for name, predictions in values.items():
        prediction = write(root, name + '.predictions.json', predictions)
        input_record = {'condition_id': name, 'image_ids': ids, 'image_ids_sha256': ids_hash,
                        'input_manifest_sha256': ('a' if name.endswith('clean') else 'b') * 64}
        input_ref = write(root, name + '.input.json', input_record)
        run = {'condition_id': name, 'prediction_sha256': prediction['sha256'],
               'input_manifest_sha256': input_record['input_manifest_sha256'],
               'input_image_ids_sha256': ids_hash, 'annotation_sha256': ann['sha256'],
               'engine_sha256': ('1' if name.startswith('int8') else '2') * 64,
               'runner_sha256': '3' * 64, 'preprocess_sha256': '4' * 64,
               'decoder_sha256': '5' * 64, 'class_map_sha256': '6' * 64,
               'inference_settings': inference, 'evaluator_settings': evaluator}
        arms[name] = {'prediction': prediction, 'input': input_ref,
                      'run': write(root, name + '.run.json', run)}
    np.savez(root / 'schedule.npz', samples=np.array([[0, 0], [1, 1], [0, 1], [1, 0]]), image_ids=ids)
    manifest = {'schema_version': 1, 'annotations': ann,
                'schedule': {'path': 'schedule.npz', 'sha256': digest(root / 'schedule.npz')},
                'evaluator_settings': evaluator, 'arms': arms}
    write(root, 'manifest.json', manifest)
    return root / 'manifest.json'


def alter_record(manifest_path, name, kind, change):
    manifest = json.loads(manifest_path.read_text())
    ref = manifest['arms'][name][kind]
    data = json.loads((manifest_path.parent / ref['path']).read_text())
    change(data)
    manifest['arms'][name][kind] = write(manifest_path.parent, ref['path'], data)
    manifest_path.write_text(json.dumps(manifest))


def evaluate(path, **kwargs):
    m = module()
    pytest.importorskip('pycocotools')
    return m.reproduce(path, **kwargs)


def test_identical_treatments_zero_in_every_draw(tmp_path):
    report = evaluate(fixture(tmp_path, equal_treatments=True))
    for key in ('clean_gap', 'corrupt_gap', 'delta_e'):
        assert report['point'][key]['all'] == 0
        assert report['draws'][key]['all'] == [0, 0, 0, 0]


def test_identical_conditions_zero_interaction(tmp_path):
    report = evaluate(fixture(tmp_path, equal_conditions=True))
    assert report['draws']['delta_e']['all'] == [0, 0, 0, 0]


def test_treatment_swap_negates_gaps_and_interval(tmp_path):
    path = fixture(tmp_path)
    first = evaluate(path)
    manifest = json.loads(path.read_text())
    for condition in ('clean', 'corrupt'):
        a, b = 'int8_' + condition, 'fp8_' + condition
        manifest['arms'][a], manifest['arms'][b] = manifest['arms'][b], manifest['arms'][a]
    path.write_text(json.dumps(manifest))
    swapped = evaluate(path)
    for key in ('clean_gap', 'corrupt_gap', 'delta_e'):
        np.testing.assert_allclose(swapped['draws'][key]['all'], -np.asarray(first['draws'][key]['all']))
        np.testing.assert_allclose(swapped['intervals'][key]['all'], -np.asarray(first['intervals'][key]['all'])[::-1])


@pytest.mark.parametrize('kind,change', [
    ('input', lambda x: x.update(image_ids=[10, 20])),
    ('run', lambda x: x['inference_settings'].update(conf=.01)),
    ('run', lambda x: x['evaluator_settings'].update(max_dets=300)),
    ('run', lambda x: x.update(engine_sha256='7' * 64)),
    ('run', lambda x: x.update(annotation_sha256='8' * 64)),
    ('run', lambda x: x.update(prediction_sha256='9' * 64)),
    ('run', lambda x: x.update(runner_sha256='a' * 64)),
])
def test_mismatched_contract_rejects_before_evaluation(tmp_path, kind, change):
    path = fixture(tmp_path)
    alter_record(path, 'int8_corrupt', kind, change)
    with pytest.raises(ValueError):
        module().validate_contract(path)


def test_duplicate_positions_preserved_and_order_remapped(tmp_path):
    report = evaluate(fixture(tmp_path))
    assert report['draws']['int8_clean']['all'] == pytest.approx([100, 0, 100 * 51 / 101, 100 * 51 / 101])
    assert report['point']['int8_clean']['all'] == pytest.approx(100 * 51 / 101)
    assert report['draws']['int8_corrupt']['all'] == [0, 0, 0, 0]


def test_undefined_endpoint_is_null_not_zero_and_categories_reported(tmp_path):
    report = evaluate(fixture(tmp_path))
    assert report['point']['int8_clean']['large'] is None
    assert report['draws']['delta_e']['large'] == [None] * 4
    assert report['support']['all']['represented_categories'] == [1]
    assert report['support']['all']['undefined_categories'] == [2]
    assert report['support']['large']['draws_without_positives'] == 4
    json.dumps(report, allow_nan=False)


def test_worker_bound_and_float_schedule_are_rejected(tmp_path):
    path = fixture(tmp_path)
    with pytest.raises(ValueError, match='workers'):
        module().reproduce(path, workers=9)
    np.savez(tmp_path / 'schedule.npz', samples=np.array([[.1, 1.]]), image_ids=[20, 10])
    data = json.loads(path.read_text())
    data['schedule']['sha256'] = digest(tmp_path / 'schedule.npz')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schedule'):
        module().validate_contract(path)


def test_missing_image_digest_cannot_be_verified_by_equal_missing_values(tmp_path):
    path = fixture(tmp_path)
    for name in ('int8_clean', 'fp8_clean', 'int8_corrupt', 'fp8_corrupt'):
        alter_record(path, name, 'input', lambda x: x.pop('image_ids_sha256'))
        alter_record(path, name, 'run', lambda x: x.pop('input_image_ids_sha256'))
    with pytest.raises(ValueError, match='image'):
        module().validate_contract(path)


def test_source_backed_settings_uses_bound_defaults_not_current_defaults(tmp_path):
    path = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    sources = {}
    for key, body in {'runner': "parser.add_argument('--imgsz', type=int, default=640)\nparser.add_argument('--conf', type=float, default=CONF_FLOOR)\n",
                      'decoder': 'MAX_DET, CONF_FLOOR, NMS_IOU = 300, 0.002, 0.7\n',
                      'preprocess': '# retained preprocess source\n'}.items():
        source = tmp_path / (key + '.py')
        source.write_text(body)
        sources[key] = {'path': source.name, 'sha256': digest(source)}
    command = 'python coco_infer_trt.py --imgsz=640'
    for name, arm in manifest['arms'].items():
        run_path = tmp_path / arm['run']['path']
        run = json.loads(run_path.read_text())
        run.pop('inference_settings')
        run.pop('evaluator_settings')
        run.update(command=command, **{key + '_sha256': ref['sha256'] for key, ref in sources.items()})
        arm['run'] = write(tmp_path, arm['run']['path'], run)
        arm['settings'] = {'inference_settings': {'imgsz': 640, 'conf': .002, 'nms_iou': .7, 'max_det': 300},
                           'evaluator_settings': manifest['evaluator_settings'],
                           'derivation': {'kind': 'normalized_source_backed',
                                          'command_sha256': hashlib.sha256(command.encode()).hexdigest(), 'sources': sources}}
    path.write_text(json.dumps(manifest))
    contract = module().validate_contract(path)
    assert contract['inference_settings']['conf'] == .002
    assert set(contract['settings_states'].values()) == {'normalized_source_backed'}
    manifest['arms']['int8_clean']['settings']['inference_settings']['conf'] = .001
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source-derived'):
        module().validate_contract(path)


def test_cli_fresh_directory_produces_strict_json_and_refuses_overwrite(tmp_path):
    path = fixture(tmp_path / 'example')
    module()
    pytest.importorskip('pycocotools')
    script = Path(__file__).resolve().parents[1] / 'analysis/reproduce_v4_four_arm.py'
    command = [sys.executable, str(script), '--manifest', str(path), '--out', str(tmp_path / 'result.json'), '--workers', '2']
    completed = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / 'result.json').read_text())
    assert report['verification']['ap_cache_used'] is False
    assert report['point']['int8_clean']['all'] == pytest.approx(100 * 51 / 101)
    again = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert again.returncode != 0
    assert 'refusing to overwrite' in again.stderr
