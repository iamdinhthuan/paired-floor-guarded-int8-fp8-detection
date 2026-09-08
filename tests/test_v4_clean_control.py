import importlib
import importlib.util
import copy
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('run_v4_clean_control'), 'V4 continuous runner missing'
    return importlib.import_module('run_v4_clean_control')


def test_control_substitution_sign_and_equal_controls():
    m = module()
    assert m.shift(.4, .5, .3, .45) == pytest.approx(-5)
    assert m.shift(.4, .5, .4, .5) == pytest.approx(0)
    assert m.shift(.5, .4, .45, .3) == pytest.approx(5)
    with pytest.raises(ValueError):
        m.shift(.4, np.nan, .3, .45)


def test_joint_macro_averages_aligned_draws_not_endpoints():
    m = module()
    x = np.linspace(-1, 1, 2000)
    assert m.macro_interval([x, -x] * 3) == pytest.approx([0, 0, 0])
    with pytest.raises(ValueError):
        m.macro_interval([x] * 5)


def records():
    return {key: dict(image_ids=[1, 2], engine_sha256=('a' if key.startswith('int8') else 'b'),
        runner_sha256='r', preprocess_sha256='p', decoder_sha256='d',
        annotation_sha256='a', class_map_sha256='c', inference_settings={'conf': .001},
        evaluator_settings={'max_dets': 100})
        for key in ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')}


@pytest.mark.parametrize('field,value', [('image_ids', [2, 1]), ('engine_sha256', 'x'),
    ('decoder_sha256', 'x'), ('annotation_sha256', 'x'),
    ('inference_settings', {'conf': .25}), ('evaluator_settings', {'max_dets': 300})])
def test_pairing_rejects_changed_arm(field, value):
    m = module()
    r = records()
    m.validate_pair(r)
    r['int8_j95'][field] = value
    with pytest.raises(ValueError):
        m.validate_pair(r)


def test_schedule_shared_by_dataset_not_precision_or_capacity():
    m = module()
    a = m.schedule([1, 3], 'voc', 5)
    np.testing.assert_array_equal(a, m.schedule([1, 3], 'voc', 5))
    assert a.shape == (5, 2)
    assert not np.array_equal(a, m.schedule([1, 3], 'kitti', 5))
    with pytest.raises(ValueError):
        m.schedule([1, 1], 'voc', 5)


def test_command_controls_change_only_source_and_output_not_engine_settings():
    m = module()
    block = {'dataset': 'kitti', 'model': 'yolo11m', 'annotation': '/a', 'class_map': '/c',
        'imgsz': 640, 'original': {'manifest': '/original.json', 'root': '/images'},
        'j95': {'manifest': '/j95.json', 'root': '/cache'}}
    engine = {'engine': '/retained.plan', 'calibration_list': '/calibration.json',
        'calibration_method': 'entropy'}
    a = m.inference_command('/project', '/out', block, engine, 'int8-entropy', 'original')
    b = m.inference_command('/project', '/out', block, engine, 'int8-entropy', 'j95')
    for flag, expected in [('--engine', '/retained.plan'), ('--conf', '0.001'), ('--imgsz', '640')]:
        assert a[a.index(flag)+1] == b[b.index(flag)+1] == expected
    assert a[a.index('--image-manifest')+1] == '/original.json'
    assert b[b.index('--image-manifest')+1] == '/j95.json'


def test_partial_or_modified_inference_never_silently_reused(tmp_path):
    m = module()
    p = tmp_path / 'arm'
    assert m.reusable_arm(p) is False
    p.mkdir()
    (p / 'predictions.json').write_text('[]')
    with pytest.raises(ValueError):
        m.reusable_arm(p)


def test_historical_recipe_source_and_explicit_threshold_must_match():
    m = module()
    r = {'engine_sha256': 'e', 'runner_sha256': 'r', 'preprocess_sha256': 'p',
         'decoder_sha256': 'd', 'command': 'python coco_infer_trt.py --imgsz 640'}
    hashes = {'runner_sha256': 'r', 'preprocess_sha256': 'p', 'decoder_sha256': 'd'}
    m.validate_historical_recipe(r, 'e', 640, hashes)
    for edit in ({'decoder_sha256': 'different'}, {'engine_sha256': 'different'},
                 {'command': 'python coco_infer_trt.py --imgsz 640 --conf 0.25'}):
        with pytest.raises(ValueError):
            m.validate_historical_recipe(dict(r, **edit), 'e', 640, hashes)


def test_gpu_gate_allows_observed_small_desktop_service_not_other_compute():
    m = module()
    assert m.busy_gpu_pids('3635790, /usr/bin/sunshine, 513\n') == []
    assert m.busy_gpu_pids('123, python, 513\n') == ['123']
    assert m.busy_gpu_pids('3635790, /usr/bin/sunshine, 2048\n') == ['3635790']


def test_named_j95_manifest_must_really_bind_jpeg95_no_subsampling():
    m = module()
    encoding = {'format': 'JPEG', 'quality': 95, 'subsampling': 0}
    d = {'encoding': encoding, 'records': [dict(encoding=encoding, corruption='codec_control', severity=0)]}
    m.validate_j95(d)
    broken = copy.deepcopy(d)
    broken['records'][0]['encoding']['quality'] = 75
    with pytest.raises(ValueError):
        m.validate_j95(broken)


def test_resumed_run_command_and_ordered_identity_cannot_be_relabelled():
    m = module()
    import hashlib, json
    command = ['python', '/runner.py', '--condition-id', 'fixed', '--conf', '0.001']
    ids_hash = hashlib.sha256(b'[1,2]').hexdigest()
    r = {'command': '/runner.py --condition-id fixed --conf 0.001', 'condition_id': 'fixed',
         'input_image_ids_sha256': ids_hash, 'n_images': 2, 'corruption': 'clean', 'severity': 0}
    inputs = {'condition_id': 'fixed', 'image_ids': [1,2], 'image_ids_sha256': ids_hash}
    m.validate_run_command(r, inputs, command, 'original')
    for edit in ({'command': '/runner.py --condition-id fixed --conf 0.25'},
                 {'condition_id': 'other'}, {'n_images': 1}):
        with pytest.raises(ValueError):
            m.validate_run_command(dict(r, **edit), inputs, command, 'original')
