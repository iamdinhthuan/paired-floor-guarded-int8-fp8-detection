#!/usr/bin/env python3
"""V4 immutable clean-control registry, serial inference and automatic paired AP tail.

This version never edits historical runs. S is a clean-control substitution,
not twelve independently replicated corruption effects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

# Must precede NumPy imports and is inherited by inference and CPU workers.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
import numpy as np

from topic_c.manifest import read_manifest, sha256_file, validate_manifest
from run_controlled_postprocess import bootstrap_task, write_json

ATTEMPT = 'cviu_v4_clean_control_v1'
SEEDS = {'voc': 202609071, 'kitti': 202609072, 'tt100k': 202609073}
ARMS = ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')
EVALUATOR = {'iou_type': 'bbox', 'iou_thresholds': [.5 + .05*i for i in range(10)],
             'recall_thresholds': 101, 'max_dets': 100, 'area': 'all',
             'undefined_categories': 'omit', 'bootstrap_unit': 'image_with_multiplicity'}


def shift(int8_original, fp8_original, int8_j95, fp8_j95):
    values = [np.asarray(v, dtype=float) for v in (int8_original, fp8_original, int8_j95, fp8_j95)]
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError('nonfinite AP')
    return 100 * ((values[1] - values[0]) - (values[3] - values[2]))


def macro_interval(rows):
    values = np.asarray(rows, dtype=float)
    if values.shape != (6, 2000) or not np.isfinite(values).all():
        raise ValueError('six blocks by 2000 aligned finite draws required')
    return np.percentile(values.mean(axis=0), [2.5, 50, 97.5]).tolist()


def validate_pair(records):
    if set(records) != set(ARMS):
        raise ValueError('exact four clean arms required')
    reference = records[ARMS[0]]
    for record in records.values():
        for key in ('image_ids', 'runner_sha256', 'preprocess_sha256', 'decoder_sha256',
                    'annotation_sha256', 'class_map_sha256', 'inference_settings', 'evaluator_settings'):
            if not record.get(key) or record[key] != reference[key]:
                raise ValueError(f'four-arm pairing mismatch: {key}')
    for precision in ('int8', 'fp8'):
        if records[f'{precision}_original']['engine_sha256'] != records[f'{precision}_j95']['engine_sha256']:
            raise ValueError('engine differs between clean controls')
    ids = reference['image_ids']
    if not ids or ids != sorted(set(ids)):
        raise ValueError('IDs must be unique and ordered')


def schedule(ids, dataset, n_boot=2000):
    if not ids or ids != sorted(set(ids)):
        raise ValueError('invalid schedule IDs')
    return np.random.default_rng(SEEDS[dataset]).integers(0, len(ids), (n_boot, len(ids)), dtype=np.int32)


def inference_command(root, out, block, engine, precision, control):
    arm = ('int8' if precision == 'int8-entropy' else precision) + '_' + control
    destination = Path(out) / 'inference' / f"{block['dataset']}_{block['model']}" / arm
    source = block[control]
    return [sys.executable, str(Path(root) / 'src/coco_infer_trt.py'),
        '--engine', engine['engine'], '--annotations', block['annotation'],
        '--image-manifest', source['manifest'], '--manifest-cache-root', source['root'],
        '--out', str(destination/'predictions.json'), '--input-record', str(destination/'input.json'),
        '--run-record', str(destination/'run.json'),
        '--condition-id', f"{ATTEMPT}__{block['dataset']}__{block['model']}__{arm}",
        '--dataset', block['dataset'], '--split', 'test', '--model', block['model'],
        '--precision', precision, '--calibrator', engine['calibration_method'],
        '--calibration-list', engine['calibration_list'],
        '--calibration-method', engine['calibration_method'], '--calibration-provenance', 'verified',
        '--corruption', 'clean' if control == 'original' else 'codec_control', '--severity', '0',
        '--class-map', block['class_map'], '--imgsz', str(block['imgsz']), '--conf', '0.001']


def reusable_arm(path):
    path = Path(path)
    names = ('predictions.json', 'input.json', 'run.json')
    if not path.exists():
        return False
    marker = path/'complete.json'
    if not marker.is_file() or any(not (path/name).is_file() for name in names):
        raise ValueError(f'partial inference preserved for review: {path}')
    if json.loads(marker.read_text()) != {name: sha256_file(path/name) for name in names}:
        raise ValueError(f'inference binding mismatch: {path}')
    return True


def checked(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f'frozen file changed: {path}')


def validate_historical_recipe(record, engine_hash, imgsz, source_hashes):
    for key, expected in dict(source_hashes, engine_sha256=engine_hash).items():
        if record.get(key) != expected:
            raise ValueError(f'historical recipe mismatch: {key}')
    command = shlex.split(record['command'])
    observed_size = int(command[command.index('--imgsz')+1]) if '--imgsz' in command else 640
    observed_conf = float(command[command.index('--conf')+1]) if '--conf' in command else .001
    if observed_size != imgsz or observed_conf != .001:
        raise ValueError('historical inference settings differ')


def validate_j95(document):
    encoding = {'format': 'JPEG', 'quality': 95, 'subsampling': 0}
    if document.get('encoding') != encoding or not document.get('records'):
        raise ValueError('JPEG95 encoding contract missing')
    for record in document['records']:
        if (record.get('encoding') != encoding or record.get('corruption') != 'codec_control'
                or record.get('severity') != 0):
            raise ValueError('JPEG95 record encoding/condition differs')


def validate_run_command(record, inputs, command, control):
    if shlex.split(record['command']) != command[1:]:
        raise ValueError('actual inference command differs from frozen command')
    identity = command[command.index('--condition-id')+1]
    digest = hashlib.sha256(json.dumps(inputs['image_ids'], separators=(',', ':')).encode()).hexdigest()
    if (record['condition_id'] != identity or inputs['condition_id'] != identity
            or inputs['image_ids_sha256'] != digest or record['input_image_ids_sha256'] != digest
            or record['n_images'] != len(inputs['image_ids']) or record['severity'] != 0
            or record['corruption'] != ('clean' if control == 'original' else 'codec_control')):
        raise ValueError('run identity/condition/count mismatch')


def runtime_versions():
    return {name: importlib.metadata.version(name)
            for name in ('numpy', 'Pillow', 'pycocotools', 'opencv-python', 'tensorrt')}


def run(command):
    print('RUN ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def prepare(root, config, out):
    if (out/'registry.json').exists():
        raise ValueError('registry already frozen; use --run')
    spec = json.loads(config.read_text())
    expected = {(d, m) for d in ('voc', 'kitti') for m in ('yolo11n', 'yolo11m', 'yolo11x')}
    expected |= {('tt100k', m) for m in ('yolo11n', 'yolo11x')}
    if (spec['attempt'] != ATTEMPT or spec['seed_mapping'] != SEEDS or spec['n_boot'] != 2000
            or {(b['dataset'], b['model']) for b in spec['blocks']} != expected or len(spec['blocks']) != 8):
        raise ValueError('frozen V4 scope changed')
    bindings = {str(config): sha256_file(config)}
    import pycocotools.coco, pycocotools.cocoeval, pycocotools._mask
    for package in (pycocotools.coco, pycocotools.cocoeval, pycocotools._mask):
        bindings[str(Path(package.__file__).resolve())] = sha256_file(package.__file__)
    for source in ('src/coco_infer_trt.py', 'src/topic_c/coco_data.py', 'src/topic_c/yolo_decode.py',
                   'src/topic_c/manifest.py', 'src/materialize_codec_control.py', 'src/run_v4_clean_control.py',
                   'src/paired_bootstrap.py', 'src/accelerate_shared_mask_bootstrap_v3.py',
                   'src/run_controlled_postprocess.py', 'src/run_fixed_universe_sensitivity.py', spec['analysis_plan']):
        bindings[str(root/source)] = sha256_file(root/source)
    from topic_c.yolo_decode import CONF_FLOOR, NMS_IOU, MAX_DET
    if (CONF_FLOOR, NMS_IOU, MAX_DET) != (.001, .7, 300):
        raise ValueError('decoder settings differ from frozen historical recipe')
    datasets = {}
    for dataset, data in spec['datasets'].items():
        annotation = root/data['annotation']
        original_path = root/data['original_manifest']
        original = read_manifest(original_path)
        ids = original['expected_image_ids']
        if len(ids) != data['n_images']:
            raise ValueError(f'wrong image count: {dataset}')
        failures = validate_manifest(original, annotation, root/data['original_root'],
                                     root/data['original_root'], require_pixels_changed=False)
        if failures:
            raise ValueError(str(failures[:10]))
        j95_path = out/'inputs'/f'{dataset}_j95.json'
        cache = root/'data'/ATTEMPT/dataset
        if not j95_path.exists():
            run([sys.executable, str(root/'src/materialize_codec_control.py'), '--dataset', dataset,
                 '--split', 'test', '--annotations', str(annotation), '--clean-root', str(root/data['original_root']),
                 '--cache-root', str(cache), '--manifest-out', str(j95_path), '--quality', '95', '--subsampling', '0',
                 '--resume-validated'])
        j95 = read_manifest(j95_path)
        validate_j95(j95)
        if j95['expected_image_ids'] != ids:
            raise ValueError('control image universe differs')
        failures = validate_manifest(j95, annotation, cache, root/data['original_root'], require_pixels_changed=False)
        if failures:
            raise ValueError(str(failures[:10]))
        for path in (annotation, original_path, j95_path, root/data['class_map'],
                     Path(str(original_path)+'.complete'), Path(str(j95_path)+'.complete')):
            bindings[str(path)] = sha256_file(path)
        datasets[dataset] = dict(dataset=dataset, annotation=str(annotation), class_map=str(root/data['class_map']),
            image_ids=ids, imgsz=data['imgsz'],
            original={'manifest': str(original_path), 'root': str(root/data['original_root'])},
            j95={'manifest': str(j95_path), 'root': str(cache)})
    blocks = []
    for item in spec['blocks']:
        block = dict(datasets[item['dataset']], model=item['model'])
        block['engines'], block['commands'] = {}, {}
        for precision, registry_name in item['engine_registries'].items():
            registry = root/registry_name
            engine = json.loads(registry.read_text())
            if (engine['dataset'], engine['model'], engine['precision'], engine['imgsz']) != (
                    item['dataset'], item['model'], precision, block['imgsz']):
                raise ValueError('engine registry identity mismatch')
            checked(engine['engine'], engine['engine_sha256'])
            checked(engine['source_onnx'], engine['source_onnx_sha256'])
            from pilot_registry import calibration_sha256
            if calibration_sha256(engine['calibration_list']) != engine['calibration_sha256']:
                raise ValueError('engine calibration binding mismatch')
            for path in (registry, Path(engine['engine']), Path(engine['source_onnx']), Path(engine['calibration_list'])):
                bindings[str(path)] = sha256_file(path)
            if item['dataset'] in ('voc', 'kitti'):
                precision_label = 'fp8-entropy' if precision == 'fp8' else precision
                pattern = f"{item['dataset']}_test__{item['model']}__{precision_label}__clean-s0__*.json"
                matches = list((root/'manifests/runs'/f"{item['dataset']}_confirmatory_final_117_v1").glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f'exact historical clean record required: {pattern}')
                historical = json.loads(matches[0].read_text())
                validate_historical_recipe(historical, engine['engine_sha256'], block['imgsz'], {
                    'runner_sha256': bindings[str(root/'src/coco_infer_trt.py')],
                    'preprocess_sha256': bindings[str(root/'src/topic_c/coco_data.py')],
                    'decoder_sha256': bindings[str(root/'src/topic_c/yolo_decode.py')]})
                bindings[str(matches[0])] = sha256_file(matches[0])
                block.setdefault('historical_recipe_records', {})[precision] = str(matches[0])
            else:
                precision_label = 'fp8-none' if precision == 'fp8' else precision
                pattern = f"tt100k_test__{item['model']}__{precision_label}__clean-s0__*.json"
                matches = list((root/'manifests/runs/tt100k_pilot_117_v1').glob(pattern))
                if len(matches) != 1:
                    raise ValueError('exact historical TT100K clean record required')
                historical = json.loads(matches[0].read_text())
                validate_historical_recipe(historical, engine['engine_sha256'], block['imgsz'], {
                    'preprocess_sha256': bindings[str(root/'src/topic_c/coco_data.py')],
                    'decoder_sha256': bindings[str(root/'src/topic_c/yolo_decode.py')]})
                bindings[str(matches[0])] = sha256_file(matches[0])
                block.setdefault('historical_recipe_records', {})[precision] = str(matches[0])
                block['historical_runner_note'] = 'historical wrapper differs; V4 uses one newly bound runner for both controls; no historical clean-corrupt common-runner claim'
            block['engines'][precision] = engine
            for control in ('original', 'j95'):
                arm = ('int8' if precision == 'int8-entropy' else precision)+'_'+control
                block['commands'][arm] = inference_command(root, out, block, engine, precision, control)
        blocks.append(block)
    import PIL
    registry = dict(schema_version=1, attempt=ATTEMPT, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        files=bindings, blocks=blocks, seed_mapping=SEEDS, n_boot=2000,
        evaluator_settings=EVALUATOR, python=sys.version, numpy=np.__version__, pillow=PIL.__version__,
        runtime_versions=runtime_versions(),
        inference_settings={'conf': .001, 'nms_iou': .7, 'max_detections': 300, 'batch': 1},
        scope='post-hoc controlled clean-only substitution; primary six holdout blocks; two separate TT100K diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'registry.json', registry)
    write_json(out/'registry.complete.json', {'sha256': sha256_file(out/'registry.json')})
    print('FROZEN 8 blocks / 32 clean arms; no V4 AP inspected', flush=True)


def validate_inference(block, arm, path, registry):
    record = json.loads((path/'run.json').read_text())
    inputs = json.loads((path/'input.json').read_text())
    precision = 'int8-entropy' if arm.startswith('int8') else 'fp8'
    control = 'original' if arm.endswith('original') else 'j95'
    validate_run_command(record, inputs, block['commands'][arm], control)
    engine = block['engines'][precision]
    expected = {'engine_sha256': engine['engine_sha256'], 'annotation_sha256': sha256_file(block['annotation']),
        'class_map_sha256': sha256_file(block['class_map']), 'prediction_sha256': sha256_file(path/'predictions.json'),
        'dataset': block['dataset'], 'model': block['model'], 'precision': precision,
        'input_manifest_sha256': read_manifest(block[control]['manifest'])['manifest_sha256']}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'run binding mismatch: {path}: {key}')
    root = Path(__file__).resolve().parents[1]
    for key, file in [('runner_sha256','src/coco_infer_trt.py'), ('preprocess_sha256','src/topic_c/coco_data.py'),
                      ('decoder_sha256','src/topic_c/yolo_decode.py')]:
        if record.get(key) != registry['files'][str(root/file)]:
            raise ValueError(f'run source mismatch: {key}')
    if (inputs['image_ids'] != block['image_ids'] or inputs['input_manifest_sha256'] != expected['input_manifest_sha256']
            or inputs['image_ids_sha256'] != record['input_image_ids_sha256']):
        raise ValueError('input identity mismatch')
    return dict(record, image_ids=inputs['image_ids'],
                inference_settings=dict(registry['inference_settings'], imgsz=block['imgsz']),
                evaluator_settings=registry['evaluator_settings'])


def busy_gpu_pids(output):
    busy = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, name, memory = [v.strip() for v in line.split(',')]
        # Observed resident remote-desktop service; never terminate it.
        if name == '/usr/bin/sunshine' and float(memory) < 1024:
            continue
        busy.append(pid)
    return busy


def wait_gpu():
    while True:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                                text=True, capture_output=True, check=True)
        if not busy_gpu_pids(result.stdout):
            return
        print('WAIT other GPU compute process; checking in 30 seconds', flush=True)
        time.sleep(30)


def execute(root, out, workers):
    registry_path = out/'registry.json'
    checked(registry_path, json.loads((out/'registry.complete.json').read_text())['sha256'])
    registry = json.loads(registry_path.read_text())
    if runtime_versions() != registry['runtime_versions'] or sys.version != registry['python']:
        raise ValueError('runtime environment differs from frozen registry')
    for path, digest in registry['files'].items():
        checked(path, digest)
    for i, block in enumerate(registry['blocks']):
        records = {}
        for arm, command in block['commands'].items():
            path = Path(command[command.index('--out')+1]).parent
            if not reusable_arm(path):
                wait_gpu()
                print(f"INFERENCE {i*4+len(records)+1}/32 {block['dataset']} {block['model']} {arm}", flush=True)
                run(command)
                validate_inference(block, arm, path, registry)
                write_json(path/'complete.json', {name: sha256_file(path/name) for name in ('predictions.json','input.json','run.json')})
            records[arm] = validate_inference(block, arm, path, registry)
        validate_pair(records)
    print('INFERENCE_DONE 32/32; automatic CPU AP/bootstrap tail', flush=True)
    analyze(out, registry, workers)


def analyze(out, registry, workers):
    from pycocotools.coco import COCO
    from paired_bootstrap import build_eval
    from accelerate_shared_mask_bootstrap_v3 import accumulate_ap_overall
    jobs, points = [], {}
    for block in registry['blocks']:
        dataset = block['dataset']
        schedule_path = out/'bootstrap'/f'{dataset}_schedule.npz'
        samples = schedule(block['image_ids'], dataset)
        if schedule_path.exists():
            with np.load(schedule_path) as saved:
                np.testing.assert_array_equal(saved['samples'], samples)
                np.testing.assert_array_equal(saved['image_ids'], block['image_ids'])
        else:
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            with schedule_path.open('xb') as handle:
                np.savez_compressed(handle, samples=samples, image_ids=block['image_ids'])
        for arm, command in block['commands'].items():
            name = f"{dataset}_{block['model']}__{arm}"
            prediction = Path(command[command.index('--out')+1])
            evaluation = build_eval(COCO(block['annotation']), json.loads(prediction.read_text()), block['image_ids'])
            point = accumulate_ap_overall(evaluation, list(range(len(block['image_ids']))))
            if (evaluation.params.maxDets != [100] or evaluation.params.iouType != 'bbox'
                    or not evaluation.params.useCats or len(evaluation.params.recThrs) != 101):
                raise ValueError('actual evaluator parameters differ')
            np.testing.assert_allclose(evaluation.params.iouThrs, EVALUATOR['iou_thresholds'], atol=1e-15, rtol=0)
            del evaluation
            points[name] = point
            write_json(prediction.parent/'metric.json', dict(AP=point, AP_points=100*point,
                annotation_sha256=sha256_file(block['annotation']), prediction_sha256=sha256_file(prediction),
                run_sha256=sha256_file(prediction.parent/'run.json'), input_sha256=sha256_file(prediction.parent/'input.json'),
                registry_sha256=sha256_file(out/'registry.json'), evaluator_settings=EVALUATOR,
                runtime_versions=registry['runtime_versions']))
            jobs.append(dict(name=name, annotations=block['annotation'], annotation_sha256=sha256_file(block['annotation']),
                prediction=str(prediction), prediction_sha256=sha256_file(prediction),
                schedule=str(schedule_path), schedule_sha256=sha256_file(schedule_path), image_ids=block['image_ids'],
                n_boot=2000, point=point, out=str(out/'bootstrap'/f'{name}.npz')))
    write_json(out/'points.json', points)
    write_json(out/'bootstrap_jobs.json', jobs)
    draws = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(bootstrap_task, job): job['name'] for job in jobs}
        for future in as_completed(futures):
            draws[futures[future]] = future.result()
            print(f'BOOTSTRAP_DONE {len(draws)}/32', flush=True)
    rows, primary = [], []
    for block in registry['blocks']:
        key = f"{block['dataset']}_{block['model']}"
        ap = [points[f'{key}__{arm}'] for arm in ARMS]
        values = shift(*(draws[f'{key}__{arm}'] for arm in ARMS))
        rows.append(dict(block=key, group='diagnostic' if block['dataset']=='tt100k' else 'primary',
            ap_points=dict(zip(ARMS, [100*v for v in ap])), S=float(shift(*ap)),
            interval=np.percentile(values, [2.5,50,97.5]).tolist()))
        if block['dataset'] != 'tt100k':
            primary.append(values)
    result = dict(scope=registry['scope'], blocks=rows,
        primary_macro={'S': float(np.mean([r['S'] for r in rows if r['group']=='primary'])),
                       'interval': macro_interval(primary)}, units='AP points', n_boot=2000,
        registry_sha256=sha256_file(out/'registry.json'),
        note='Intervals conditional on retained engines and encoded bytes; TT100K diagnostics not pooled.')
    write_json(out/'summary.json', result)
    files = [out/'registry.json', out/'points.json', out/'bootstrap_jobs.json', out/'summary.json']
    files += list((out/'bootstrap').glob('*'))
    write_json(out/'complete.json', {str(p.relative_to(out)): sha256_file(p) for p in files})
    print('COMPLETE V4 clean-control: 32 inference + 32 paired bootstrap arms', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not (args.prepare or args.run):
        raise SystemExit('choose --prepare and/or --run; workers 1..8')
    root = args.project_root.resolve()
    out = root/'outputs'/ATTEMPT
    out.mkdir(parents=True, exist_ok=True)
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(root, args.config.resolve(), out)
        if args.run:
            execute(root, out, args.workers)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('FAILED: preserving evidence; no automatic changed-setting retry', flush=True)
        raise
