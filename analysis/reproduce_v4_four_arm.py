#!/usr/bin/env python3
"""Evaluate four hash-bound prediction arms without AP caches.

Manifest paths are relative to the manifest directory. Arms are int8_clean,
fp8_clean, int8_corrupt and fp8_corrupt, each with prediction/input/run file
references {path, sha256}. Annotations and schedule have the same reference
schema. The NPZ schedule contains integer ``samples`` (positions, not IDs) and
ordered ``image_ids``. Settings must be recorded in runs or source-derived
sidecars; records are never rewritten. Outputs use AP points and JSON null for
undefined endpoints. This checks internal bindings, not timestamp authenticity
or that historical execution actually occurred.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
for source in (Path(__file__).resolve().parent / 'src', Path(__file__).resolve().parents[1] / 'src'):
    sys.path.insert(0, str(source))
import numpy as np
from bootstrap_format_contrast import ARM_NAMES, bootstrap_arm_ap_draws, validate_linked_inputs
from paired_bootstrap import BINS, accumulate_ap

EVALUATOR = {'endpoint': 'area', 'iou_type': 'bbox', 'max_dets': 100, 'use_categories': True}
INFERENCE_KEYS = {'imgsz', 'conf', 'nms_iou', 'max_det'}


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def referenced(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('file reference requires path and sha256')
    relative = Path(ref['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or root not in path.parents:
        raise ValueError('file reference escapes manifest directory')
    if not path.is_file() or sha(path) != ref['sha256']:
        raise ValueError(f'file SHA-256 mismatch: {relative}')
    return path


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def normalized_settings(root, arm, run):
    if 'settings' not in arm:
        return run.get('inference_settings'), run.get('evaluator_settings'), 'recorded_explicit'
    settings = arm['settings']
    derivation = settings.get('derivation', {})
    if derivation.get('kind') != 'normalized_source_backed':
        raise ValueError('unsupported settings derivation')
    command = run.get('command')
    if not isinstance(command, str) or hashlib.sha256(command.encode()).hexdigest() != derivation.get('command_sha256'):
        raise ValueError('settings command binding mismatch')
    sources = {}
    for key in ('runner', 'preprocess', 'decoder'):
        sources[key] = referenced(root, derivation['sources'][key])
        if sha(sources[key]) != run.get(key + '_sha256'):
            raise ValueError('settings source does not match run')
    constants = {}
    for node in ast.parse(sources['decoder'].read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
                elif isinstance(target, ast.Tuple) and isinstance(value, (tuple, list)):
                    constants.update({name.id: val for name, val in zip(target.elts, value) if isinstance(name, ast.Name)})
    defaults = {}
    for node in ast.walk(ast.parse(sources['runner'].read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in ('--imgsz', '--conf'):
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        defaults[node.args[0].value] = (constants.get(keyword.value.id) if isinstance(keyword.value, ast.Name)
                                                          else ast.literal_eval(keyword.value))
    words = shlex.split(command)
    actual = {}
    for option, caster in (('--imgsz', int), ('--conf', float)):
        values = [words[i + 1] for i, word in enumerate(words[:-1]) if word == option]
        values += [word.split('=', 1)[1] for word in words if word.startswith(option + '=')]
        if len(values) > 1:
            raise ValueError('ambiguous repeated command option')
        value = values[0] if values else defaults.get(option)
        if value is None:
            raise ValueError('effective inference setting cannot be recovered')
        actual[option[2:]] = caster(value)
    actual.update(nms_iou=constants.get('NMS_IOU'), max_det=constants.get('MAX_DET'))
    if actual != settings.get('inference_settings'):
        raise ValueError('source-derived inference settings mismatch')
    return actual, settings.get('evaluator_settings'), 'normalized_source_backed'


def validate_contract(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or set(manifest.get('arms', {})) != set(ARM_NAMES):
        raise ValueError('manifest requires exactly four named arms')
    evaluator = manifest.get('evaluator_settings')
    if evaluator != EVALUATOR:
        raise ValueError('unsupported evaluator settings; this version uses COCO area AP/maxDets100')
    annotation_path = referenced(root, manifest['annotations'])
    annotations = json.loads(annotation_path.read_text())
    arms, inputs, runs, inference, states = {}, [], {}, [], {}
    for name in ARM_NAMES:
        arm = manifest['arms'][name]
        paths = {key: referenced(root, arm[key]) for key in ('prediction', 'input', 'run')}
        record = json.loads(paths['input'].read_text())
        run = json.loads(paths['run'].read_text())
        for key in ('engine_sha256', 'runner_sha256', 'preprocess_sha256', 'decoder_sha256', 'class_map_sha256'):
            if not valid_digest(run.get(key)):
                raise ValueError(f'{name}: missing full {key}')
        if run.get('annotation_sha256') != manifest['annotations']['sha256'] or run.get('prediction_sha256') != arm['prediction']['sha256']:
            raise ValueError(f'{name}: run prediction/annotation binding mismatch')
        if not valid_digest(record.get('image_ids_sha256')):
            raise ValueError(f'{name}: missing full image-ID hash')
        if (run.get('condition_id') != record.get('condition_id')
                or run.get('input_manifest_sha256') != record.get('input_manifest_sha256')
                or run.get('input_image_ids_sha256') != record.get('image_ids_sha256')
                or not valid_digest(record.get('input_manifest_sha256'))):
            raise ValueError(f'{name}: input/run binding mismatch')
        inf, ev, state = normalized_settings(root, arm, run)
        if not isinstance(inf, dict) or set(inf) != INFERENCE_KEYS or ev != evaluator:
            raise ValueError(f'{name}: missing or mismatched settings')
        if (not isinstance(inf['imgsz'], int) or inf['imgsz'] <= 0 or not isinstance(inf['max_det'], int)
                or inf['max_det'] <= 0 or not 0 <= inf['conf'] <= 1 or not 0 <= inf['nms_iou'] <= 1):
            raise ValueError('invalid inference settings')
        arms[name], runs[name], states[name] = paths, run, state
        inputs.append(record)
        inference.append(inf)
    image_ids = validate_linked_inputs(inputs)
    gt_ids = [entry['id'] for entry in annotations['images']]
    if len(gt_ids) != len(set(gt_ids)) or not set(image_ids).issubset(gt_ids):
        raise ValueError('annotation image universe mismatch')
    if any(value != inference[0] for value in inference[1:]):
        raise ValueError('inference settings differ across arms')
    for key in ('runner_sha256', 'preprocess_sha256', 'decoder_sha256', 'class_map_sha256'):
        if len({run[key] for run in runs.values()}) != 1:
            raise ValueError(f'cross-arm {key} mismatch')
    for treatment in ('int8', 'fp8'):
        if runs[treatment + '_clean']['engine_sha256'] != runs[treatment + '_corrupt']['engine_sha256']:
            raise ValueError('engine mismatch within treatment')
    for a, b in ((0, 1), (2, 3)):
        if inputs[a]['input_manifest_sha256'] != inputs[b]['input_manifest_sha256']:
            raise ValueError('treatments do not share encoded-input manifest')
    schedule_path = referenced(root, manifest['schedule'])
    with np.load(schedule_path, allow_pickle=False) as archive:
        samples, schedule_ids = archive['samples'], archive['image_ids']
    if (samples.dtype.kind not in 'iu' or samples.ndim != 2 or samples.shape[1] != len(image_ids)
            or not samples.shape[0] or np.any(samples < 0) or np.any(samples >= len(image_ids))
            or schedule_ids.tolist() != image_ids):
        raise ValueError('invalid paired schedule or image ordering')
    return {'manifest': manifest, 'manifest_sha256': sha(manifest_path), 'annotations': annotations,
            'arms': arms, 'runs': runs, 'image_ids': image_ids, 'samples': samples,
            'inference_settings': inference[0], 'settings_states': states}


def evaluation(document, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = copy.deepcopy(document)
    gt.createIndex()
    category_ids = set(gt.getCatIds())
    for prediction in predictions:
        if prediction['image_id'] not in image_ids or prediction['category_id'] not in category_ids:
            raise ValueError('prediction references foreign image or category')
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = {**copy.deepcopy(document), 'annotations': []}
        dt.createIndex()
    result = COCOeval(gt, dt, iouType='bbox')
    result.params.imgIds = image_ids
    result.params.areaRng = [[low, high] for _, low, high in BINS]
    result.params.areaRngLbl = [name for name, _, _ in BINS]
    result.params.maxDets = [100]
    result.evaluate()
    return result


def named(values, labels):
    return {label: float(values[i]) if np.isfinite(values[i]) else None for i, label in enumerate(labels)}


def metrics(aps):
    clean_gap = aps[1] - aps[0]
    corrupt_gap = aps[3] - aps[2]
    return {**{name: aps[i] for i, name in enumerate(ARM_NAMES)}, 'clean_gap': clean_gap,
            'corrupt_gap': corrupt_gap, 'delta_e': corrupt_gap - clean_gap,
            'int8_loss': aps[0] - aps[2], 'fp8_loss': aps[1] - aps[3]}


def support_report(evaluation, samples):
    categories = evaluation.params.catIds
    n_images, n_areas = len(evaluation._paramsEval.imgIds), len(BINS)
    output = {}
    for a, (label, _, _) in enumerate(BINS):
        counts = np.zeros((len(categories), n_images), dtype=np.int32)
        for c in range(len(categories)):
            for i in range(n_images):
                entry = evaluation.evalImgs[c * n_areas * n_images + a * n_images + i]
                if entry is not None:
                    counts[c, i] = np.count_nonzero(np.asarray(entry['gtIgnore']) == 0)
        point_counts = counts.sum(axis=1)
        represented = np.stack([(counts[:, row].sum(axis=1) > 0) for row in samples])
        output[label] = {'positive_instances': int(point_counts.sum()),
                         'represented_categories': [int(c) for c, n in zip(categories, point_counts) if n > 0],
                         'undefined_categories': [int(c) for c, n in zip(categories, point_counts) if n == 0],
                         'category_positive_counts': {str(c): int(n) for c, n in zip(categories, point_counts)},
                         'draw_represented_category_counts': represented.sum(axis=1).tolist(),
                         'draws_missing_any_point_supported_category': int(np.count_nonzero(np.any(~represented[:, point_counts > 0], axis=1))),
                         'draws_without_positives': int(np.count_nonzero(~represented.any(axis=1)))}
    return output


def reproduce(manifest_path, *, workers=1):
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    contract = validate_contract(manifest_path)
    evaluations = [evaluation(contract['annotations'], json.loads(contract['arms'][name]['prediction'].read_text()),
                              contract['image_ids']) for name in ARM_NAMES]
    # COCOeval sorts IDs: remap schedule positions without sorting the frozen schedule.
    actual_ids = evaluations[0]._paramsEval.imgIds
    remap = np.asarray([actual_ids.index(x) for x in contract['image_ids']])
    samples = remap[contract['samples']]
    point = metrics(100 * np.asarray([accumulate_ap(ev, list(range(len(actual_ids)))) for ev in evaluations]))
    raw = 100 * bootstrap_arm_ap_draws(evaluations, samples, workers=workers)
    draws = metrics(np.transpose(raw, (1, 0, 2)))
    labels = [row[0] for row in BINS]
    intervals = {}
    for key, values in draws.items():
        intervals[key] = {}
        for i, label in enumerate(labels):
            valid = values[:, i][np.isfinite(values[:, i])]
            intervals[key][label] = np.percentile(valid, [2.5, 50, 97.5]).tolist() if len(valid) else [None] * 3
    return {'schema_version': 1, 'units': 'AP points (0–100)', 'manifest_sha256': contract['manifest_sha256'],
            'n_images': len(actual_ids), 'n_boot': len(samples), 'arm_order': list(ARM_NAMES),
            'point': {key: named(value, labels) for key, value in point.items()},
            'draws': {key: {label: [named(row, labels)[label] for row in values] for label in labels}
                      for key, values in draws.items()}, 'intervals': intervals,
            'absolute_corrupted_ap': {name: named(point[name], labels) for name in ('int8_corrupt', 'fp8_corrupt')},
            'support': support_report(evaluations[0], samples),
            'verification': {'input_file_hashes': 'verified', 'run_prediction_annotation_bindings': 'verified',
                             'paired_ordered_ids': 'verified', 'common_encoded_manifest_per_condition': 'verified_recorded_binding',
                             'within_treatment_engine_hash': 'equal_full_recorded_hash',
                             'cross_arm_runner_preprocess_decoder_classmap': 'equal_full_recorded_hash',
                             'settings': contract['settings_states'], 'ap_cache_used': False,
                             'historical_execution_or_timestamp_authenticity': 'not_established_by_this_report',
                             'interval_coverage_guarantee': 'not_tested'},
            'evaluator_settings': {**EVALUATOR, 'iou_thresholds': evaluations[0].params.iouThrs.tolist(),
                                   'recall_thresholds': evaluations[0].params.recThrs.tolist(),
                                   'undefined_category_endpoint': 'omitted, never assigned zero'},
            'inference_settings': contract['inference_settings'],
            'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pycocotools')},
            'uncertainty': 'Paired finite-image percentile summary conditional on the frozen four arms; undefined draws omitted per endpoint.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing to overwrite report')
    report = reproduce(args.manifest, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'point': report['point'], 'verification': report['verification']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
