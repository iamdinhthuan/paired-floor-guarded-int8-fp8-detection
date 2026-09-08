#!/usr/bin/env python3
"""Package the fixed KITTI final / YOLO11m / original-clean versus fog-s1 case."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_v4_four_arm import ARM_NAMES, EVALUATOR, sha, validate_contract


def select_example(audit):
    fixed = audit['fixed_example']
    if (fixed['dataset'], fixed['model'], fixed['corruption'], fixed['severity']) != ('kitti', 'yolo11m', 'fog', 1):
        raise ValueError('wrong fixed example identity')
    semantics = audit['dataset_inputs']['kitti']['original_control_semantics']
    if (semantics.get('control') != 'original_source' or semantics.get('records') != 1197
            or semantics.get('original_source_identity_records') != 1197):
        raise ValueError('clean is not audited original-source identity')
    selected = {}
    for row in fixed['runs']:
        run = row['run_record']
        if (row['status'] != 'verified' or run['dataset'] != 'kitti' or run['model'] != 'yolo11m'
                or run['n_images'] != 1197 or (run['corruption'], run['severity']) not in (('clean', 0), ('fog', 1))):
            raise ValueError('incomplete or wrong-scope fixed example')
        precision = {'int8-entropy': 'int8', 'fp8': 'fp8'}.get(run['precision'])
        name = precision + ('_clean' if run['corruption'] == 'clean' else '_corrupt') if precision else None
        if name not in ARM_NAMES or name in selected:
            raise ValueError('missing/duplicate fixed-example arm')
        selected[name] = row
    if set(selected) != set(ARM_NAMES):
        raise ValueError('all four fixed-example arms are required')
    return selected


def bound_copy(source, out, relative, expected):
    source, out = Path(source), Path(out)
    if sha(source) != expected:
        raise ValueError(f'changed source: {source}')
    dest = out / relative
    if dest.exists():
        raise FileExistsError(f'refusing to overwrite {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    if sha(dest) != expected:
        raise ValueError('copied bytes differ')
    return {'path': relative, 'sha256': expected}


def compare_report(report, expected):
    errors = {key: abs(report['point'][key]['all'] - value) for key, value in expected['point_ap'].items()}
    errors['delta_e'] = abs(report['point']['delta_e']['all'] - expected['point_delta_e'])
    if max(errors.values()) > 1e-8:
        raise ValueError(f'prediction-level point recomputation mismatch: {errors}')
    interval_error = None
    if expected.get('delta_e_interval') is not None:
        interval_error = float(np.max(np.abs(np.asarray(report['intervals']['delta_e']['all']) - expected['delta_e_interval'])))
        if interval_error > 1e-8:
            raise ValueError(f'historical interval not reproduced: {interval_error}')
    return {'status': 'PASS', 'max_point_error_ap_points': max(errors.values()),
            'interval_max_error_ap_points': interval_error, 'tolerance_ap_points': 1e-8,
            'ap_recomputed_from_predictions': True, 'ap_cache_used_in_recomputation': False}


def build(root, audit_path, out, historical_bootstrap=None):
    root, audit_path, out = Path(root).resolve(), Path(audit_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'refusing to overwrite package: {out}')
    audit = json.loads(audit_path.read_text())
    selected = select_example(audit)
    audit_root = Path(audit['root'])
    def resolve(source):
        return root / Path(source).relative_to(audit_root)
    out.mkdir(parents=True)
    annotation_record = audit['dataset_inputs']['kitti']['annotation']
    annotation = bound_copy(resolve(annotation_record['path']), out, 'data/annotations.json', annotation_record['sha256'])
    class_record = audit['dataset_inputs']['kitti']['class_map']
    bound_copy(resolve(class_record['path']), out, 'data/class_map.json', class_record['sha256'])
    sample_row = selected['int8_clean']
    sources = {}
    for key in ('runner', 'preprocess', 'decoder'):
        record = sample_row['checks'][key + '_sha256']
        sources[key] = bound_copy(resolve(record['path']), out, 'sources/' + key + '.py', record['expected_sha256'])
    manifest = {'schema_version': 1, 'scope': {'dataset': 'kitti', 'model': 'yolo11m', 'n_images': 1197,
                'corruption': 'fog', 'severity': 1, 'clean_control': 'original_source_NOT_JPEG95',
                'role': 'fixed final-holdout reproduction example'},
                'annotations': annotation, 'evaluator_settings': EVALUATOR, 'arms': {}}
    expected = {'point_ap': {}, 'source_audit_sha256': sha(audit_path),
                'historical_evaluator_source_binding': 'not recorded; canonical COCO AP recomputation tested numerically'}
    for name in ARM_NAMES:
        row = selected[name]
        run = row['run_record']
        arm = {}
        checks = row['checks']['run_prediction_metric']['checks']
        hashes = {'run': checks['metric_run_bytes']['expected_sha256'],
                  'prediction': run['prediction_sha256'],
                  'input': sha(resolve(row['paths']['input'])),
                  'metric': row['checks']['metric_report']['expected_sha256']}
        for kind in ('run', 'prediction', 'input', 'metric'):
            ref = bound_copy(resolve(row['paths'][kind]), out, f'data/{name}_{kind}.json', hashes[kind])
            if kind != 'metric':
                arm[kind] = ref
        arm['settings'] = {'inference_settings': {'imgsz': 640, 'conf': .001, 'nms_iou': .7, 'max_det': 300},
                           'evaluator_settings': EVALUATOR, 'derivation': {'kind': 'normalized_source_backed',
                           'command_sha256': hashlib.sha256(run['command'].encode()).hexdigest(), 'sources': sources}}
        manifest['arms'][name] = arm
        expected['point_ap'][name] = 100 * row['metric_record']['stats']['AP']
    points = expected['point_ap']
    expected['point_delta_e'] = points['fp8_corrupt'] - points['int8_corrupt'] - points['fp8_clean'] + points['int8_clean']
    ids = json.loads((out / manifest['arms']['int8_clean']['input']['path']).read_text())['image_ids']
    seed, draws, historical = 202609072, 2000, []
    expected['delta_e_interval'] = None
    expected['schedule_semantics'] = 'new V4 finite-image sensitivity; not the historical interval'
    if historical_bootstrap is not None:
        components, excesses = [], []
        for precision in ('int8-entropy', 'fp8'):
            component_path = Path(historical_bootstrap) / f'yolo11m__{precision}__fog-s1.json'
            component = json.loads(component_path.read_text())
            if component['n_images'] != 1197 or component['n_boot'] != 2000:
                raise ValueError('historical bootstrap scope mismatch')
            short = 'int8' if precision.startswith('int8') else 'fp8'
            for condition in ('clean', 'corrupt'):
                name = short + '_' + condition
                binding = component['input_hashes']['quant_' + condition]
                if (binding['prediction_sha256'] != manifest['arms'][name]['prediction']['sha256']
                        or binding['input_record_sha256'] != manifest['arms'][name]['input']['sha256']):
                    raise ValueError('historical component prediction/input binding mismatch')
            cache = resolve(component['draw_cache']['path'])
            if sha(cache) != component['draw_cache']['sha256']:
                raise ValueError('historical baseline cache hash mismatch')
            with np.load(cache, allow_pickle=False) as data:
                if int(data['seed']) != component['seed'] or int(data['n_boot']) != 2000 or data['excess'].shape != (2000, 4):
                    raise ValueError('historical baseline metadata mismatch')
                excesses.append(data['excess'].copy())
            bound_copy(component_path, out, f'baseline/{precision}.json', sha(component_path))
            historical.append({'component_sha256': sha(component_path), 'draw_baseline_sha256': sha(cache)})
            components.append(component)
        if components[0]['seed'] != components[1]['seed']:
            raise ValueError('historical schedules differ between treatments')
        for condition in ('clean', 'corrupt'):
            if components[0]['input_hashes']['fp32_' + condition] != components[1]['input_hashes']['fp32_' + condition]:
                raise ValueError('historical shared reference differs; cancellation invalid')
        seed = components[0]['seed']
        expected['delta_e_interval'] = (100 * np.percentile((excesses[0] - excesses[1])[:, 0], [2.5, 50, 97.5])).tolist()
        expected['schedule_semantics'] = 'historical schedule regenerated from recorded seed using retained default_rng.choice sampler; numeric interval checked independently'
        expected['historical_baselines'] = historical
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(len(ids), size=len(ids), replace=True) for _ in range(draws)], dtype=np.int32)
    np.savez_compressed(out / 'schedule.npz', samples=samples, image_ids=np.asarray(ids), seed=seed)
    manifest['schedule'] = {'path': 'schedule.npz', 'sha256': sha(out / 'schedule.npz')}
    manifest['schedule_provenance'] = {'seed': seed, 'n_boot': draws, 'semantics': expected['schedule_semantics']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    validate_contract(out / 'manifest.json')
    for relative in ('bootstrap_format_contrast.py', 'paired_bootstrap.py', 'topic_c/__init__.py', 'topic_c/manifest.py'):
        source = root / 'src' / relative
        bound_copy(source, out, 'src/' + relative, sha(source))
    for source, relative in ((root / 'analysis/reproduce_v4_four_arm.py', 'reproduce_v4_four_arm.py'),
                             (Path(__file__).resolve(), 'package_v4_holdout_example.py')):
        bound_copy(source, out, relative, sha(source))
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pycocotools')}
    (out / 'requirements.txt').write_text(''.join(f'{key}=={value}\n' for key, value in versions.items()))
    expected['versions'] = versions
    expected['schedule_seed'] = seed
    (out / 'expected.json').write_text(json.dumps(expected, indent=2) + '\n')
    (out / 'README.md').write_text(
        '# Fixed final-holdout prediction-level example\n\n'
        'KITTI final 1,197 images, YOLO11m, fog severity 1. Historical clean uses **original-source bytes, not JPEG-95**. '
        'This distinction is deliberate; do not label this example codec-matched.\n\n'
        'Four full predictions, converted annotations and unchanged run/input records are evaluated without AP caches, '
        'engines, training, or images. Sources normalize recorded effective inference settings without editing raw runs. '
        'The historical evaluator source was not recorded; numerical reproduction under the vendored canonical evaluator '
        'does not retroactively establish that source identity. The run records are internally hash-bound, not independently timestamp-authenticated.\n\n'
        'Use Python 3.11 on Linux:\n\n```bash\npython -m pip install -r requirements.txt\n'
        'python reproduce_v4_four_arm.py --manifest manifest.json --out /tmp/v4_holdout_report.json --workers 4\n'
        'python package_v4_holdout_example.py --verify-report /tmp/v4_holdout_report.json --expected expected.json\n```\n\n'
        'Choose a new output filename on each invocation. The comparison tolerance is 1e-8 AP points. '
        + expected['schedule_semantics'] + '. The baseline component summaries were derived from historical caches only '
        'to create independent expected intervals; no AP cache is included or consumed by the reproduction command.\n\n'
        'This is a local, unpublished research package. KITTI-derived annotations remain subject to KITTI terms, '
        'not the code MIT license. Check redistribution permission before public deposit. '
        'No image data, checkpoints, engines or credentials are included. This example tests implementation, not 95% interval coverage.\n')
    inventory = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'SHA256.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return expected


def fresh_check(package, workers):
    package = Path(package).resolve()
    if not 1 <= workers <= 4:
        raise ValueError('fresh-check workers must be 1..4')
    archive = package.parent / 'example.zip'
    if archive.exists():
        raise FileExistsError('refusing to overwrite archive')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                stream.write(path, str(path.relative_to(package)))
    with tempfile.TemporaryDirectory(prefix='cviu-v4-holdout-') as temporary:
        fresh = Path(temporary)
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(fresh)
        inventory = json.loads((fresh / 'SHA256.json').read_text())
        if any(sha(fresh / name) != value for name, value in inventory.items()):
            raise ValueError('clean extraction hash validation failed')
        output = package.parent / 'fresh_report.json'
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        subprocess.run([sys.executable, str(fresh / 'reproduce_v4_four_arm.py'), '--manifest', str(fresh / 'manifest.json'),
                        '--out', str(output), '--workers', str(workers)], cwd=fresh, env=env, check=True)
        result = compare_report(json.loads(output.read_text()), json.loads((fresh / 'expected.json').read_text()))
        result.update(archive_sha256=sha(archive), report_sha256=sha(output), clean_extraction_files_verified=len(inventory),
                      actual_versions={key: importlib.metadata.version(key) for key in ('numpy', 'pycocotools')})
        (package.parent / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--historical-bootstrap', type=Path)
    parser.add_argument('--fresh-check', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-report', type=Path)
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    if args.verify_report:
        if not args.expected:
            parser.error('--expected required for report comparison')
        print(json.dumps(compare_report(json.loads(args.verify_report.read_text()), json.loads(args.expected.read_text())), indent=2))
        return
    if any(value is None for value in (args.root, args.audit, args.out)):
        parser.error('--root, --audit and --out required for packaging')
    build(args.root, args.audit, args.out, args.historical_bootstrap)
    print('PACKAGE READY ' + str(args.out), flush=True)
    if args.fresh_check:
        print(json.dumps(fresh_check(args.out, args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
