#!/usr/bin/env python3
"""Build an allowlisted local reviewer supplement, or verify its extracted hashes.

No upload is performed. Dataset-derived example payloads need separate rights
review before distribution. The archive supplements, not replaces, v2.1.0.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


SUBMISSION_SOURCE_FILES = (
    'main.tex', 'supplement.tex', 'references.bib',
    'cas-dc.cls', 'cas-common.sty', 'elsarticle-num.bst',
    'README_OVERLEAF.txt',
)
SUBMISSION_UPLOAD_FILES = (
    '01_CVIU_main_revised.pdf',
    '02_CVIU_supplement_revised.pdf',
    '03_highlights_CVIU.txt',
    '04_graphical_abstract_CVIU.pdf',
    '04_graphical_abstract_CVIU.png',
    '05_CVIU_cover_letter.pdf',
    'REVIEWER_EVIDENCE_README.md',
    'cover_letter_CVIU.tex',
    'graphical_abstract_CVIU.tex',
)
SUBMISSION_BINDING_NAME = 'SUBMISSION_BINDING.json'
LEGACY_EDITABLE_MANUSCRIPT = {
    'Thuan_paper_3_CVIU_revised/main.tex',
    'Thuan_paper_3_CVIU_revised/supplement.tex',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('only root-relative paths are allowed')
    actual = root / path
    if not actual.resolve().is_relative_to(root.resolve()):
        raise ValueError('only root-relative paths are allowed')
    return actual


def active_submission_files(root, package_name):
    """Return the explicit active-publication inventory for a package root."""
    root = root.resolve()
    package = safe_path(root, package_name)
    source = package / 'source'
    package_relative = package.relative_to(root)
    names = {str(package_relative / name) for name in SUBMISSION_UPLOAD_FILES}
    names.update(str(package_relative / 'source' / name) for name in SUBMISSION_SOURCE_FILES)

    def require_file(path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'missing active submission dependency: {path.relative_to(root)}')

    for name in names:
        require_file(safe_path(root, name))

    input_pattern = re.compile(r'\\input\{([^}]+)\}')
    image_pattern = re.compile(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}')
    # TeX resolves these paths from each entry point's compilation directory.
    # Every discovered dependency must still live under the bounded source tree.
    pending = [
        (source / 'main.tex', source),
        (source / 'supplement.tex', source),
        (package / 'cover_letter_CVIU.tex', package),
        (package / 'graphical_abstract_CVIU.tex', package),
    ]
    scanned = set()
    while pending:
        tex, compile_root = pending.pop()
        scan_key = (tex, compile_root)
        if scan_key in scanned:
            continue
        scanned.add(scan_key)
        text = tex.read_text(encoding='utf-8')
        for raw in input_pattern.findall(text):
            relative = Path(raw)
            if relative.is_absolute() or '..' in relative.parts or '\\' in raw:
                raise ValueError(f'active submission dependency must remain under source: {raw}')
            dependency = compile_root / relative
            if not dependency.suffix and not dependency.is_file():
                dependency = dependency.with_suffix('.tex')
            if not dependency.resolve().is_relative_to(source.resolve()):
                raise ValueError(f'active submission dependency must remain under source: {raw}')
            require_file(dependency)
            names.add(str(dependency.relative_to(root)))
            if dependency.suffix == '.tex':
                pending.append((dependency, compile_root))
        for raw in image_pattern.findall(text):
            relative = Path(raw)
            if relative.is_absolute() or '..' in relative.parts or '\\' in raw:
                raise ValueError(f'active submission image must remain under source: {raw}')
            dependency = compile_root / relative
            if compile_root == source and not dependency.is_file():
                dependency = source / 'figures' / relative
            if not dependency.resolve().is_relative_to(source.resolve()):
                raise ValueError(f'active submission image must remain under source: {raw}')
            require_file(dependency)
            names.add(str(dependency.relative_to(root)))
    return names


def bind_submission(root, package_name, evidence_names):
    """Replace legacy editable roots with an explicitly bound active version."""
    root = root.resolve()
    package = safe_path(root, package_name)
    package_relative = package.relative_to(root).as_posix()
    active = active_submission_files(root, package_relative)
    names = set(evidence_names) - LEGACY_EDITABLE_MANUSCRIPT
    names.update(active)
    historical = sorted(
        name for name in names
        if name.startswith('Thuan_paper_3_CVIU_revised/')
    )
    source_prefix = package_relative + '/source/'
    binding = {
        'schema': 'cviu-v4-submission-binding-v1',
        'active_submission_package': package_relative,
        'active_manuscript_sources': [
            source_prefix + 'main.tex', source_prefix + 'supplement.tex',
        ],
        'active_source_files': sorted(name for name in active if name.startswith(source_prefix)),
        'upload_assets': sorted(active - {name for name in active if name.startswith(source_prefix)}),
        'active_files_sha256': {name: digest(safe_path(root, name)) for name in sorted(active)},
        'historical_manuscript_artifacts': historical,
        'excluded_stale_editable_sources': sorted(LEGACY_EDITABLE_MANUSCRIPT),
        'verification': {'v4_numerical_regeneration_required': True},
    }
    return names, binding


def write_archive(root, names, output, *, binding=None):
    paths = {name: safe_path(root, name) for name in sorted(set(names))}
    hashes = {name: digest(path) for name, path in paths.items()}
    binding_bytes = None
    if binding is not None:
        if SUBMISSION_BINDING_NAME in paths:
            raise ValueError(f'reserved archive member: {SUBMISSION_BINDING_NAME}')
        binding_bytes = (json.dumps(binding, indent=2, sort_keys=True) + '\n').encode()
        hashes[SUBMISSION_BINDING_NAME] = hashlib.sha256(binding_bytes).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, path in paths.items():
            z.write(path, name)
        if binding_bytes is not None:
            z.writestr(SUBMISSION_BINDING_NAME, binding_bytes)
        z.writestr('EVIDENCE_SHA256.json', json.dumps(hashes, indent=2) + '\n')
    return len(hashes)


def verify_files(root):
    hashes = json.loads((root / 'EVIDENCE_SHA256.json').read_text())
    entries = list(root.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('evidence symlinks are not permitted')
    actual = {str(p.relative_to(root)) for p in entries if p.is_file()}
    if actual != set(hashes) | {'EVIDENCE_SHA256.json'}:
        raise ValueError('evidence inventory mismatch: missing or extra files')
    for name, expected in hashes.items():
        if digest(safe_path(root, name)) != expected:
            raise ValueError(f'evidence hash mismatch: {name}')
    return len(hashes)


def verify_submission_binding(root, package_name, canonical_names):
    """Verify the active-versus-historical meaning of a hash-valid archive."""
    root = root.resolve()
    marker = root / SUBMISSION_BINDING_NAME
    if not marker.is_file():
        raise ValueError('active submission binding is missing')
    inventory = json.loads((root / 'EVIDENCE_SHA256.json').read_text())
    if SUBMISSION_BINDING_NAME not in inventory:
        raise ValueError('active submission binding is outside evidence inventory')
    archived_names = set(inventory) - {SUBMISSION_BINDING_NAME}
    canonical_names = set(canonical_names)
    if archived_names != canonical_names:
        raise ValueError('canonical evidence inventory mismatch: missing or extra files')
    expected_names, expected = bind_submission(root, package_name, canonical_names)
    if expected_names != canonical_names:
        raise ValueError('canonical evidence inventory contains stale editable manuscript')
    document = json.loads(marker.read_text())
    if document != expected:
        raise ValueError('active manuscript submission binding mismatch')
    return len(expected['active_files_sha256'])


def bound_summary_sources(root, summary_path):
    bindings = json.loads(summary_path.read_text())['source_sha256']
    for name, expected in bindings.items():
        if digest(safe_path(root, name)) != expected:
            raise ValueError(f'upstream hash mismatch: {name}')
    return set(bindings)


def completed_v4_files(root, directory, marker, required, field=None):
    """A completion marker is accepted only with every bound file present."""
    marker_path = safe_path(root, str(directory / marker))
    if not marker_path.is_file():
        raise ValueError(f'V4 incomplete: missing completion marker {directory / marker}')
    document = json.loads(marker_path.read_text())
    bindings = document[field] if field else document
    if not isinstance(bindings, dict) or not required.issubset(bindings):
        raise ValueError('V4 incomplete completion inventory')
    names = {str(directory / marker)}
    for relative, expected in bindings.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('V4 completion path must be relative')
        if Path(relative).suffix not in {'.json', '.jsonl', '.npz', '.csv', '.tex', '.md'} or 'prediction' in relative.lower():
            raise ValueError('V4 completion exceeds compact artifact allowlist')
        name = str(directory / relative)
        path = safe_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'V4 completion file hash mismatch: {name}')
        names.add(name)
    return names


def v4_clean_files(root):
    """Recompute controlled S summaries from the 32 terminal AP vectors."""
    import numpy as np
    directory = Path('outputs/analysis/cviu_v4/clean_control')
    arms = ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')
    blocks = {f'{dataset}_{model}' for dataset in ('voc', 'kitti') for model in ('yolo11n', 'yolo11m', 'yolo11x')}
    blocks |= {'tt100k_yolo11n', 'tt100k_yolo11x'}
    expected = {'registry.json', 'points.json', 'bootstrap_jobs.json', 'summary.json'}
    expected |= {f'bootstrap/{block}__{arm}{suffix}' for block in blocks for arm in arms for suffix in ('.npz', '.json')}
    expected |= {f'bootstrap/{dataset}_schedule.npz' for dataset in ('voc', 'kitti', 'tt100k')}
    names = completed_v4_files(root, directory, 'complete.json', expected)
    if names != {str(directory / name) for name in expected | {'complete.json'}}:
        raise ValueError('V4 clean completion exceeds exact allowlist')
    base = root / directory
    registry = json.loads((base / 'registry.json').read_text())
    seal = json.loads((base / 'registry.complete.json').read_text())
    if seal != {'sha256': digest(base / 'registry.json')}:
        raise ValueError('V4 registry hash mismatch')
    names.add(str(directory / 'registry.complete.json'))
    summary = json.loads((base / 'summary.json').read_text())
    points = json.loads((base / 'points.json').read_text())
    jobs = json.loads((base / 'bootstrap_jobs.json').read_text())
    if (summary.get('n_boot') != 2000 or registry.get('n_boot') != 2000
            or summary.get('registry_sha256') != digest(base / 'registry.json')
            or len(summary['blocks']) != 8 or {row['block'] for row in summary['blocks']} != blocks
            or len(registry['blocks']) != 8
            or {f"{b['dataset']}_{b['model']}" for b in registry['blocks']} != blocks
            or set(points) != {f'{block}__{arm}' for block in blocks for arm in arms}
            or len(jobs) != 32 or {job['name'] for job in jobs} != set(points)):
        raise ValueError('V4 clean summary grid incomplete or changed')
    primary_draws, primary_points = [], []
    jobs_by_name = {job['name']: job for job in jobs}
    for row in summary['blocks']:
        block = row['block']
        group = 'diagnostic' if block.startswith('tt100k_') else 'primary'
        if row.get('group') != group:
            raise ValueError('V4 diagnostic/primary grouping changed')
        arrays = []
        for arm in arms:
            name = f'{block}__{arm}'
            job = jobs_by_name[name]
            identity = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            marker = json.loads((base / f'bootstrap/{name}.json').read_text())
            if marker != {'identity': identity, 'npz_sha256': digest(base / f'bootstrap/{name}.npz')}:
                raise ValueError('V4 AP vector job identity/hash mismatch')
            schedule = base / 'bootstrap' / (block.split('_', 1)[0] + '_schedule.npz')
            if job.get('n_boot') != 2000 or job.get('schedule_sha256') != digest(schedule):
                raise ValueError('V4 common schedule identity mismatch')
            with np.load(schedule, allow_pickle=False) as schedule_file:
                sample = schedule_file['samples']
                ids = schedule_file['image_ids'].tolist()
                if (job.get('image_ids') != ids or sample.dtype.kind not in 'iu'
                        or sample.shape != (2000, len(ids)) or np.any(sample < 0) or np.any(sample >= len(ids))):
                    raise ValueError('V4 common schedule/image order mismatch')
            with np.load(base / f'bootstrap/{block}__{arm}.npz', allow_pickle=False) as archive:
                if set(archive.files) != {'ap'}:
                    raise ValueError('V4 AP vector fields changed')
                values = archive['ap'].copy()
            if values.shape != (2000,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError('V4 AP vector is invalid')
            arrays.append(values)
            if not np.isclose(row['ap_points'][arm], 100 * points[f'{block}__{arm}'], rtol=0, atol=1e-8):
                raise ValueError('V4 clean AP summary differs from point ledger')
        s_draws = 100 * (arrays[1] - arrays[0] - arrays[3] + arrays[2])
        p = [points[f'{block}__{arm}'] for arm in arms]
        point = 100 * (p[1] - p[0] - p[3] + p[2])
        if (not np.isclose(point, row['S'], rtol=0, atol=1e-8)
                or not np.allclose(np.percentile(s_draws, [2.5, 50, 97.5]), row['interval'], rtol=0, atol=1e-8)):
            raise ValueError('V4 clean S summary/interval differs from paired AP vectors')
        if group == 'primary':
            primary_draws.append(s_draws)
            primary_points.append(point)
    macro = summary['primary_macro']
    if (not np.isclose(np.mean(primary_points), macro['S'], rtol=0, atol=1e-8)
            or not np.allclose(np.percentile(np.mean(primary_draws, axis=0), [2.5, 50, 97.5]), macro['interval'], rtol=0, atol=1e-8)):
        raise ValueError('V4 six-block macro does not preserve common draws')
    return names


def v4_mapped_source(root, value):
    """Map only the recorded project root; never dereference arbitrary hosts."""
    path = Path(value)
    for prefix in (root.resolve(), Path('/home/thuan/topic_c_ivc')):
        if path.is_absolute() and path.is_relative_to(prefix):
            return str(path.relative_to(prefix))
    if not path.is_absolute():
        return str(path)
    raise ValueError(f'V4 source outside recorded project: {value}')


def v4_recovered_draws(root, summary):
    """Check all recovered CIs against paired terminal clean and DeltaE draws."""
    import numpy as np
    from build_v4_holdout_table import historical_schedule
    base = root / 'outputs/analysis/cviu_v4/holdout_synthesis/recovered'
    registry = json.loads((base / 'recovery_registry.json').read_text())
    labels = [f"{row['dataset']}_{row['model']}" for row in summary['blocks']]
    expected = {f'{d}_{m}' for d in ('voc', 'kitti') for m in ('yolo11n', 'yolo11m', 'yolo11x')}
    jobs = {job['name']: job for job in registry['jobs']}
    if (len(labels) != 6 or set(labels) != expected or registry['expected_jobs'] != 12
            or len(registry['jobs']) != 12
            or set(jobs) != {f'{label}__{p}' for label in expected for p in ('int8-entropy', 'fp8')}):
        raise ValueError('V4 recovered job/block grid changed')
    matrices = {}
    for metric in ('delta_e', 'corrupted_gap'):
        with np.load(base / f'paired_{metric}_draws.npz', allow_pickle=False) as archive:
            if set(archive.files) != {f'block_{metric}_native', f'macro_{metric}_native', 'block_labels'}:
                raise ValueError('V4 recovered paired vector fields changed')
            draws = archive[f'block_{metric}_native'].copy()
            macro = archive[f'macro_{metric}_native']
            if (archive['block_labels'].tolist() != labels or draws.shape != (6, 2000)
                    or not np.isfinite(draws).all() or macro.shape != (2000,)
                    or not np.allclose(macro, draws.mean(axis=0), atol=1e-12, rtol=0)):
                raise ValueError('V4 recovered paired vector labels/mean changed')
        matrices[metric] = draws
    for index, (label, row) in enumerate(zip(labels, summary['blocks'])):
        schedule = base / 'schedules' / (row['dataset'] + '_historical.npz')
        with np.load(schedule, allow_pickle=False) as archive:
            ids = archive['image_ids'].tolist()
            if not np.array_equal(archive['samples'], historical_schedule(ids, int(archive['seed']))):
                raise ValueError('V4 recovered historical schedule changed')
        aps = []
        for precision in ('int8-entropy', 'fp8'):
            job = jobs[f'{label}__{precision}']
            path = base / 'clean_ap' / (job['name'] + '.npz')
            identity = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if (json.loads(path.with_suffix('.json').read_text()) != {'identity': identity, 'npz_sha256': digest(path)}
                    or job['n_boot'] != 2000 or job['image_ids'] != ids or job['schedule_sha256'] != digest(schedule)):
                raise ValueError('V4 recovered clean AP job identity/schedule changed')
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {'ap'}:
                    raise ValueError('V4 recovered clean AP fields changed')
                ap = archive['ap'].copy()
            if ap.shape != (2000,) or not np.isfinite(ap).all() or np.any(ap < 0) or np.any(ap > 1):
                raise ValueError('V4 recovered clean AP vector invalid')
            aps.append(ap)
        clean_gap = aps[1] - aps[0]
        if not np.allclose(matrices['corrupted_gap'][index], matrices['delta_e'][index] + clean_gap, atol=1e-12, rtol=0):
            raise ValueError('V4 recovered gap differs from paired DeltaE plus clean gap')
        for metric, values in [('clean_gap', clean_gap)] + [(m, v[index]) for m, v in matrices.items()]:
            if not np.allclose(np.percentile(100 * values, [2.5, 50, 97.5]), row[metric + '_percentile95_ap_points'], atol=1e-8, rtol=0):
                raise ValueError('V4 recovered block interval differs from paired vectors')
    for metric, values in matrices.items():
        if not np.allclose(np.percentile(100 * values.mean(axis=0), [2.5, 50, 97.5]),
                           summary['primary_macro'][metric + '_percentile95_ap_points'], atol=1e-8, rtol=0):
            raise ValueError('V4 recovered macro interval differs from paired vectors')
    return matrices['delta_e']


def verify_v4_numerical(root):
    """Rebuild historical DeltaE, recovered gaps, and controlled S without AP evaluation."""
    import numpy as np
    from build_v4_holdout_table import build
    root = root.resolve()
    v4_clean_files(root)
    summary = json.loads((root / 'outputs/analysis/cviu_v4/holdout_synthesis/recovered/holdout_synthesis.json').read_text())
    recorded = v4_recovered_draws(root, summary)
    ledger = safe_path(root, v4_mapped_source(root, summary['sources'][0]['path']))
    rebuilt, draws, labels = build(root, root, ledger_path=ledger)
    if (labels != [f"{row['dataset']}_{row['model']}" for row in summary['blocks']]
            or not np.allclose(recorded, draws, atol=1e-12, rtol=0)):
        raise ValueError('V4 historical DeltaE differs from retained paired source draws')
    for actual, expected in zip(rebuilt['blocks'] + [rebuilt['primary_macro']], summary['blocks'] + [summary['primary_macro']]):
        for key, value in actual.items():
            if isinstance(value, (int, float, list)) and value is not None:
                if not np.allclose(value, expected[key], atol=1e-8, rtol=0):
                    raise ValueError(f'V4 historical summary differs from retained sources: {key}')


def v4_example_files(root, holdout, *, verified_extraction=False):
    """Verify the compact example from its ZIP or its exact unpacked payload."""
    root = root.resolve()
    example = Path('outputs/analysis/cviu_v4/holdout_example')
    base = root / example
    verification = json.loads((base / 'verification.json').read_text())
    report = json.loads((base / 'fresh_report.json').read_text())
    inventory = json.loads((base / 'package/SHA256.json').read_text())
    expected_example = json.loads((base / 'package/expected.json').read_text())
    archive_sha256 = verification.get('archive_sha256')
    if (verification.get('status') != 'PASS'
            or verification.get('ap_cache_used_in_recomputation') is not False
            or verification.get('report_sha256') != digest(base / 'fresh_report.json')
            or verification.get('clean_extraction_files_verified') != len(inventory)
            or not isinstance(archive_sha256, str)
            or re.fullmatch(r'[0-9a-f]{64}', archive_sha256) is None):
        raise ValueError('V4 fresh holdout example verification binding mismatch')
    package_entries = list((base / 'package').rglob('*'))
    if any(path.is_symlink() for path in package_entries):
        raise ValueError('V4 example payload symlinks are not permitted')
    actual = {str(path.relative_to(base / 'package')) for path in package_entries if path.is_file()}
    if actual != set(inventory) | {'SHA256.json'}:
        raise ValueError('V4 example unpacked inventory mismatch')
    if (report.get('manifest_sha256') != digest(base / 'package/manifest.json')
            or report.get('n_images') != 1197 or report.get('n_boot') != 2000
            or expected_example.get('source_audit_sha256') != digest(safe_path(root, holdout))):
        raise ValueError('V4 example manifest/audit/scope binding mismatch')
    from package_v4_holdout_example import compare_report
    compare_report(report, expected_example)
    archive_path = base / 'example.zip'
    if not archive_path.is_file() and not verified_extraction:
        raise ValueError('V4 holdout example original archive is required for source/build selection')
    archive = None
    if archive_path.is_file():
        if digest(archive_path) != archive_sha256:
            raise ValueError('V4 fresh holdout example original archive hash mismatch')
        archive = zipfile.ZipFile(archive_path)
        if set(archive.namelist()) != set(inventory) | {'SHA256.json'}:
            archive.close()
            raise ValueError('V4 example ZIP inventory mismatch')
    try:
        names = set()
        for relative, expected in inventory.items():
            if Path(relative).suffix not in {'.json', '.npz', '.py', '.txt', '.md'}:
                raise ValueError('V4 example payload exceeds reproduction allowlist')
            name = str(example / 'package' / relative)
            if digest(safe_path(root, name)) != expected:
                raise ValueError('V4 example unpacked payload hash mismatch')
            if archive is not None and hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError('V4 example ZIP payload hash mismatch')
            names.add(name)
    finally:
        if archive is not None:
            archive.close()
    names.update(str(example / relative) for relative in ('package/SHA256.json', 'verification.json', 'fresh_report.json'))
    return names


def v4_selected_files(root, *, verified_extraction=False):
    root = root.resolve()
    names = v4_clean_files(root)  # Gate before reading any legacy or partial output.
    recovery = Path('outputs/analysis/cviu_v4/holdout_synthesis/recovered')
    names.update(completed_v4_files(root, recovery, 'recovery.complete.json',
                 {'recovery_registry.json', 'completion.json', 'paired_corrupted_gap_draws.npz'}, 'files_sha256'))
    marker = json.loads((root / recovery / 'recovery.complete.json').read_text())
    if (marker.get('complete') is not True or marker.get('clean_bootstrap_arms') != 12
            or marker.get('corrupted_gap_intervals_recovered') != 6 or marker.get('macro_interval_recovered') is not True):
        raise ValueError('V4 holdout recovery is incomplete')
    names.update(completed_v4_files(root, recovery, 'completion.json',
                 {'holdout_synthesis.json', 'holdout_synthesis.csv', 'holdout_synthesis.tex', 'paired_delta_e_draws.npz', 'README.md'}, 'files_sha256'))
    complete = json.loads((root / recovery / 'completion.json').read_text())
    if (complete.get('blocks') != 6 or complete.get('corruption_cells') != 72
            or complete.get('corrupted_gap_intervals_missing') != 0 or complete.get('delta_e_intervals_verified') != 6):
        raise ValueError('V4 recovered holdout evidence has missing endpoints')
    recovered = json.loads((root / recovery / 'holdout_synthesis.json').read_text())
    if recovered.get('recovery_registry_sha256') != digest(root / recovery / 'recovery_registry.json'):
        raise ValueError('V4 recovered summary registry hash mismatch')
    for record in recovered['sources']:
        if record['hash_kind'] != 'file_bytes':
            raise ValueError('unsupported V4 compact synthesis hash semantics')
        name = v4_mapped_source(root, record['path'])
        if digest(safe_path(root, name)) != record['sha256']:
            raise ValueError(f'V4 synthesis source hash mismatch: {name}')
        if Path(name).suffix not in {'.json', '.npz'} or '/predictions/' in name:
            raise ValueError('V4 synthesis source exceeds compact allowlist')
        names.add(name)
    verify_v4_numerical(root)
    # These are remote read-only audit reports. Their engine/image hashes remain
    # metadata; packaging does not pretend to re-audit absent large artifacts.
    holdout = 'outputs/analysis/cviu_v4/holdout_provenance/holdout_provenance.json'
    audit = json.loads((root / holdout).read_text())
    if audit.get('counts', {}).get('treatments') != 12 or not audit.get('status_semantics'):
        raise ValueError('V4 holdout audit scope/status semantics missing')
    names.add(holdout)
    tt_name = 'outputs/analysis/cviu_v4/tt100k_annotations/complete_audit/summary.json'
    tt = json.loads((root / tt_name).read_text())
    ledger = tt['ledger']
    if ledger.get('rows') != 16811 or digest(safe_path(root, ledger['path'])) != ledger['sha256']:
        raise ValueError('V4 TT100K completed image ledger missing or changed')
    names.update({tt_name, ledger['path']})
    names.update(v4_example_files(root, holdout, verified_extraction=verified_extraction))
    # Include exact scientific sources, not all files in any workspace folder.
    sources = ['analysis/audit_v4_holdout_provenance.py', 'analysis/audit_v4_tt100k_annotations.py',
               'analysis/reproduce_v4_four_arm.py', 'analysis/package_v4_holdout_example.py', 'analysis/build_v4_holdout_table.py',
               'analysis/build_v4_publication_tables.py', 'analysis/build_cviu_revision_artifacts.py',
               'tests/test_v4_publication_tables.py', 'tests/test_realization_publication_tables.py',
               'analysis/build_paper_artifacts.py', 'tests/test_v4_graphical_abstract.py',
               'src/run_v4_clean_control.py', 'src/materialize_codec_control.py', 'src/coco_infer_trt.py',
               'src/topic_c/coco_data.py', 'src/topic_c/yolo_decode.py', 'src/paired_bootstrap.py',
               'src/run_controlled_postprocess.py', 'src/run_fixed_universe_sensitivity.py', 'src/accelerate_shared_mask_bootstrap_v3.py',
               'src/fixed_universe_bootstrap.py', 'src/topic_c/shared_mask_pilot.py', 'src/pilot_registry.py',
               'src/run_confirmatory_bootstrap.py', 'configs/confirmatory_bootstrap_v1.json',
               'configs/cviu_v4_clean_control_v1.json', 'Thuan_paper_3_CVIU_revised/CVIU_V4_Ke_hoach_nang_cap.md',
               'Thuan_paper_3_CVIU_revised/REVIEWER_EVIDENCE_README.md']
    sources += [f'tests/test_v4_{name}.py' for name in ('holdout_provenance', 'tt100k_annotations', 'protocol_contract',
                                                     'holdout_example', 'holdout_table', 'clean_control')]
    registry = json.loads((root / 'outputs/analysis/cviu_v4/clean_control/registry.json').read_text())
    frozen = {}
    recovery_registry = json.loads((root / recovery / 'recovery_registry.json').read_text())
    bindings = list(registry['files'].items()) + list(recovery_registry['source_files_sha256'].items())
    for value, expected in bindings:
        try:
            name = v4_mapped_source(root, value)
        except ValueError:
            continue  # Environment and large artifact inventories remain metadata-only.
        if name in sources or name in ('src/topic_c/manifest.py',):
            if name in frozen and frozen[name] != expected:
                raise ValueError(f'V4 frozen scientific source bindings disagree: {name}')
            frozen[name] = expected
    for name in sources:
        path = safe_path(root, name)
        if not path.is_file() or (name in frozen and digest(path) != frozen[name]):
            raise ValueError(f'V4 scientific source missing or changed since freeze: {name}')
        names.add(name)
    # The original-path realization ledger feeds the corrected area/height table.
    # Preserve its historical bytes and the retained completion binding unchanged.
    completion_name = 'paper/confirmatory_evidence/master_completion.json'
    completion = json.loads((root / completion_name).read_text())
    realization = completion['validated_components']['realization_analysis']
    realization_name = v4_mapped_source(root, realization['path'])
    if (realization_name != 'outputs/reports/corruption_realization_analysis_v1.json'
            or digest(safe_path(root, realization_name)) != realization['sha256']):
        raise ValueError('V4 realization publication source hash mismatch')
    names.update({completion_name, realization_name})
    return names


def selected_files(root, *, include_v4=False, submission_package=None, verified_extraction=False):
    names = v4_selected_files(root, verified_extraction=verified_extraction) if include_v4 else set()
    tide_dir = Path('outputs/analysis/cviu_novelty_v2/tide')
    complete = json.loads((root / tide_dir / 'complete.json').read_text())
    names.update(str(tide_dir / n) for n in ('complete.json', 'summary.json', 'publication_audit.json',
                                           'condition_records.csv', 'paired_error_interactions.csv'))
    names.update(str(tide_dir / 'records' / n) for n in complete['record_hashes'])
    for folder in ('outputs/analysis/pairing_covariance_v1', 'outputs/analysis/holdout_interpretation_v1'):
        names.update(str(Path(folder) / n) for n in ('summary.json', 'cells.csv'))
    names.update(str(Path('outputs/analysis/shared_mask_pilot_v3') / n)
                 for n in ('point_summary.json', 'point_summary.json.complete',
                           'bootstrap_summary.json', 'bootstrap_summary.json.complete'))
    if not include_v4:
        example_dir = Path('artifacts/cviu_four_arm_example_v1')
        example = json.loads((root / example_dir / 'example.json').read_text())
        names.add(str(example_dir / 'example.json'))
        names.update(str(example_dir / n) for n in example['files'])
    cov = json.loads((root / 'outputs/analysis/pairing_covariance_v1/summary.json').read_text())
    attachment = json.loads((root / 'outputs/analysis/shared_mask_pilot_v3/bootstrap_summary.json').read_text())
    for bindings in (cov['source_sha256'], attachment['source_artifacts_sha256']):
        for name, expected in bindings.items():
            if digest(safe_path(root, name)) != expected:
                raise ValueError(f'upstream hash mismatch: {name}')
            names.add(name)
    primary = json.loads((root / 'configs/ivc_format_contrast_v1.json').read_text())
    sources = {d['dataset']: d for d in primary['datasets']}
    for record_name in complete['record_hashes']:
        p = root / tide_dir / 'records' / record_name
        r = json.loads(p.read_text())
        key = 'clean_source_attempt' if r['corruption'] == 'clean' else 'corruption_source_attempt'
        name = str(Path('manifests/runs') / sources[r['dataset']][key] / p.name)
        if digest(safe_path(root, name)) != r['run_record_sha256']:
            raise ValueError(f'primary run hash mismatch: {name}')
        names.add(name)
    names.update(['LICENSE', 'configs/cviu_novelty_v1.json', 'configs/ivc_format_contrast_v1.json',
                  'configs/shared_mask_pilot_v3.json',
                  'paper/confirmatory_evidence/untouched_holdout_analysis.json',
                  'outputs/analysis/cviu_novelty_v2/four_arm_reproduction.json',
                  'outputs/analysis/cviu_novelty_v2/four_arm_reproduction.verification.json',
                  'analysis/build_reviewer_evidence.py', 'analysis/build_holdout_interpretation.py',
                  'analysis/build_tide_corrective_tables.py', 'analysis/analyze_pairing_covariance.py',
                  'src/run_tide_error_decomposition.py', 'src/bootstrap_format_contrast.py',
                  'src/topic_c/manifest.py', 'src/topic_c/__init__.py',
                  'tests/test_holdout_interpretation.py', 'tests/test_reviewer_evidence.py',
                  'tests/test_cviu_covariance_audit.py', 'tests/test_tide_error_decomposition.py',
                  'Thuan_paper_3_CVIU_revised/REVIEWER_EVIDENCE_README.md'])
    codec = Path('outputs/analysis/codec_interaction_v1')
    names.update(str(codec / n) for n in ('summary.json', 'blocks.csv', 'cells.csv'))
    names.update(bound_summary_sources(root, root / codec / 'summary.json'))
    # Exact clean metric/run records used by the original/J95 component audit.
    prefixes = (
        ('', 'coco_uniform_p0_v1', 'clean'),
        ('', 'codec_control_p0_v1', 'codec_control'),
        ('artifacts/four_dataset_pilot_v1', 'voc_pilot_117_v1', 'clean'),
        ('artifacts/four_dataset_pilot_v1', 'kitti_pilot_117_v1', 'clean'),
        ('artifacts/four_dataset_pilot_v1', 'tt100k_pilot_117_v1', 'clean'),
    )
    for prefix, attempt, corruption in prefixes:
        for metric in (root / prefix / 'outputs/metrics' / attempt).glob('*.json'):
            record = json.loads(metric.read_text())
            if record.get('corruption') != corruption or record.get('severity') != 0:
                continue
            run = root / prefix / 'manifests/runs' / attempt / metric.name
            if digest(run) != record['run_record_sha256']:
                raise ValueError(f'clean run binding mismatch: {run}')
            names.update(str(p.relative_to(root)) for p in (metric, run))
    names.update([
        'analysis/build_codec_interaction.py', 'tests/test_codec_interaction.py',
        'analysis/build_cviu_contract_figures.py', 'tests/test_cviu_revision_contract.py',
        'analysis/validate_cviu_paper_package.py',
        'paper/generated/direct_format_contrast_macro.csv',
        'paper/generated/direct_absolute_guardrail.csv',
        'Thuan_paper_3_CVIU_revised/main.tex',
        'Thuan_paper_3_CVIU_revised/supplement.tex',
        'Thuan_paper_3_CVIU_revised/generated/codec_interaction_summary.tex',
        'Thuan_paper_3_CVIU_revised/generated/codec_interaction_values.tex',
    ])
    treatment = Path('outputs/analysis/treatment_verification_v1')
    manifest = json.loads((root / treatment / 'manifest.json').read_text())
    for record in manifest['compact_evidence_sources']:
        if record['hash_kind'] != 'file_bytes':
            raise ValueError('unsupported treatment source hash semantics')
        name = record['path']
        if digest(safe_path(root, name)) != record['sha256']:
            raise ValueError(f'treatment source hash mismatch: {name}')
        names.add(name)
    names.update(str(treatment / n) for n in ('manifest.json', 'summary.json'))
    names.add('Thuan_paper_3_CVIU_revised/generated/treatment_verification_summary.tex')
    # Include exactly the compact sources used by the two completed follow-ups.
    # Recompute the reported draw-level values before accepting their bindings.
    from audit_completed_followups import audit
    names.update(audit(root)['verified_files'])
    names.update(['analysis/audit_completed_followups.py', 'tests/test_completed_followups.py'])
    if submission_package is not None:
        names, _ = bind_submission(root, submission_package, names)
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--include-v4', action='store_true', help='require completed V4 results; replace historical example with verified final holdout example')
    p.add_argument('--submission-package', type=Path,
                   help='root-relative active submission package to bind (implies --include-v4)')
    a = p.parse_args()
    include_v4 = a.include_v4 or a.submission_package is not None
    if a.verify:
        count = verify_files(a.root)
        bound = 0
        if a.submission_package is not None:
            canonical = selected_files(
                a.root, include_v4=True, submission_package=a.submission_package,
                verified_extraction=True,
            )
            bound = verify_submission_binding(a.root, a.submission_package, canonical)
        elif include_v4:
            verify_v4_numerical(a.root)
        message = f'PASS: {count} evidence file hashes'
        if a.submission_package is not None:
            message += f'; {bound} active submission files bound'
        if include_v4:
            message += '; V4 numerical regeneration'
        print(message)
    else:
        if a.output is None:
            p.error('--output required for build')
        names = selected_files(a.root, include_v4=include_v4, submission_package=a.submission_package)
        binding = None
        if a.submission_package is not None:
            names, binding = bind_submission(a.root, a.submission_package, names)
        count = write_archive(a.root, names, a.output, binding=binding)
        print(f'Created {a.output}: {count} allowlisted evidence files')


if __name__ == '__main__':
    main()
