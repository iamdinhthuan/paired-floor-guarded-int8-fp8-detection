import zipfile
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
import build_reviewer_evidence as package


def submission_fixture(root):
    source = root / 'submission_package/source'
    (source / 'generated').mkdir(parents=True)
    (source / 'figures').mkdir()
    (source / 'main.tex').write_text(
        r'\input{generated/table.tex}' '\n'
        r'\includegraphics{figure.pdf}' '\n'
    )
    (source / 'supplement.tex').write_text('supplement\n')
    for name in ('references.bib', 'cas-dc.cls', 'cas-common.sty', 'elsarticle-num.bst'):
        (source / name).write_text(name + '\n')
    (source / 'README_OVERLEAF.txt').write_text('compile main and supplement\n')
    (source / 'generated/table.tex').write_text('table\n')
    (source / 'figures/figure.pdf').write_bytes(b'%PDF-1.4\n')
    for name in (
        '01_CVIU_main_revised.pdf', '02_CVIU_supplement_revised.pdf',
        '03_highlights_CVIU.txt', '04_graphical_abstract_CVIU.pdf',
        '04_graphical_abstract_CVIU.png', '05_CVIU_cover_letter.pdf',
        'REVIEWER_EVIDENCE_README.md',
        'cover_letter_CVIU.tex', 'graphical_abstract_CVIU.tex',
    ):
        (root / 'submission_package' / name).write_text(name + '\n')
    return source


def test_archive_contains_only_allowlisted_files_and_detects_tamper(tmp_path):
    (tmp_path / 'record.json').write_text('{"value":1}')
    (tmp_path / 'private_token').write_text('not for distribution')
    dest = tmp_path / 'result.zip'
    package.write_archive(tmp_path, ['record.json'], dest)
    with zipfile.ZipFile(dest) as z:
        assert set(z.namelist()) == {'record.json', 'EVIDENCE_SHA256.json'}
        z.extractall(tmp_path / 'extracted')
    package.verify_files(tmp_path / 'extracted')
    (tmp_path / 'extracted/record.json').write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        package.verify_files(tmp_path / 'extracted')


@pytest.mark.parametrize('name', ['../outside', '/etc/passwd'])
def test_archive_rejects_paths_outside_root(tmp_path, name):
    with pytest.raises(ValueError, match='relative'):
        package.write_archive(tmp_path, [name], tmp_path / 'bad.zip')


@pytest.mark.parametrize('name', ['../submission_package', '/tmp/submission_package'])
def test_submission_binding_rejects_package_paths_outside_root(tmp_path, name):
    with pytest.raises(ValueError, match='relative'):
        package.active_submission_files(tmp_path, name)


def test_active_submission_inventory_is_reachable_and_explicit(tmp_path):
    source = submission_fixture(tmp_path)
    (source / 'main.log').write_text('secret build path\n')
    (source / 'unused.tex').write_text('not referenced\n')
    (tmp_path / 'submission_package/review_notes').mkdir()
    (tmp_path / 'submission_package/review_notes/private.md').write_text('review\n')
    (tmp_path / 'submission_package/random.bin').write_bytes(b'random')

    names = package.active_submission_files(tmp_path, 'submission_package')

    assert names == {
        'submission_package/source/main.tex',
        'submission_package/source/supplement.tex',
        'submission_package/source/references.bib',
        'submission_package/source/cas-dc.cls',
        'submission_package/source/cas-common.sty',
        'submission_package/source/elsarticle-num.bst',
        'submission_package/source/README_OVERLEAF.txt',
        'submission_package/source/generated/table.tex',
        'submission_package/source/figures/figure.pdf',
        'submission_package/01_CVIU_main_revised.pdf',
        'submission_package/02_CVIU_supplement_revised.pdf',
        'submission_package/03_highlights_CVIU.txt',
        'submission_package/04_graphical_abstract_CVIU.pdf',
        'submission_package/04_graphical_abstract_CVIU.png',
        'submission_package/05_CVIU_cover_letter.pdf',
        'submission_package/REVIEWER_EVIDENCE_README.md',
        'submission_package/cover_letter_CVIU.tex',
        'submission_package/graphical_abstract_CVIU.tex',
    }


def test_active_submission_rejects_missing_referenced_dependency(tmp_path):
    source = submission_fixture(tmp_path)
    (source / 'generated/table.tex').unlink()

    with pytest.raises(FileNotFoundError, match='active submission dependency'):
        package.active_submission_files(tmp_path, 'submission_package')


def test_active_submission_rejects_dependency_escape(tmp_path):
    source = submission_fixture(tmp_path)
    (tmp_path / 'submission_package/private.tex').write_text('private\n')
    (source / 'main.tex').write_text(r'\input{../private.tex}' + '\n')

    with pytest.raises(ValueError, match='remain under source'):
        package.active_submission_files(tmp_path, 'submission_package')


def test_active_submission_includes_ancillary_only_figure(tmp_path):
    submission_fixture(tmp_path)
    figure = tmp_path / 'submission_package/source/figures/ancillary_only.pdf'
    figure.write_bytes(b'%PDF-1.4 ancillary\n')
    (tmp_path / 'submission_package/graphical_abstract_CVIU.tex').write_text(
        r'\includegraphics{source/figures/ancillary_only.pdf}' + '\n'
    )

    names = package.active_submission_files(tmp_path, 'submission_package')

    assert 'submission_package/source/figures/ancillary_only.pdf' in names


def test_submission_binding_replaces_only_stale_editable_manuscript(tmp_path):
    submission_fixture(tmp_path)
    legacy = tmp_path / 'Thuan_paper_3_CVIU_revised'
    (legacy / 'generated').mkdir(parents=True)
    (legacy / 'main.tex').write_text('stale main\n')
    (legacy / 'supplement.tex').write_text('stale supplement\n')
    (legacy / 'CVIU_V4_Ke_hoach_nang_cap.md').write_text('frozen plan\n')
    (legacy / 'generated/v4_clean_control.tex').write_text('frozen table\n')
    before = {
        'analysis/audit.py',
        'Thuan_paper_3_CVIU_revised/main.tex',
        'Thuan_paper_3_CVIU_revised/supplement.tex',
        'Thuan_paper_3_CVIU_revised/CVIU_V4_Ke_hoach_nang_cap.md',
        'Thuan_paper_3_CVIU_revised/generated/v4_clean_control.tex',
    }

    names, binding = package.bind_submission(tmp_path, 'submission_package', before)

    assert 'analysis/audit.py' in names
    assert not package.LEGACY_EDITABLE_MANUSCRIPT & names
    assert 'submission_package/source/main.tex' in names
    assert 'submission_package/source/supplement.tex' in names
    assert binding['schema'] == 'cviu-v4-submission-binding-v1'
    assert binding['active_submission_package'] == 'submission_package'
    assert binding['active_manuscript_sources'] == [
        'submission_package/source/main.tex',
        'submission_package/source/supplement.tex',
    ]
    assert binding['historical_manuscript_artifacts'] == [
        'Thuan_paper_3_CVIU_revised/CVIU_V4_Ke_hoach_nang_cap.md',
        'Thuan_paper_3_CVIU_revised/generated/v4_clean_control.tex',
    ]
    assert set(binding['active_files_sha256']) == names - (before - package.LEGACY_EDITABLE_MANUSCRIPT)


def test_archive_verifies_active_submission_binding_semantically(tmp_path):
    submission_fixture(tmp_path)
    names, binding = package.bind_submission(tmp_path, 'submission_package', set())
    archive_path = tmp_path / 'evidence.zip'
    package.write_archive(tmp_path, names, archive_path, binding=binding)
    with zipfile.ZipFile(archive_path) as archive:
        assert package.SUBMISSION_BINDING_NAME in archive.namelist()
        archive.extractall(tmp_path / 'extracted')

    package.verify_files(tmp_path / 'extracted')
    package.verify_submission_binding(tmp_path / 'extracted', 'submission_package', names)

    marker = tmp_path / 'extracted' / package.SUBMISSION_BINDING_NAME
    changed = json.loads(marker.read_text())
    changed['active_manuscript_sources'] = [
        'submission_package/source/supplement.tex',
        'submission_package/source/main.tex',
    ]
    marker.write_text(json.dumps(changed, indent=2, sort_keys=True) + '\n')
    hashes = tmp_path / 'extracted/EVIDENCE_SHA256.json'
    inventory = json.loads(hashes.read_text())
    inventory[package.SUBMISSION_BINDING_NAME] = hashlib.sha256(marker.read_bytes()).hexdigest()
    hashes.write_text(json.dumps(inventory, indent=2) + '\n')
    package.verify_files(tmp_path / 'extracted')  # Hash-consistent metadata is insufficient.
    with pytest.raises(ValueError, match='active manuscript'):
        package.verify_submission_binding(tmp_path / 'extracted', 'submission_package', names)


@pytest.mark.parametrize('extra_name', [
    'submission_package/source/main.log',
    'submission_package/review_notes/private.md',
    'private_token',
])
def test_submission_binding_rejects_hash_consistent_noncanonical_file(tmp_path, extra_name):
    submission_fixture(tmp_path)
    canonical, binding = package.bind_submission(tmp_path, 'submission_package', set())
    extra = tmp_path / extra_name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('synthetic forbidden fixture\n')
    archive_path = tmp_path / 'evidence.zip'
    package.write_archive(tmp_path, canonical | {extra_name}, archive_path, binding=binding)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_path / 'extracted')

    package.verify_files(tmp_path / 'extracted')
    with pytest.raises(ValueError, match='canonical evidence inventory'):
        package.verify_submission_binding(
            tmp_path / 'extracted', 'submission_package', canonical,
        )


def test_submission_verify_cli_requires_binding_and_v4_numerical_evidence(tmp_path):
    import subprocess
    import sys
    submission_fixture(tmp_path)
    names, binding = package.bind_submission(tmp_path, 'submission_package', set())
    package.write_archive(tmp_path, names, tmp_path / 'evidence.zip', binding=binding)
    with zipfile.ZipFile(tmp_path / 'evidence.zip') as archive:
        archive.extractall(tmp_path / 'extracted')

    result = subprocess.run([
        sys.executable, package.__file__, '--root', str(tmp_path / 'extracted'),
        '--verify', '--submission-package', 'submission_package',
    ], text=True, capture_output=True)

    assert result.returncode != 0
    assert 'V4 incomplete' in result.stderr


def test_verification_rejects_extra_payload(tmp_path):
    (tmp_path / 'record.json').write_text('{}')
    package.write_archive(tmp_path, ['record.json'], tmp_path / 'result.zip')
    with zipfile.ZipFile(tmp_path / 'result.zip') as z:
        z.extractall(tmp_path / 'extracted')
    (tmp_path / 'extracted/extra.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory'):
        package.verify_files(tmp_path / 'extracted')


def test_verification_rejects_symlink_even_when_contents_match(tmp_path):
    (tmp_path / 'record.json').write_text('{}')
    (tmp_path / 'second.json').write_text('{}')
    package.write_archive(tmp_path, ['record.json', 'second.json'], tmp_path / 'result.zip')
    with zipfile.ZipFile(tmp_path / 'result.zip') as z:
        z.extractall(tmp_path / 'extracted')
    (tmp_path / 'extracted/record.json').unlink()
    (tmp_path / 'extracted/record.json').symlink_to('second.json')
    with pytest.raises(ValueError, match='symlink'):
        package.verify_files(tmp_path / 'extracted')


def test_new_analysis_bindings_are_verified_before_packaging(tmp_path):
    import json
    folder = tmp_path / 'outputs/analysis/codec_interaction_v1'
    folder.mkdir(parents=True)
    (tmp_path / 'source.csv').write_text('value\n1\n')
    (folder / 'summary.json').write_text(json.dumps({
        'source_sha256': {'source.csv': package.digest(tmp_path / 'source.csv')}
    }))
    assert 'source.csv' in package.bound_summary_sources(tmp_path, folder / 'summary.json')
    (tmp_path / 'source.csv').write_text('value\n2\n')
    with pytest.raises(ValueError, match='hash'):
        package.bound_summary_sources(tmp_path, folder / 'summary.json')


def test_package_includes_recomputable_completed_followups_when_evidence_is_installed():
    from pathlib import Path
    import audit_completed_followups as followups
    root = Path(__file__).resolve().parents[1]
    if not (root / followups.TT / 'joint_macro.json').exists():
        pytest.skip('retained follow-up evidence is not installed')
    inventory = root / 'EVIDENCE_SHA256.json'
    names = set(json.loads(inventory.read_text())) if inventory.exists() else package.selected_files(root)
    expected = set(followups.audit(root)['verified_files'])
    assert expected <= names
    assert 'analysis/audit_completed_followups.py' in names


def test_v4_mode_is_explicit_and_fails_before_legacy_selection(tmp_path):
    assert hasattr(package, 'v4_selected_files'), 'V4 completion gate missing'
    with pytest.raises(ValueError, match='V4.*complete'):
        package.selected_files(tmp_path, include_v4=True)


def test_v4_terminal_binding_rejects_missing_changed_and_escape_files(tmp_path):
    assert hasattr(package, 'completed_v4_files'), 'V4 terminal binding validator missing'
    folder = tmp_path / 'result'
    folder.mkdir()
    (folder / 'summary.json').write_text('{}')
    marker = folder / 'complete.json'
    marker.write_text(json.dumps({'summary.json': package.digest(folder / 'summary.json')}))
    assert package.completed_v4_files(tmp_path, Path('result'), 'complete.json', {'summary.json'}) == {'result/complete.json', 'result/summary.json'}
    (folder / 'summary.json').write_text('{"different":true}')
    with pytest.raises(ValueError, match='hash'):
        package.completed_v4_files(tmp_path, Path('result'), 'complete.json', {'summary.json'})
    marker.write_text(json.dumps({'../summary.json': '0' * 64}))
    with pytest.raises(ValueError):
        package.completed_v4_files(tmp_path, Path('result'), 'complete.json', {'summary.json'})


def clean_fixture(root):
    folder = root / 'outputs/analysis/cviu_v4/clean_control'
    (folder / 'bootstrap').mkdir(parents=True)
    points, jobs, rows, blocks = {}, [], [], []
    arms = ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')
    for dataset in ('voc', 'kitti', 'tt100k'):
        models = ('yolo11n', 'yolo11x') if dataset == 'tt100k' else ('yolo11n', 'yolo11m', 'yolo11x')
        np.savez(folder / 'bootstrap' / (dataset + '_schedule.npz'), samples=np.zeros((2000, 1), dtype=int), image_ids=[1])
        for model in models:
            key = dataset + '_' + model
            blocks.append({'dataset': dataset, 'model': model, 'image_ids': [1]})
            for arm in arms:
                name = key + '__' + arm
                points[name] = .5
                np.savez(folder / 'bootstrap' / (name + '.npz'), ap=np.full(2000, .5))
                job = {'name': name, 'out': '/remote/bootstrap/' + name + '.npz', 'n_boot': 2000,
                       'image_ids': [1], 'schedule': '/remote/bootstrap/' + dataset + '_schedule.npz',
                       'schedule_sha256': package.digest(folder / 'bootstrap' / (dataset + '_schedule.npz'))}
                jobs.append(job)
                (folder / 'bootstrap' / (name + '.json')).write_text(json.dumps({
                    'identity': hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    'npz_sha256': package.digest(folder / 'bootstrap' / (name + '.npz'))}))
            rows.append({'block': key, 'group': 'diagnostic' if dataset == 'tt100k' else 'primary',
                         'ap_points': dict.fromkeys(arms, 50.), 'S': 0., 'interval': [0., 0., 0.]})
    registry = {'blocks': blocks, 'n_boot': 2000, 'files': {}}
    (folder / 'registry.json').write_text(json.dumps(registry))
    registry_hash = package.digest(folder / 'registry.json')
    (folder / 'registry.complete.json').write_text(json.dumps({'sha256': registry_hash}))
    (folder / 'points.json').write_text(json.dumps(points))
    (folder / 'bootstrap_jobs.json').write_text(json.dumps(jobs))
    (folder / 'summary.json').write_text(json.dumps({'blocks': rows, 'n_boot': 2000, 'registry_sha256': registry_hash,
                         'primary_macro': {'S': 0., 'interval': [0., 0., 0.]}}))
    refresh_clean_complete(folder)
    return folder


def refresh_clean_complete(folder):
    names = ['registry.json', 'points.json', 'bootstrap_jobs.json', 'summary.json']
    names += [str(p.relative_to(folder)) for p in (folder / 'bootstrap').iterdir()]
    (folder / 'complete.json').write_text(json.dumps({name: package.digest(folder / name) for name in names}))


def test_v4_completion_does_not_authorize_arbitrary_large_artifacts(tmp_path):
    assert hasattr(package, 'completed_v4_files'), 'V4 terminal gate missing'
    folder = tmp_path / 'result'
    folder.mkdir()
    (folder / 'engine.plan').write_bytes(b'engine')
    (folder / 'complete.json').write_text(json.dumps({'engine.plan': package.digest(folder / 'engine.plan')}))
    with pytest.raises(ValueError, match='allowlist'):
        package.completed_v4_files(tmp_path, Path('result'), 'complete.json', {'engine.plan'})


def test_v4_clean_summary_recomputed_from_all_32_vectors(tmp_path):
    assert hasattr(package, 'v4_clean_files'), 'V4 draw-level validation missing'
    folder = clean_fixture(tmp_path)
    names = package.v4_clean_files(tmp_path)
    assert len([name for name in names if name.endswith('.npz')]) == 35
    changed = folder / 'bootstrap/voc_yolo11n__fp8_original.npz'
    np.savez(changed, ap=np.full(2000, .7))
    refresh_clean_complete(folder)  # Consistent hashes cannot hide inconsistent scientific numbers.
    with pytest.raises(ValueError, match='summary|interval|identity'):
        package.v4_clean_files(tmp_path)


def test_v4_clean_complete_cannot_smuggle_weights_or_partial_grid(tmp_path):
    assert hasattr(package, 'v4_clean_files'), 'V4 allowlist missing'
    folder = clean_fixture(tmp_path)
    (folder / 'weights.pt').write_bytes(b'not evidence')
    marker = json.loads((folder / 'complete.json').read_text())
    marker['weights.pt'] = package.digest(folder / 'weights.pt')
    (folder / 'complete.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='allowlist'):
        package.v4_clean_files(tmp_path)


def recovered_fixture(root):
    from build_v4_holdout_table import historical_schedule
    folder = root / 'outputs/analysis/cviu_v4/holdout_synthesis/recovered'
    (folder / 'clean_ap').mkdir(parents=True)
    (folder / 'schedules').mkdir()
    labels, blocks, jobs = [], [], []
    for dataset in ('voc', 'kitti'):
        schedule = folder / 'schedules' / (dataset + '_historical.npz')
        np.savez(schedule, samples=historical_schedule([1], 7), image_ids=[1], seed=7)
        for model in ('yolo11n', 'yolo11m', 'yolo11x'):
            key = dataset + '_' + model
            labels.append(key)
            blocks.append({'dataset': dataset, 'model': model,
                           'clean_gap_percentile95_ap_points': [10., 10., 10.],
                           'corrupted_gap_percentile95_ap_points': [5., 5., 5.],
                           'delta_e_percentile95_ap_points': [-5., -5., -5.]})
            for precision, point in (('int8-entropy', .4), ('fp8', .5)):
                name = key + '__' + precision
                output = folder / 'clean_ap' / (name + '.npz')
                np.savez(output, ap=np.full(2000, point))
                job = {'name': name, 'image_ids': [1], 'n_boot': 2000,
                       'schedule_sha256': package.digest(schedule)}
                jobs.append(job)
                output.with_suffix('.json').write_text(json.dumps({
                    'identity': hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    'npz_sha256': package.digest(output)}))
    (folder / 'recovery_registry.json').write_text(json.dumps({'expected_jobs': 12, 'jobs': jobs}))
    np.savez(folder / 'paired_delta_e_draws.npz', block_delta_e_native=np.full((6, 2000), -.05),
             macro_delta_e_native=np.full(2000, -.05), block_labels=labels)
    np.savez(folder / 'paired_corrupted_gap_draws.npz', block_corrupted_gap_native=np.full((6, 2000), .05),
             macro_corrupted_gap_native=np.full(2000, .05), block_labels=labels)
    summary = {'blocks': blocks, 'primary_macro': {'delta_e_percentile95_ap_points': [-5., -5., -5.],
                                                 'corrupted_gap_percentile95_ap_points': [5., 5., 5.]}}
    return folder, summary


def test_v4_recovered_intervals_are_recomputed_not_trusted(tmp_path):
    assert hasattr(package, 'v4_recovered_draws'), 'recovered draw-level validation missing'
    _, summary = recovered_fixture(tmp_path)
    package.v4_recovered_draws(tmp_path, summary)
    summary['blocks'][0]['corrupted_gap_percentile95_ap_points'] = [6., 6., 6.]
    with pytest.raises(ValueError, match='interval'):
        package.v4_recovered_draws(tmp_path, summary)


def test_v4_recovered_vectors_reject_unpaired_gap(tmp_path):
    assert hasattr(package, 'v4_recovered_draws'), 'recovered draw-level validation missing'
    folder, summary = recovered_fixture(tmp_path)
    with np.load(folder / 'paired_corrupted_gap_draws.npz') as original:
        changed = dict(original)
    changed['block_corrupted_gap_native'] = np.full((6, 2000), .06)
    np.savez(folder / 'paired_corrupted_gap_draws.npz', **changed)
    with pytest.raises(ValueError, match='paired'):
        package.v4_recovered_draws(tmp_path, summary)


def test_v4_recovered_vectors_reject_changed_job(tmp_path):
    folder, summary = recovered_fixture(tmp_path)
    path = folder / 'recovery_registry.json'
    document = json.loads(path.read_text())
    document['jobs'][0]['image_ids'] = [2]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='identity'):
        package.v4_recovered_draws(tmp_path, summary)


def test_v4_example_verifies_exact_unpacked_copy_without_nested_zip(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    source = root / 'outputs/analysis/cviu_v4/holdout_example'
    if not (source / 'example.zip').is_file():
        pytest.skip('verified V4 holdout example is not installed')
    target = tmp_path / 'outputs/analysis/cviu_v4/holdout_example'
    shutil.copytree(source / 'package', target / 'package')
    for name in ('verification.json', 'fresh_report.json'):
        shutil.copy2(source / name, target / name)
    holdout = Path('outputs/analysis/cviu_v4/holdout_provenance/holdout_provenance.json')
    (tmp_path / holdout).parent.mkdir(parents=True)
    shutil.copy2(root / holdout, tmp_path / holdout)
    inventory = json.loads((target / 'package/SHA256.json').read_text())
    expected = {str(Path('outputs/analysis/cviu_v4/holdout_example/package') / name)
                for name in inventory}
    expected.update(str(Path('outputs/analysis/cviu_v4/holdout_example') / name)
                    for name in ('package/SHA256.json', 'verification.json', 'fresh_report.json'))

    with pytest.raises(ValueError, match='original archive is required'):
        package.v4_example_files(tmp_path, holdout)
    names = package.v4_example_files(tmp_path, holdout, verified_extraction=True)

    assert names == expected
    assert 'outputs/analysis/cviu_v4/holdout_example/example.zip' not in names
    changed_relative = next(iter(inventory))
    changed = target / 'package' / changed_relative
    changed.write_bytes(changed.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='payload hash'):
        package.v4_example_files(tmp_path, holdout, verified_extraction=True)
    shutil.copy2(source / 'package' / changed_relative, changed)
    shutil.copy2(source / 'example.zip', target / 'example.zip')
    with (target / 'example.zip').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='original archive hash'):
        package.v4_example_files(tmp_path, holdout)


def test_verify_include_v4_requires_numerical_evidence(tmp_path):
    import subprocess
    import sys
    (tmp_path / 'record.json').write_text('{}')
    package.write_archive(tmp_path, ['record.json'], tmp_path / 'result.zip')
    with zipfile.ZipFile(tmp_path / 'result.zip') as archive:
        archive.extractall(tmp_path / 'extracted')
    result = subprocess.run([sys.executable, package.__file__, '--root', str(tmp_path / 'extracted'),
                             '--verify', '--include-v4'], text=True, capture_output=True)
    assert result.returncode != 0, 'V4 verification silently ignored the missing numerical evidence'
    assert 'V4 incomplete' in result.stderr


def test_v4_bundle_includes_import_dependencies_when_evidence_is_installed():
    root = Path(__file__).resolve().parents[1]
    if not (root / 'outputs/analysis/cviu_v4/clean_control/complete.json').exists():
        pytest.skip('completed V4 evidence is not installed')
    inventory = root / 'EVIDENCE_SHA256.json'
    names = set(json.loads(inventory.read_text())) if inventory.exists() else package.v4_selected_files(root)
    assert {'src/fixed_universe_bootstrap.py', 'src/topic_c/shared_mask_pilot.py',
            'src/pilot_registry.py', 'src/run_confirmatory_bootstrap.py',
            'configs/confirmatory_bootstrap_v1.json'} <= names


def test_v4_publication_generators_run_from_only_allowlisted_sources(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    if not (root / 'outputs/analysis/cviu_v4/clean_control/complete.json').exists():
        pytest.skip('completed V4 evidence is not installed')
    inventory = root / 'EVIDENCE_SHA256.json'
    names = set(json.loads(inventory.read_text())) if inventory.exists() else package.v4_selected_files(root)
    for name in names:
        if (name.startswith(('analysis/', 'tests/')) or name.startswith('outputs/analysis/cviu_v4/clean_control/')
                or name.startswith('outputs/analysis/cviu_v4/holdout_synthesis/recovered/')
                or name == 'outputs/reports/corruption_realization_analysis_v1.json'):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
    assert (tmp_path / 'tests/test_v4_publication_tables.py').is_file()
    assert (tmp_path / 'tests/test_realization_publication_tables.py').is_file()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(tmp_path / 'analysis')}
    subprocess.run([sys.executable, 'analysis/build_v4_publication_tables.py', '--output-dir', str(tmp_path / 'tables')],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    subprocess.run([sys.executable, '-c', 'from pathlib import Path; from build_cviu_revision_artifacts import realization_tables; '
                    'realization_tables(Path("."), Path("tables"), seed_only=True)'],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    clean = (tmp_path / 'tables/v4_clean_control.tex').read_text()
    holdout = (tmp_path / 'tables/v4_holdout_four_ap.tex').read_text()
    realization = (tmp_path / 'tables/corruption_realization_seed_summary.tex').read_text()
    assert '+0.0574' in clean and '[-0.1839, 0.2518]' in clean
    assert '[0.9105, 1.1554]' in holdout and '[-0.7843, -0.2908]' in holdout
    assert 'Area' in realization and 'Height' in realization
