#!/usr/bin/env python3
"""Recompute the two completed follow-ups from compact paired draw evidence.

No GPU, raw predictions or private host access is required. This is a
summary/draw-level audit, not a fresh AP evaluation or upstream engine audit.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


TT = Path('outputs/bootstrap/tt100k_fixed_universe_full_v1_20260905')
CONTROL = Path('outputs/controlled_build/controlled_retinanet_tf32off_v1_20260906/analysis_v1')
REMOTE = '/home/thuan/topic_c_ivc/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(root, name, expected):
    # Immutable remote identities map only to their matching root-relative path.
    name = str(name).removeprefix(REMOTE)
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'unsafe evidence path: {name}')
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f'evidence hash mismatch: {name}')
    return path


def mean_percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) == 0 or not np.isfinite(arr).all():
        raise ValueError('expected nonempty finite cell-by-draw matrix')
    return np.percentile(arr.mean(axis=0), [2.5, 50., 97.5]).tolist()


def recipe_draws(dc, ac, fc, dx, ax, fx):
    default = (fx - dx) - (fc - dc)
    aligned = (fx - ax) - (fc - ac)
    omega = default - aligned
    np.testing.assert_allclose(omega, (ax - dx) - (ac - dc), rtol=0, atol=1e-11)
    return {'delta_e_default': default, 'delta_e_aligned': aligned, 'omega': omega}


def audit(root):
    names = {str(TT / 'joint_macro.json'), str(CONTROL / 'complete.json')}

    def bound(name, sha):
        path = checked_file(root, name, sha)
        names.add(str(path.relative_to(root)))
        return path

    joint = json.loads((root / TT / 'joint_macro.json').read_text())
    expected_cells = {f'tt100k__{m}__{c}-s{s}' for m, c, s in itertools.product(
        ('yolo11n', 'yolo11m', 'yolo11x'),
        ('fog', 'gaussian_noise', 'jpeg', 'motion_blur'), (1, 3, 5))}
    bindings = joint['source_bindings']
    if len(bindings) != 36 or {b['cell'] for b in bindings} != expected_cells:
        raise ValueError('TT100K must contain the exact 36-cell grid')
    keys = ('delta_e_all', 'delta_psi_height')
    draws = {method: {key: [] for key in keys}
             for method in ('fixed_universe', 'ordinary_reference')}
    points = {key: [] for key in keys}
    for b in bindings:
        report = json.loads(bound(b['report_path'], b['report_sha256']).read_text())
        bound(b['component'], b['component_sha256'])
        for key in keys:
            points[key].append(report['point'][key] * 100)
        for method, field, n in (('fixed_universe', 'draws', 10000),
                                  ('ordinary_reference', 'ordinary_draws', 2000)):
            path = bound(b[field]['path'], b[field]['sha256'])
            with np.load(path, allow_pickle=False) as cache:
                if method == 'fixed_universe':
                    if str(cache['schedule_identity_sha256']) != joint['schedule']['schedule_identity_sha256']:
                        raise ValueError('TT100K fixed schedule identity mismatch')
                for key in keys:
                    cache_key = 'delta_psi' if method == 'ordinary_reference' and key == 'delta_psi_height' else key
                    arr = cache[cache_key]
                    if arr.shape != (n,):
                        raise ValueError('TT100K draw count mismatch')
                    draws[method][key].append(arr * 100)
    tt_result = {}
    for method, group in draws.items():
        tt_result[method] = {}
        for key, values in group.items():
            point = float(np.mean(points[key]))
            interval = mean_percentiles(values)
            ref = joint[method][key]
            np.testing.assert_allclose(point, ref['point_native_ap'] * 100, rtol=0, atol=1e-10)
            np.testing.assert_allclose(interval, np.array(ref['percentile_interval']) * 100, rtol=0, atol=1e-10)
            row = {'point': point, 'percentile_interval': interval}
            if method == 'fixed_universe':
                checkpoint = mean_percentiles(np.asarray(values)[:, :2000])
                shift = float(np.max(np.abs(np.array(checkpoint) - interval)))
                np.testing.assert_allclose(shift, ref['max_checkpoint_percentile_shift_native_ap'] * 100,
                                           rtol=0, atol=1e-10)
                row['checkpoint_max_shift'] = shift
            tt_result[method][key] = row

    complete = json.loads((root / CONTROL / 'complete.json').read_text())
    for name, sha in complete.items():
        bound(CONTROL / name, sha)
    summary = json.loads((root / CONTROL / 'summary.json').read_text())
    manifest = json.loads((root / CONTROL / 'manifest.json').read_text())
    identity = manifest.pop('manifest_sha256')
    canonical = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != canonical or identity != summary['manifest_sha256']:
        raise ValueError('controlled canonical manifest identity mismatch')
    caches = {}
    for name, sha in summary['draw_caches'].items():
        path = bound(name, sha)
        with np.load(path, allow_pickle=False) as cache:
            arr = cache['ap'] * 100
        if arr.shape != (2000,) or not np.isfinite(arr).all():
            raise ValueError('controlled AP draw count/value mismatch')
        caches[path.stem] = arr
    conditions = {(c, s) for c, s in itertools.product(
        ('fog', 'gaussian_noise', 'jpeg', 'motion_blur'), (1, 3, 5))}
    expected_caches = {f'{a}__{c}-s{s}' for a, (c, s) in itertools.product(
        ('default', 'aligned', 'fp8'), conditions | {('clean', 0)})}
    if set(caches) != expected_caches or len(summary['cells']) != 12:
        raise ValueError('controlled grid inventory mismatch')
    if {(c['corruption'], c['severity']) for c in summary['cells']} != conditions:
        raise ValueError('controlled cells must cover all 12 conditions')
    pooled = {key: [] for key in ('delta_e_default', 'delta_e_aligned', 'omega')}
    point_values = {key: [] for key in pooled}
    for cell in summary['cells']:
        condition = f"{cell['corruption']}-s{cell['severity']}"
        values = recipe_draws(*(caches[f'{a}__{c}'] for c in ('clean-s0', condition)
                               for a in ('default', 'aligned', 'fp8')))
        pts = recipe_draws(*(cell[f'{a}_{c}_ap'] for c in ('clean', 'corrupt')
                            for a in ('default', 'aligned', 'fp8')))
        for key, vector in values.items():
            np.testing.assert_allclose(pts[key], cell[key], rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.percentile(vector, [2.5, 50, 97.5]),
                                       cell[f'{key}_interval'], rtol=0, atol=1e-10)
            pooled[key].append(vector)
            point_values[key].append(pts[key])
    controlled = {}
    for key, values in pooled.items():
        point = float(np.mean(point_values[key]))
        interval = mean_percentiles(values)
        np.testing.assert_allclose(point, summary['macro'][key]['point'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(interval, summary['macro'][key]['percentile_interval'], rtol=0, atol=1e-10)
        controlled[key] = {'point': point, 'percentile_interval': interval}
    return {'units': 'AP points', 'tt100k': tt_result, 'controlled': controlled,
            'scope': 'compact draw/summary verification, not fresh AP or upstream engine validation',
            'verified_files': sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve()), indent=2))


if __name__ == '__main__':
    main()
