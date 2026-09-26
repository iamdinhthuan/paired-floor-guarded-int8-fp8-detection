#!/usr/bin/env python3
"""Render V4 publication tables from hash-bound completed artifacts, read-only.

No inference, resampling, source rewriting, or dependency on the frozen runner.
Percentiles are computed after within-draw averaging, never from CI endpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

PRIMARY = tuple(f'{d}_yolo11{s}' for d in ('voc', 'kitti') for s in ('n', 'm', 'x'))
DIAGNOSTIC = ('tt100k_yolo11n', 'tt100k_yolo11x')
ARMS = ('int8_original', 'fp8_original', 'int8_j95', 'fp8_j95')
AP_FIELDS = ('int8_clean_ap_points', 'fp8_clean_ap_points',
             'int8_corrupted_mean_ap_points', 'fp8_corrupted_mean_ap_points')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_completion(marker, required):
    """Validate all marker members and require every consumed input to be bound."""
    marker = Path(marker)
    require(marker.is_file(), f'missing completion marker: {marker}')
    document = read_json(marker)
    require(document.get('complete', True) is True, 'incomplete recovery')
    hashes = document.get('files_sha256', document)
    require(isinstance(hashes, dict) and hashes and set(required) <= set(hashes),
            'completion marker does not bind all required inputs')
    for relative, digest in hashes.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe completion path')
        source = marker.parent/path
        require(source.is_file() and sha256_file(source) == digest,
                f'completion hash mismatch: {source}')
    return hashes


def equal(actual, expected, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=1e-10, rtol=0), f'{label} mismatch')


def finite(values, shape, ap=False):
    array = np.asarray(values, dtype=float)
    require(array.shape == shape and np.isfinite(array).all(), 'invalid finite draw/AP shape')
    if ap:
        require((array >= 0).all() and (array <= 1).all(), 'AP outside [0, 1]')
    return array


def interval(values):
    return np.percentile(values, [2.5, 50, 97.5]).tolist()


def paired_jobs(jobs, blocks, arms):
    expected = {b+'__'+a for b in blocks for a in arms}
    require(len(jobs) == len(expected) and {j['name'] for j in jobs} == expected,
            'exact unique block/arm jobs required')
    indexed = {j['name']: j for j in jobs}
    datasets = {}
    for block in blocks:
        reference = indexed[block+'__'+arms[0]]
        ids = reference['image_ids']
        require(ids and ids == sorted(set(ids)), 'paired image order must be unique and sorted')
        identity = (ids, reference['schedule_sha256'], reference['annotation_sha256'])
        dataset = block.split('_')[0]
        require(dataset not in datasets or datasets[dataset] == identity,
                'dataset blocks use different common draws')
        datasets[dataset] = identity
        for arm in arms:
            job = indexed[block+'__'+arm]
            require(job['n_boot'] == 2000, '2000 draws required')
            require((job['image_ids'], job['schedule_sha256'], job['annotation_sha256']) == identity,
                    'four-arm pairing schedule/order/annotation mismatch')
    return indexed


def summarize_clean(summary, points, draws, jobs):
    blocks = PRIMARY + DIAGNOSTIC
    require(summary['units'] == 'AP points' and summary['n_boot'] == 2000, 'clean units/draw count')
    rows = summary['blocks']
    require(len(rows) == 8 and {r['block'] for r in rows} == set(blocks), 'exact eight clean blocks required')
    indexed = paired_jobs(jobs, blocks, ARMS)
    require(set(points) == set(draws) == set(indexed), 'exact 32 AP points/vectors required')
    row_index = {r['block']: r for r in rows}
    result, primary_draws = [], []
    for block in blocks:
        source = row_index[block]
        group = 'primary' if block in PRIMARY else 'diagnostic'
        require(source['group'] == group, 'primary/diagnostic scope mismatch')
        ap = finite([points[block+'__'+a] for a in ARMS], (4,), ap=True)
        vectors = np.asarray([finite(draws[block+'__'+a], (2000,), ap=True) for a in ARMS])
        for arm, point in zip(ARMS, ap):
            equal(point, indexed[block+'__'+arm]['point'], 'job AP point')
            equal(100*point, source['ap_points'][arm], 'summary AP point')
        point = float(100*((ap[1]-ap[0])-(ap[3]-ap[2])))
        values = 100*((vectors[1]-vectors[0])-(vectors[3]-vectors[2]))
        ci = interval(values)
        equal(point, source['S'], 'clean shift')
        equal(ci, source['interval'], 'clean shift interval')
        result.append(dict(block=block, group=group, ap_points=dict(zip(ARMS, 100*ap)), S=point, interval=ci))
        if group == 'primary':
            primary_draws.append(values)
    macro = dict(S=float(np.mean([r['S'] for r in result[:6]])),
                 interval=interval(np.mean(primary_draws, axis=0)))
    for field in macro:
        equal(macro[field], summary['primary_macro'][field], 'primary clean macro '+field)
    return dict(blocks=result, primary_macro=macro)


def load_ap_jobs(base, jobs, folder, hashes):
    """Bind cached AP vectors to canonical job identities and retained schedules."""
    draws, schedules = {}, {}
    for job in jobs:
        name = job['name']
        relative = f'{folder}/{name}.npz'
        require(relative in hashes and f'{folder}/{name}.json' in hashes, 'unbound AP cache')
        metadata = read_json(base/f'{folder}/{name}.json')
        identity = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(metadata == dict(identity=identity, npz_sha256=hashes[relative]), 'cache/job identity mismatch')
        with np.load(base/relative, allow_pickle=False) as saved:
            require(set(saved.files) == {'ap'}, 'unexpected AP cache fields')
            draws[name] = finite(saved['ap'], (2000,), ap=True)
        schedule_name = Path(job['schedule']).name
        schedule_relative = ('bootstrap/' if folder == 'bootstrap' else 'schedules/') + schedule_name
        require(schedule_relative in hashes and hashes[schedule_relative] == job['schedule_sha256'],
                'unbound or wrong schedule hash')
        if schedule_relative not in schedules:
            with np.load(base/schedule_relative, allow_pickle=False) as saved:
                ids, samples = saved['image_ids'], saved['samples']
                require(samples.shape == (2000, len(ids)) and samples.dtype.kind in 'iu'
                        and (samples >= 0).all() and (samples < len(ids)).all(), 'malformed retained schedule')
                schedules[schedule_relative] = ids.tolist()
        require(job['image_ids'] == schedules[schedule_relative], 'job/schedule image order mismatch')
    return draws


def load_clean(base):
    base = Path(base)
    hashes = verify_completion(base/'complete.json', {'summary.json', 'points.json', 'bootstrap_jobs.json', 'registry.json'})
    summary, registry = read_json(base/'summary.json'), read_json(base/'registry.json')
    require(summary['registry_sha256'] == hashes['registry.json'], 'summary/registry binding mismatch')
    require(registry['n_boot'] == 2000 and len(registry['blocks']) == 8
            and {b['dataset']+'_'+b['model'] for b in registry['blocks']} == set(PRIMARY+DIAGNOSTIC),
            'registry clean scope mismatch')
    jobs = read_json(base/'bootstrap_jobs.json')
    draws = load_ap_jobs(base, jobs, 'bootstrap', hashes)
    return summarize_clean(summary, read_json(base/'points.json'), draws, jobs)


def summarize_holdout(summary, delta, gap, labels):
    require(summary['units'] == 'AP points' and summary['n_boot'] == 2000, 'holdout units/draw count')
    rows = summary['blocks']
    row_labels = [r['dataset']+'_'+r['model'] for r in rows]
    require(len(rows) == 6 and set(row_labels) == set(PRIMARY), 'exact six primary holdout blocks required')
    require(list(labels) == row_labels, 'holdout draw labels/order mismatch')
    delta, gap = finite(delta, (6, 2000)), finite(gap, (6, 2000))
    result = copy.deepcopy(summary)
    for i, row in enumerate(result['blocks']):
        require(row['conditions'] == 12 and row['clean_control'] == 'original_source',
                'holdout requires twelve cells and original-source clean')
        i0, f0, ic, fc = finite(np.array([row[k] for k in AP_FIELDS])/100, (4,), ap=True)*100
        values = dict(clean_gap_ap_points=f0-i0, corrupted_gap_ap_points=fc-ic,
                      delta_e_ap_points=(fc-ic)-(f0-i0),
                      delta_e_percentile95_ap_points=interval(delta[i]*100),
                      corrupted_gap_percentile95_ap_points=interval(gap[i]*100))
        for field, value in values.items():
            equal(value, row[field], 'holdout '+field)
            row[field] = value
    require(summary['primary_macro']['blocks'] == 6, 'holdout macro scope mismatch')
    for stem, vectors in [('delta_e', delta), ('corrupted_gap', gap)]:
        values = {stem+'_ap_points': float(np.mean([r[stem+'_ap_points'] for r in result['blocks']])),
                  stem+'_percentile95_ap_points': interval(vectors.mean(axis=0)*100)}
        for field, value in values.items():
            equal(value, summary['primary_macro'][field], 'holdout macro '+field)
            result['primary_macro'][field] = value
    return result


def load_holdout(base):
    base = Path(base)
    hashes = verify_completion(base/'recovery.complete.json',
                               {'completion.json', 'recovery_registry.json', 'paired_corrupted_gap_draws.npz'})
    verify_completion(base/'completion.json', {'holdout_synthesis.json', 'paired_delta_e_draws.npz'})
    registry = read_json(base/'recovery_registry.json')
    jobs = registry['jobs']
    paired_jobs(jobs, PRIMARY, ('int8-entropy', 'fp8'))
    clean = load_ap_jobs(base, jobs, 'clean_ap', hashes)
    summary = read_json(base/'holdout_synthesis.json')
    with np.load(base/'paired_delta_e_draws.npz', allow_pickle=False) as saved:
        delta, labels = saved['block_delta_e_native'], saved['block_labels'].tolist()
        equal(saved['macro_delta_e_native'], delta.mean(axis=0), 'stored DeltaE macro draws')
    with np.load(base/'paired_corrupted_gap_draws.npz', allow_pickle=False) as saved:
        gap = saved['block_corrupted_gap_native']
        require(saved['block_labels'].tolist() == labels, 'gap/DeltaE block pairing mismatch')
        equal(saved['macro_corrupted_gap_native'], gap.mean(axis=0), 'stored gap macro draws')
    require(labels == list(PRIMARY), 'unexpected retained holdout order')
    for i, block in enumerate(labels):
        equal(gap[i], delta[i]+clean[block+'__fp8']-clean[block+'__int8-entropy'],
              'recovered gap/clean/DeltaE draw pairing')
    result = summarize_holdout(summary, delta, gap, labels)
    for job in jobs:
        block, precision = job['name'].split('__')
        row = result['blocks'][labels.index(block)]
        equal(job['point']*100, row['fp8_clean_ap_points' if precision == 'fp8' else 'int8_clean_ap_points'],
              'historical clean point reconstruction')
    return result


def label(block):
    dataset, model = block.split('_')
    return {'voc': 'VOC', 'kitti': 'KITTI', 'tt100k': 'TT100K'}[dataset], model.replace('yolo', 'YOLO')


def ci_text(ci):
    return f'[{ci[0]:.4f}, {ci[2]:.4f}]'


def render_clean(result):
    lines = [r'% Generated by analysis/build_v4_publication_tables.py; do not edit.',
             r'\begin{table*}[tp]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\renewcommand{\arraystretch}{1.18}',
             r'\caption{Effect of replacing original-source clean inputs with JPEG-95 clean controls. All values are AP points; engines and image draws are held fixed.}',
             r'\label{tab:v4_clean_control}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrc@{}}', r'\toprule',
             r' & & \multicolumn{2}{c}{Original clean} & \multicolumn{2}{c}{JPEG-Q95 clean} & & \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'Dataset & Model & INT8 & FP8 & INT8 & FP8 & Shift $S$ & 95\% interval \\', r'\midrule',
             r'\multicolumn{8}{@{}l}{\textbf{Primary: six VOC/KITTI holdout blocks}} \\', r'\addlinespace[2pt]']
    for i, row in enumerate(result['blocks']):
        if i == 3:
            lines.append(r'\addlinespace[4pt]')
        if i == 6:
            macro = result['primary_macro']
            lines += [r'\midrule', r'\multicolumn{6}{@{}l}{\textbf{Primary equal-block mean}} & '
                      +f"{macro['S']:+.4f} & {ci_text(macro['interval'])}"+r' \\',
                      r'\midrule', r'\multicolumn{8}{@{}l}{\textbf{Diagnostic: TT100K (not pooled)}} \\', r'\addlinespace[2pt]']
        fields = [*label(row['block']), *[f'{row["ap_points"][a]:.2f}' for a in ARMS],
                  f'{row["S"]:+.4f}', ci_text(row['interval'])]
        lines.append(' & '.join(fields)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular*}',
        r'\par\vspace{4pt}\begin{minipage}{\textwidth}\footnotesize',
        r'\textit{Reading the shift.} $S=G_{\mathrm{original}}-G_{\mathrm{J95}}=\Delta E_{\mathrm{J95}}-\Delta E_{\mathrm{original}}$, with corrupted predictions held fixed. A positive value denotes a larger interaction after clean-control substitution, not improved corrupted accuracy. One estimate is reported per block.',
        r'\par\smallskip\textit{Uncertainty.} Paired-image percentile intervals are based on 2,000 draws, conditional on retained engines and encoded bytes. The primary mean is formed within each draw before percentiles are taken; TT100K diagnostics are excluded. Contrasts are computed from unrounded AP.',
        r'\end{minipage}', r'\end{table*}', ''])


def estimate_ci(point, ci):
    # A bare '[' after \\ is parsed as an optional vertical-space argument.
    return r'\shortstack{'+f'{point:+.4f}'+r'\\ \mbox{'+ci_text(ci)+'}}'


def render_holdout(result):
    lines = [r'% Generated by analysis/build_v4_publication_tables.py; do not edit.',
             r'\begin{table*}[tp]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\renewcommand{\arraystretch}{1.22}',
             r'\caption{Selection-disjoint holdout results with original-source clean controls. Absolute AP is shown in Panel A; the corresponding gaps and paired uncertainty are shown in Panel B. All values are AP points.}',
             r'\label{tab:v4_holdout_four_ap}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}', r'\toprule',
             r'\multicolumn{6}{@{}l}{\textbf{A. Absolute task accuracy}} \\', r'\addlinespace[3pt]',
             r' & & \multicolumn{2}{c}{Original clean AP} & \multicolumn{2}{c}{Mean corrupted AP} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r'Dataset & Model & INT8 & FP8 & INT8 & FP8 \\', r'\midrule']
    for i, row in enumerate(result['blocks']):
        if i == 3:
            lines.append(r'\addlinespace[4pt]')
        fields = [*label(row['dataset']+'_'+row['model']), *[f'{row[k]:.2f}' for k in AP_FIELDS]]
        lines.append(' & '.join(fields)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\par\vspace{9pt}',
              r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrcrc@{}}', r'\toprule',
              r'\multicolumn{7}{@{}l}{\textbf{B. FP8--INT8 gaps and clean-adjusted interaction}} \\', r'\addlinespace[3pt]',
              r'Dataset & Model & \shortstack{Clean gap\\$G_0$} & \shortstack{Corrupted gap\\$G_c$} & \shortstack{95\% interval\\for $G_c$} & \shortstack{Interaction\\$\Delta E$} & \shortstack{95\% interval\\for $\Delta E$} \\', r'\midrule']
    for i, row in enumerate(result['blocks']):
        if i == 3:
            lines.append(r'\addlinespace[4pt]')
        fields = [*label(row['dataset']+'_'+row['model']), f'{row["clean_gap_ap_points"]:+.4f}',
                  f'{row["corrupted_gap_ap_points"]:+.4f}', ci_text(row['corrupted_gap_percentile95_ap_points']),
                  f'{row["delta_e_ap_points"]:+.4f}', ci_text(row['delta_e_percentile95_ap_points'])]
        lines.append(' & '.join(fields)+r' \\')
    macro = result['primary_macro']
    lines += [r'\midrule', r'\multicolumn{3}{@{}l}{\textbf{Equal-block mean (six blocks)}} & '
              +f"{macro['corrupted_gap_ap_points']:+.4f} & {ci_text(macro['corrupted_gap_percentile95_ap_points'])} & "
              +f"{macro['delta_e_ap_points']:+.4f} & {ci_text(macro['delta_e_percentile95_ap_points'])}"+r' \\']
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular*}',
        r'\par\vspace{5pt}\begin{minipage}{\textwidth}\footnotesize',
        r'\textit{How to read the table.} A positive $G_c$ denotes higher corrupted AP for FP8; a negative $\Delta E$ denotes a smaller signed gap than on clean inputs. Thus higher corrupted accuracy and a contracting advantage can be observed together.',
        r'\par\smallskip\textit{Definitions.} $G_0=A_{F,0}-A_{I,0}$, $G_c=\overline{A}_{F,c}-\overline{A}_{I,c}$, and $\Delta E=G_c-G_0$. Corrupted AP is averaged equally over twelve corruption--severity cells per block. Contrasts are computed from unrounded AP, not from the two-decimal values in Panel A.',
        r'\par\smallskip\textit{Uncertainty and provenance.} Paired 95\% percentile intervals are based on 2,000 draws; block and macro means are formed before percentiles are taken. The $G_c$ intervals are reconstructed from retained $\Delta E$ draws and original-clean AP draws on the recorded historical schedule. Historical NumPy/evaluator versions were not recorded. These estimates are not terminal-codec matched; recovery details are provided in S1.',
        r'\end{minipage}', r'\end{table*}', ''])


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=root/'outputs/analysis/cviu_v4')
    parser.add_argument('--output-dir', type=Path, default=root/'submission_support_20260911/source/generated')
    args = parser.parse_args()
    clean = load_clean(args.inputs/'clean_control')
    holdout = load_holdout(args.inputs/'holdout_synthesis/recovered')
    outputs = {'v4_clean_control.tex': render_clean(clean), 'v4_holdout_four_ap.tex': render_holdout(holdout)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        target = args.output_dir/name
        require(not target.resolve().is_relative_to(args.inputs.resolve()), 'output must not overwrite source evidence')
        target.write_text(content)
        print(target)


if __name__ == '__main__':
    main()
