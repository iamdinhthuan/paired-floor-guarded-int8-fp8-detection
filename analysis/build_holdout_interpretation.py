#!/usr/bin/env python3
"""Post hoc sign diagnostic of the frozen 72-cell holdout; no new inference."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bootstrap_format_contrast import canonical_hash
from topic_c.manifest import sha256_file


def validate_grid(rows):
    expected = {(d, m, c, s) for d in ('voc', 'kitti')
                for m in ('yolo11n', 'yolo11m', 'yolo11x')
                for c in ('gaussian_noise', 'motion_blur', 'fog', 'jpeg') for s in (1, 3, 5)}
    actual = {(r['dataset'], r['model'], r['corruption'], r['severity']) for r in rows}
    if len(rows) != 72 or actual != expected:
        raise ValueError('complete unique 72-cell holdout grid required')


def summarize(rows):
    if not rows:
        raise ValueError('empty group')
    raw_positive = negative = reversals = below = crossing = 0
    effects = []
    thresholds = {str(t): 0 for t in (.25, .5, 1.)}
    for row in rows:
        aps = [float(row[k]) for k in ('int8_clean_ap', 'fp8_clean_ap',
                                      'int8_corrupted_ap', 'fp8_corrupted_ap')]
        effect = float(row['delta_e'])
        interval = list(map(float, row['delta_e_percentile95']))
        if (len(interval) != 3 or not all(math.isfinite(x) for x in aps + [effect] + interval)
                or not all(0 <= x <= 1 for x in aps) or interval != sorted(interval)):
            raise ValueError('invalid AP or percentile interval')
        raw, clean = aps[3] - aps[2], aps[1] - aps[0]
        if not math.isclose(raw-clean, effect, rel_tol=0, abs_tol=1e-12):
            raise ValueError('four-arm identity mismatch')
        effects.append(effect * 100)
        raw_positive += raw > 0
        negative += effect < 0
        if raw > 0 and effect < 0:
            reversals += 1
            below += interval[2] < 0
            crossing += interval[0] <= 0 <= interval[2]
            for t in thresholds:
                thresholds[t] += abs(effect * 100) >= float(t)
    return dict(cells=len(rows), raw_positive=raw_positive, adjusted_negative=negative,
                reversals=reversals, reversal_interval_below_zero=below,
                reversal_interval_crosses_zero=crossing,
                reversal_magnitude_counts_ap=thresholds, mean_delta_e_ap=sum(effects)/len(effects))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'paper/confirmatory_evidence/untouched_holdout_analysis.json'
    report = json.loads(source.read_text())
    if report['analysis_sha256'] != canonical_hash(report, 'analysis_sha256'):
        raise ValueError('holdout analysis hash mismatch')
    rows = report['cells']
    validate_grid(rows)
    groups = {d: summarize([r for r in rows if d == 'all' or r['dataset'] == d])
              for d in ('voc', 'kitti', 'all')}
    if not math.isclose(groups['all']['mean_delta_e_ap'],
                        100 * report['overall_balanced_equal_cell']['delta_e_point'], abs_tol=1e-10):
        raise ValueError('holdout mean mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'cells.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(schema_version=1, scope='Post hoc finite-grid diagnostic; unadjusted intervals; no independent replication.',
                  source=str(source.relative_to(root)), source_sha256=sha256_file(source),
                  implementation_sha256=sha256_file(__file__), groups=groups,
                  cells_sha256=sha256_file(args.output / 'cells.csv'))
    result['summary_sha256'] = canonical_hash(result, 'summary_sha256')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Dataset & Cells & Raw $G_c>0$ & $\Delta E<0$ & Reversals & CI below 0 & CI crosses 0\\', r'\midrule']
    for d, g in groups.items():
        label = {'voc':'VOC', 'kitti':'KITTI', 'all':'All'}[d]
        lines.append(label + ' & ' + ' & '.join(str(g[k]) for k in
                     ('cells', 'raw_positive', 'adjusted_negative', 'reversals',
                      'reversal_interval_below_zero', 'reversal_interval_crosses_zero')) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(lines) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
