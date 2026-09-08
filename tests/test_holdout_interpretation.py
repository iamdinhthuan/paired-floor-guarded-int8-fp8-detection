import copy
import pytest
import build_holdout_interpretation as audit


def cell(raw, clean, interval):
    return dict(int8_clean_ap=.4, fp8_clean_ap=.4 + clean,
                int8_corrupted_ap=.2, fp8_corrupted_ap=.2 + raw,
                delta_e=raw-clean, delta_e_percentile95=interval)


def test_reversal_denominator_excludes_already_negative_and_zero_gaps():
    rows = [cell(.02, .04, [-.03, -.02, -.01]),
            cell(.01, .02, [-.03, -.01, .01]),
            cell(-.01, .02, [-.05, -.03, -.01]),
            cell(0, .02, [-.03, -.02, -.01]),
            cell(.02, .01, [0, .01, .03])]
    result = audit.summarize(rows)
    assert result['cells'] == 5
    assert result['raw_positive'] == 3
    assert result['adjusted_negative'] == 4
    assert result['reversals'] == 2
    assert result['reversal_interval_below_zero'] == 1
    assert result['reversal_interval_crosses_zero'] == 1


def test_four_arm_inconsistency_is_rejected():
    row = cell(.02, .04, [-.03, -.02, -.01])
    row['delta_e'] = .2
    with pytest.raises(ValueError, match='four-arm'):
        audit.summarize([row])


def test_nonfinite_and_unordered_intervals_are_rejected():
    for interval in [[float('nan'), 0, 1], [.1, -.1, 0]]:
        with pytest.raises(ValueError):
            audit.summarize([cell(.02, .04, interval)])


def test_grid_validation_rejects_duplicate_cells():
    rows = [dict(dataset=d, model=m, corruption=c, severity=s)
            for d in ['voc', 'kitti'] for m in ['yolo11n', 'yolo11m', 'yolo11x']
            for c in ['gaussian_noise', 'motion_blur', 'fog', 'jpeg'] for s in [1, 3, 5]]
    audit.validate_grid(rows)
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match='72-cell'):
        audit.validate_grid(rows)
