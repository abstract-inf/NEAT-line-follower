"""Keep failed-track denominators and incomplete-run claims explicit."""
from copy import deepcopy

from neat_simple_study import confirmation_reporting as reporting
from neat_smooth_study.common import read, write_json
from test_simple_confirmation_training import prepared_run, train


def test_paired_failure_is_not_a_fast_finish():
    rows = [{'track_hash': str(i), 'completed': True, 'completion_time_s': 5.+i,
             'geometry_lateral_rmse_m': .01, 'command_jitter': .003} for i in range(3)]
    reference = deepcopy(rows)
    rows[0].update(completed=False, completion_time_s=None)
    reference[1].update(completed=False, completion_time_s=None)
    result = reporting.pair(rows, reference)
    assert result['shared_successes'] == 1 and result['courses'] == 3
    assert result['track_hashes'] == ['2']
    assert result['completion_time_s'] == {'neat': 7., 'reference': 7.}


def test_interim_training_report_handles_unobserved_motion_without_claiming_final_results(prepared_run):
    fixture = prepared_run
    directory = train(fixture, workers=6, stop_after=1)
    path = directory / 'validation/generation-0000.json'
    record = read(path)
    for split in ('primary', 'retention'):
        for row in record[split]:
            row.update(average_forward_speed_m_s=.4, peak_forward_speed_m_s=.5, termination='completed')
    # No straight was observed: yaw is unknown, not fabricated as zero.
    write_json(path, record)
    result = reporting.report(fixture.run)
    assert result['final_assessment'] is None
    assert len(result['training']['1']) == 1 and result['training']['2'] == []
    assert result['training']['1'][0]['primary']['maximum_straight_yaw_rms_rad_s'] is None
    assert not (fixture.run / 'report/methods-reference.tex').exists()
    assert (fixture.run / 'report/figures/validation-learning.svg').exists()
