"""Protect matched denominators, committed-only reporting and snapshot integrity."""
from copy import deepcopy
import math

import pytest

from research import report_simple_pilot as report


def rows():
    return {str(index): {'track_hash': str(index), 'group': 'mixed', 'cap_m_s': 1., 'error': '',
                        'completed': True, 'completion_time_s': 10. + index,
                        'geometry_lateral_rmse_m': .01, 'command_jitter': .001}
            for index in range(3)}


def test_paired_times_exclude_early_failures_and_show_denominator():
    left, right = rows(), rows()
    left['0'].update(completed=False, completion_time_s=None)
    right['1'].update(completed=False, completion_time_s=None)
    result = report.paired(left, right)
    assert result['shared_successes'] == 1 and result['total_courses'] == 3
    assert result['track_hashes'] == ['2']
    assert result['completion_time_s'] == {'left': 12., 'right': 12.}
    right['2'].update(completed=False, completion_time_s=None)
    assert report.paired(left, right) == {'shared_successes': 0, 'total_courses': 3, 'track_hashes': []}
    del right['0']
    with pytest.raises(ValueError, match='geometries'):
        report.paired(left, right)


@pytest.mark.parametrize('field,value', [('cap_m_s', .5), ('group', 'speed'), ('error', 'failed'),
                                        ('geometry_lateral_rmse_m', math.nan), ('command_jitter', -.01),
                                        ('completion_time_s', None), ('completion_time_s', 0.)])
def test_control_rows_cannot_silently_change_environment_or_validity(field, value):
    values = rows()
    values['0'][field] = value
    with pytest.raises(ValueError):
        report.checked_rows(list(values.values()), {str(i): 'mixed' for i in range(3)}, 1.)


def test_uncommitted_csv_rows_do_not_enter_analysis():
    data = b'generation,training_episodes\n0,600\n1,600\n2,'
    assert report.table(data, 1) == [{'generation': '0', 'training_episodes': '600'}]


def test_snapshot_replay_is_exact_and_rejects_tampered_inputs(tmp_path):
    original = {'protocol': b'{"test":1}', 'numeric-results': b'[1,2,3]'}
    snapshot = report.freeze(deepcopy(original), tmp_path)
    restored, manifest = report.replay(snapshot)
    assert {key: restored[key] for key in original} == original
    assert restored['analysis-source'] == report.Path(report.__file__).read_bytes()
    target = snapshot / manifest['inputs']['numeric-results']['file']
    target.write_bytes(b'[4,5,6]')
    with pytest.raises(ValueError, match='bytes changed'):
        report.replay(snapshot)
