from dataclasses import asdict
from types import SimpleNamespace

import pytest

from line_follower.config import EpisodeConfig
from line_follower.pid import ResearchPIDConfig
import run_pid_quality_tuning as study


def episode(name='isolated', *, completed=True, speed=1., kind='isolated', yaw=.1, time=5.):
    return {'track_hash': name, 'kind': kind, 'completed': completed,
            'termination': 'completed' if completed else 'timeout',
            'completion_time_s': time if completed else None, 'time_limit_s': 45.,
            'straight_windows': [{'observed': speed is not None, 'speed_m_s': speed,
                                  'yaw_rms_rad_s': yaw if speed is not None else None}],
            'command_jitter': .001, 'geometry_lateral_rmse_m': .002}


def result(rows):
    return {'rows': rows, 'error': None}


def validation(*, failures=0, case_completed=True, speed=1.):
    rows = [episode('isolated', speed=speed)]
    rows += [episode(f'mixed-{i}', kind='mixed', completed=i >= failures) for i in range(59)]
    return result([*rows, episode('long', kind='long_case', completed=case_completed)])


def test_speed_completion_and_long_case_gates_cannot_be_traded_away():
    assert study.summarize(validation(failures=3), 1.)['eligible']
    assert not study.summarize(validation(failures=4), 1.)['eligible']
    assert not study.summarize(validation(case_completed=False), 1.)['eligible']
    assert not study.summarize(validation(speed=.949), 1.)['eligible']
    assert not study.summarize(validation(speed=None), 1.)['eligible']
    assert study.summarize(validation(speed=.475), .5)['eligible']
    assert not study.summarize(result([]), 1.)['eligible']


def test_selection_completion_then_two_percent_time_band_then_quality():
    base = study.summarize(validation(), 1.)
    summaries = {'fast': {**base, 'failure_penalized_mean_time_s': 10., 'steady_yaw_rms': .2},
                 'smooth': {**base, 'failure_penalized_mean_time_s': 10.19, 'steady_yaw_rms': .1},
                 'too-slow': {**base, 'failure_penalized_mean_time_s': 10.21, 'steady_yaw_rms': 0.},
                 'incomplete': {**base, 'completed': 60, 'failure_penalized_mean_time_s': 1., 'steady_yaw_rms': 0.}}
    winner, shortlist = study.select(summaries)
    assert winner == 'smooth'
    assert set(shortlist) == {'fast', 'smooth'}
    assert study.select({'failed': study.summarize(validation(case_completed=False), 1.)}) == (None, [])


def test_candidate_grid_always_searches_and_preserves_controller_recovery():
    parameters = asdict(ResearchPIDConfig(kp=.6, ki=.05, kd=.03, base=1., derivative_filter_s=.1))
    controls = {'current-tune': parameters}
    policies, _ = study.initial_candidates(controls, 1.)
    again, _ = study.initial_candidates(controls, 1.)
    assert policies == again and len(policies) == 130
    tiny = [p for n, p in policies.items() if n.startswith('tiny-')]
    assert len(tiny) == 10
    for name, policy in policies.items():
        assert policy['base'] == 1.
        assert all(policy[k] == parameters[k] for k in parameters if k not in study.FIELDS)
        assert all(lo <= policy[k] <= hi for k, (lo, hi) in zip(study.FIELDS, study.BOUNDS))
        if name.startswith('tiny-'):
            assert sum(policy[k] != parameters[k] for k in study.FIELDS) == 1


@pytest.mark.parametrize('cap,limit', [(1., 180.), (1.5, 180.), (.5, 360.)])
def test_long_case_uses_identical_explicit_timeout_for_every_policy(cap, limit):
    original = asdict(EpisodeConfig())
    configured = study.long_episode(original, cap)
    assert EpisodeConfig(**configured).time_limit(45.77, cap) == limit
    assert original == asdict(EpisodeConfig())
    assert {k: v for k, v in configured.items() if 'time_limit' not in k} == {
        k: v for k, v in original.items() if 'time_limit' not in k}


def test_local_adapter_restores_normal_episode_and_records_case_as_development():
    normal = asdict(EpisodeConfig())
    observed = []
    context = SimpleNamespace(plan={'episode': normal})

    def evaluate(task):
        observed.append((context.plan['episode'].copy(), task['parameters']))
        return [episode(path, kind='ignored') for path in task['paths']]

    context.evaluate = evaluate
    entries = [{'path': 'ordinary', 'absolute_path': 'ordinary', 'geometry_hash': 'ordinary',
                'group': 'speed', 'kind': 'isolated'},
               {'path': 'long', 'absolute_path': 'long', 'geometry_hash': 'long',
                'group': 'long_case', 'kind': 'long_case'}]
    rows = study.evaluate_policy(context, {'parameters': {'base': 1.}, 'entries': entries, 'cap': .5})
    assert observed[0][0] == normal
    assert observed[1][0]['maximum_time_limit'] == 360.
    assert context.plan['episode'] is normal
    assert [r['kind'] for r in rows] == ['isolated', 'long_case']
    assert all(r['analysis_role'] == 'development-tuning' for r in rows)


def test_local_adapter_restores_episode_after_case_error():
    normal = asdict(EpisodeConfig())
    context = SimpleNamespace(plan={'episode': normal})

    def fail(task):
        raise RuntimeError('case failed')

    context.evaluate = fail
    entry = {'path': 'long', 'absolute_path': 'long', 'geometry_hash': 'long',
             'group': 'long_case', 'kind': 'long_case'}
    with pytest.raises(RuntimeError, match='case failed'):
        study.evaluate_policy(context, {'parameters': {}, 'entries': [entry], 'cap': 1.})
    assert context.plan['episode'] is normal


def test_pairing_discloses_shared_finish_denominators():
    left = result([episode('a'), episode('b', completed=False), episode('c')])
    right = result([episode('a'), episode('b'), episode('c', completed=False)])
    paired = study.paired(left, right)
    assert paired['paired_tracks'] == 3 and paired['shared_finishes'] == 1
    assert paired['candidate_only_finishes'] == paired['control_only_finishes'] == 1
    assert paired['completion_time_s']['denominator'] == 1
    assert paired['command_jitter']['denominator'] == 1


def test_failed_coarse_search_can_continue_without_becoming_eligible():
    failed = study.summarize(validation(speed=.4, case_completed=False), 1.)
    assert study.promote({'current-tune': failed}, 12, ('current-tune',)) == ['current-tune']
    assert study.select({'current-tune': failed}) == (None, [])


class StubPool:
    _processes = 1

    def __init__(self):
        self.calls = 0

    def imap(self, worker, tasks):
        self.calls += len(tasks)
        return iter([{**validation(), 'wall_seconds': .01} for _ in tasks])


def test_atomic_candidate_cache_resumes_and_rejects_changed_inputs(tmp_path):
    pool = StubPool()
    policies = {'current-tune': {'base': 1.}}
    entries = [{'path': 'a', 'geometry_hash': 'a', 'kind': 'isolated'}]
    first = study.batch(pool, tmp_path, {'protocol_sha256': 'abc'}, policies, entries, 1., 'coarse')
    second = study.batch(pool, tmp_path, {'protocol_sha256': 'abc'}, policies, entries, 1., 'coarse')
    assert pool.calls == 1 and first == second
    assert not list(tmp_path.rglob('*.tmp'))
    with pytest.raises(ValueError, match='Stale candidate cache'):
        study.batch(pool, tmp_path, {'protocol_sha256': 'changed'}, policies, entries, 1., 'coarse')
    with pytest.raises(ValueError, match='Stale candidate cache'):
        study.batch(pool, tmp_path, {'protocol_sha256': 'abc'}, {'current-tune': {'base': .75}}, entries, 1., 'coarse')
