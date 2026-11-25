"""Final test access requires all seeds and an unchanged controller freeze."""
import pytest

from neat_simple_study import confirmation_assessment as assessment
from neat_smooth_study.common import digest, write_json


def test_assessment_rejects_four_finished_seeds_before_loading_models_or_tracks(tmp_path):
    write_json(tmp_path / 'prepared/protocol.json', {})
    for seed in range(1, 5):
        write_json(tmp_path / f'seed-{seed:02d}/complete.json', {'placeholder': True})
    with pytest.raises(ValueError, match=r'incomplete seeds: \[5\]'):
        assessment.freeze_controllers(tmp_path, {})
    assert not (tmp_path / 'assessment/suite').exists()
    assert not (tmp_path / 'assessment/frozen/manifest.json').exists()


def test_existing_test_tracks_cannot_precede_controller_selection(tmp_path):
    write_json(tmp_path / 'prepared/protocol.json', {})
    write_json(tmp_path / 'assessment/suite/suite.json', {'test': []})
    with pytest.raises(ValueError, match='after loading final tracks'):
        assessment.freeze_controllers(tmp_path, {})


def test_a_frozen_controller_cannot_change_between_test_resumes(tmp_path):
    write_json(tmp_path / 'prepared/protocol.json', {})
    model = tmp_path / 'assessment/frozen/test.pkl'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'original')
    manifest = {'protocol_sha256': digest(tmp_path / 'prepared/protocol.json'),
                'controllers': {'test.pkl': {'sha256': digest(model)}}}
    write_json(model.parent / 'manifest.json', manifest)
    assert assessment.freeze_controllers(tmp_path, {}) == manifest
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='controller changed'):
        assessment.freeze_controllers(tmp_path, {})


def test_prepare_test_commits_models_before_calling_the_track_generator(tmp_path, monkeypatch):
    def freeze(run, plan):
        write_json(run / 'assessment/frozen/manifest.json', {'fixed': True})
        return {'fixed': True}
    def generate(*args):
        assert (tmp_path / 'assessment/frozen/manifest.json').exists()
        raise RuntimeError('Reached generator only after freeze')
    write_json(tmp_path / 'prepared/inputs/all-development-hashes.json', ['old-geometry'])
    monkeypatch.setattr(assessment, 'freeze_controllers', freeze)
    monkeypatch.setattr(assessment, 'protected_geometry', lambda _: set())
    monkeypatch.setattr(assessment, 'make_suite', generate)
    with pytest.raises(RuntimeError, match='only after freeze'):
        assessment.prepare_test(tmp_path, {})
