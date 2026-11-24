import pytest

import run_fast_neat_replicates as launcher


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher.study, "OUTPUT", tmp_path)
    monkeypatch.setattr(launcher.study, "verify", lambda path: {})
    monkeypatch.setattr(launcher.study, "finalize", lambda *args: None)
    monkeypatch.setattr(launcher.study, "main", calls.append)
    return tmp_path, calls


def test_preview_never_dispatches_evolution(dispatch):
    _, calls = dispatch
    launcher.main([])
    assert calls == []


def test_start_runs_pending_seeds_in_order_and_skips_completed(dispatch):
    root, calls = dispatch
    directory = root / "seed-03/training"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text('{"complete": true}')
    launcher.main(["--start"])
    assert calls == [["train", "--seed", str(seed)] for seed in (2, 4, 5)]


def test_resume_distinguishes_started_and_new_seeds(dispatch):
    root, calls = dispatch
    checkpoint = root / "seed-02/training/checkpoints/latest.pkl.gz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (root / "seed-02/start.json").write_text("{}")
    launcher.main(["--seeds", "2", "3", "--start", "--resume"])
    assert calls == [["train", "--seed", "2", "--resume"], ["train", "--seed", "3"]]


def test_invalid_later_seed_prevents_all_dispatch(dispatch):
    root, calls = dispatch
    (root / "seed-03").mkdir()
    (root / "seed-03/start.json").write_text("{}")
    with pytest.raises(SystemExit):
        launcher.main(["--start", "--resume"])
    assert calls == []


def test_changed_frozen_inputs_prevent_dispatch(dispatch, monkeypatch):
    _, calls = dispatch
    def reject(path):
        raise ValueError("Frozen study file changed")
    monkeypatch.setattr(launcher.study, "verify", reject)
    with pytest.raises(ValueError, match="Frozen study file changed"):
        launcher.main(["--start"])
    assert calls == []


def test_memory_safe_dispatch_records_change_and_restores_engine(dispatch, monkeypatch):
    import line_follower.experiments as engine
    from fast_study_memory import MemorySafeEvaluator
    root, calls = dispatch
    original = engine.Evaluator
    def train(command):
        assert engine.Evaluator is MemorySafeEvaluator
        calls.append(command)
    monkeypatch.setattr(launcher.study, "main", train)
    launcher.main(["--seeds", "5", "--start", "--memory-safe"])
    assert calls == [["train", "--seed", "5"]]
    assert len(list((root / "seed-05/memory-recovery").glob("*/record.json"))) == 1
    assert engine.Evaluator is original


def test_worker_override_is_recorded_and_scoped(dispatch, monkeypatch):
    import json
    from fast_study_memory import MemorySafeEvaluator
    root, calls = dispatch
    def train(command):
        assert MemorySafeEvaluator.WORKERS == 2
        calls.append(command)
    monkeypatch.setattr(launcher.study, "main", train)
    launcher.main(["--seeds", "5", "--start", "--memory-safe", "--evaluation-workers", "2"])
    record, = (root / "seed-05/memory-recovery").glob("*/record.json")
    assert json.loads(record.read_text())["runtime_workers_override"] == 2
    assert MemorySafeEvaluator.WORKERS is None


def test_worker_override_requires_recorded_memory_mode(dispatch):
    _, calls = dispatch
    with pytest.raises(SystemExit):
        launcher.main(["--start", "--evaluation-workers", "2"])
    assert not calls
