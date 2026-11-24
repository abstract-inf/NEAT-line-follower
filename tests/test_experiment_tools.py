"""Exercise evaluation, serialization, and UI tools without evolving a population."""

from dataclasses import asdict, replace
from importlib.resources import files
import json
import os
import random
import time

import neat
import numpy as np
import pytest

from line_follower.config import EpisodeConfig, RobotConfig, freeze_geometry, verify_geometry
from line_follower.controllers import load_model, load_neat_config, save_model
from line_follower.experiments import Evaluator, write_json, _restore_logs, _save_checkpoint
from line_follower.tracks import straight_track


def test_fresh_launcher_sessions_preserve_existing_runs_and_reserve_unique_folders(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from run_training import new_run_directory

    base = tmp_path / "session"
    base.mkdir()
    sentinel = base / "previous-results.csv"
    sentinel.write_text("preserve this run", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(lambda _: new_run_directory(base, create=True), range(4)))
    assert {path.name for path in outputs} == {"session-02", "session-03", "session-04", "session-05"}
    assert all(path.is_dir() for path in outputs)
    assert sentinel.read_text() == "preserve this run"


def test_launcher_dry_run_path_selection_creates_nothing(tmp_path):
    from run_training import new_run_directory

    base = tmp_path / "uncreated-parent" / "session"
    assert new_run_directory(base) == base
    assert not base.parent.exists()


def test_launcher_resume_keeps_saved_schedule_and_timeout_without_rewriting_globals(tmp_path):
    from run_training import session_training_settings
    from line_follower.experiments import TrainConfig

    saved = TrainConfig(generations=5, time_limit=None, backend="gpu", gpu_batch_size=200)
    requested = replace(saved, generations=20, time_limit=60)
    checkpoint = tmp_path / "checkpoint.pkl.gz"
    _save_checkpoint(checkpoint, {"settings": saved})
    effective, overrides = session_training_settings(requested, checkpoint)
    assert effective == saved
    assert overrides == {"generations": {"requested": 20, "checkpoint": 5},
                         "time_limit": {"requested": 60, "checkpoint": None}}
    assert session_training_settings(requested, None) == (requested, {})


def test_progress_file_lock_does_not_abort_evaluation(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from line_follower.experiments import _progress_reporter

    original = Path.replace
    def locked_progress(path, target):
        if Path(target).name == "progress.json":
            raise PermissionError("Windows reader holds progress.json")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", locked_progress)
    report = _progress_reporter(tmp_path, 2, "training")
    report({"completed_episodes": 570, "total_episodes": 804})
    assert "episodes 570/804" in capsys.readouterr().out


def test_essential_metadata_retries_transient_locks_and_does_not_hide_persistent_failure(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.replace
    attempts = []
    def briefly_locked(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("Windows reader")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", briefly_locked)
    monkeypatch.setattr("line_follower.experiments.time.sleep", lambda seconds: None)
    assert write_json(tmp_path / "run.json", {"saved": True})
    assert len(attempts) == 3
    assert json.loads((tmp_path / "run.json").read_text()) == {"saved": True}
    def permanently_locked(path, target):
        raise PermissionError("Persistent file lock")
    monkeypatch.setattr(Path, "replace", permanently_locked)
    with pytest.raises(PermissionError):
        write_json(tmp_path / "essential.json", {"saved": True})


def test_recovery_audit_accepts_only_observer_and_logging_changes(tmp_path):
    import zipfile
    from prepare_resume import audit_sources

    package = tmp_path / "line_follower"
    package.mkdir()
    before = {"viewer.py": "DISPLAY = 1\n", "fitness.py": "FITNESS = 1\n",
              "experiments.py": "def write_json():\n    return 1\ndef train():\n    return 1\n"}
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, source in before.items():
            archive.writestr(f"line_follower/{name}", source)
            (package / name).write_bytes(source.encode())
    (package / "viewer.py").write_text("DISPLAY = 2\n")
    (package / "experiments.py").write_text(before["experiments.py"].replace("def write_json():\n    return 1", "def write_json():\n    return 2"))
    with zipfile.ZipFile(path) as archive:
        _, changes = audit_sources(archive, tmp_path)
        assert len(changes) == 2
        (package / "fitness.py").write_text("FITNESS = 2\n")
        with pytest.raises(ValueError, match="Scientific/non-observer"):
            audit_sources(archive, tmp_path)
        (package / "fitness.py").write_bytes(before["fitness.py"].encode())
        (package / "experiments.py").write_text(before["experiments.py"].replace("def train():\n    return 1", "def train():\n    return 2"))
        with pytest.raises(ValueError, match="Scientific/non-observer"):
            audit_sources(archive, tmp_path)


def test_gpu_capacity_recovery_rejects_smaller_buffers_and_physics_changes():
    from prepare_resume import capacity_increase_only

    source = b"CONTACTS_PER_WORLD = 32\nCONSTRAINTS_PER_WORLD = 128\ndef step():\n    return 1\n"
    increased = source.replace(b"= 32", b"= 512").replace(b"= 128", b"= 2048")
    assert capacity_increase_only(source, increased)
    assert not capacity_increase_only(source, increased.replace(b"return 1", b"return 2"))
    assert not capacity_increase_only(source, source.replace(b"= 32", b"= 16"))


def fixed_genomes():
    """Network fixtures only: no population, selection, mutation, or evolution."""
    config = load_neat_config(files("line_follower").joinpath("configs/neat.txt"))
    previous = random.getstate()
    random.seed(42)
    genomes = []
    for key in (1, 2):
        genome = neat.DefaultGenome(key)
        genome.configure_new(config.genome_config)
        genomes.append(genome)
    random.setstate(previous)
    return config, genomes


def test_fitness_console_uses_selection_fitness_and_previous_saved_generation(tmp_path, capsys):
    from types import SimpleNamespace
    from line_follower.experiments import _report_training_fitness, _restore_logs

    (tmp_path / "generations.csv").write_text(
        "generation,training_mean,training_best\n0,0.3,0.35\n", encoding="utf-8")
    evaluated = {10: [SimpleNamespace(score=.2), SimpleNamespace(score=.8)],
                 20: [SimpleNamespace(score=.4), SimpleNamespace(score=.6)]}
    tracks = [straight_track(.4), straight_track(.6)]
    _report_training_fitness(tmp_path, 1, evaluated, tracks)
    output = capsys.readouterr().out
    # The mean of all episode scores is .5, but selection uses mean/min fitness.
    assert "training fitness: average=0.400000, best=0.450000" in output
    assert "change: average=+0.100000, best=+0.100000" in output
    assert "average=0.300000, best=0.400000 (genome 20)" in output
    assert "average=0.700000, best=0.800000 (genome 10)" in output
    rows = (tmp_path / "episode-fitness.csv").read_text()
    assert "average_score,best_score,best_genome_id" in rows
    _restore_logs(tmp_path, 1)
    assert len((tmp_path / "episode-fitness.csv").read_text().splitlines()) == 1


def test_serial_and_two_persistent_workers_evaluate_identically(tmp_path):
    config, genomes = fixed_genomes()
    tracks = [straight_track(.4)]
    with Evaluator(RobotConfig(), EpisodeConfig(), config, workers=1, time_limit=.10) as serial:
        expected = serial.evaluate(genomes, tracks)
    with Evaluator(RobotConfig(), EpisodeConfig(), config, workers=2, time_limit=.10) as parallel:
        updates = []
        actual = parallel.evaluate(genomes, tracks, progress=updates.append,
                                   live={"directory": str(tmp_path), "token": "fixture"})
        repeated = parallel.evaluate(genomes, tracks)
    assert actual == expected == repeated
    assert updates[-1] == {"completed_episodes": 2, "total_episodes": 2}
    assert all(r.termination == "timeout" for episodes in actual.values() for r in episodes)
    snapshots = [json.loads(path.read_text()) for path in tmp_path.glob("genome-*.json")]
    assert len(snapshots) == 2
    assert all(s["robots"][0]["status"] == "timeout" for s in snapshots)


def test_model_roundtrip_preserves_outputs_and_configuration(tmp_path):
    config, genomes = fixed_genomes()
    path = tmp_path / "network-fixture.pkl"
    metadata = {"robot": asdict(RobotConfig()), "fixture_only": True}
    save_model(path, genomes[0], config, metadata)
    loaded, restored = load_model(path)
    expected = neat.nn.FeedForwardNetwork.create(genomes[0], config)
    observation = np.array([0]*7+[1]+[0]*7, dtype=float)
    assert np.allclose(loaded.act(observation, 1/60), expected.activate(observation.tolist()))
    assert restored == metadata
    assert not path.with_suffix(".pkl.tmp").exists()


def test_geometry_requires_explicit_selection_and_detects_changes(tmp_path):
    robot, path = RobotConfig(), tmp_path / "geometry.json"
    with pytest.raises(ValueError, match="Select geometry"):
        verify_geometry(path, robot)
    freeze_geometry(path, robot)
    assert verify_geometry(path, robot)["selected_for_training"]
    with pytest.raises(ValueError, match="differs"):
        verify_geometry(path, replace(robot, sensor_offset=.14))
    with pytest.raises(FileExistsError):
        freeze_geometry(path, robot)


def test_checkpoint_codec_and_log_rollback_without_evolution(tmp_path):
    import gzip
    import pickle

    path = tmp_path / "checkpoint.pkl.gz"
    state = {"robot": RobotConfig(), "random_state": random.getstate(), "next_generation": 1}
    _save_checkpoint(path, state)
    with gzip.open(path, "rb") as stream:
        assert pickle.load(stream) == state
    log = tmp_path / "generations.csv"
    log.write_text("generation,value\n0,committed\n1,uncommitted\n1,duplicate\n", encoding="utf-8")
    _restore_logs(tmp_path, next_generation=1)
    assert log.read_text().splitlines() == ["generation,value", "0,committed"]


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GPU_TEST") != "1", reason="Opt-in CUDA backend check")
def test_gpu_fixed_network_evaluation_matches_short_cpu_episodes():
    config, genomes = fixed_genomes()
    tracks = [straight_track(.4)]
    with Evaluator(RobotConfig(), EpisodeConfig(), config, time_limit=.10) as cpu:
        expected = cpu.evaluate(genomes, tracks)
    with Evaluator(RobotConfig(), EpisodeConfig(), config, backend="gpu", gpu_batch_size=2, time_limit=.10) as gpu:
        updates = []
        actual = gpu.evaluate(genomes, tracks, progress=updates.append)
        assert updates[-1]["completed_episodes"] == updates[-1]["total_episodes"] == 2
    for key in expected:
        a, b = expected[key][0], actual[key][0]
        assert a.control_steps == b.control_steps == 6
        assert a.termination == b.termination == "timeout"
        assert a.progress_m == pytest.approx(b.progress_m, abs=.0001)
        assert a.lateral_rmse_m == pytest.approx(b.lateral_rmse_m, abs=.0001)
        assert a.average_forward_speed_m_s == pytest.approx(b.average_forward_speed_m_s, abs=.001)


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GPU_TEST") != "1", reason="Opt-in CUDA backend check")
def test_gpu_batch_resets_and_handles_different_completion_times():
    config, genomes = fixed_genomes()
    genome = genomes[0]
    for node in genome.nodes.values():
        node.bias = float(np.log(.6/.4)/5)
        node.activation = "sigmoid"
    for connection in genome.connections.values():
        connection.weight = 0.0
    tracks = [straight_track(.030), replace(straight_track(.12), start=(.2, .3, .6))]
    with Evaluator(RobotConfig(), EpisodeConfig(), config, time_limit=2) as cpu:
        expected = cpu.validate(genome, tracks)
    with Evaluator(RobotConfig(), EpisodeConfig(), config, backend="gpu", gpu_batch_size=2, time_limit=2) as gpu:
        actual = gpu.validate(genome, tracks)
        repeated = gpu.validate(genome, tracks)
        assert gpu.gpu.padding_world_steps > 0  # Mixed-duration validation worlds still share one batch.
    assert all(result.completed for result in expected+actual+repeated)
    for a,b,c in zip(expected, actual, repeated):
        assert b.control_steps == c.control_steps
        assert a.control_steps == b.control_steps
        assert b.lateral_rmse_m < .0001
        assert b.progress == c.progress == 1


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GPU_TEST") != "1", reason="Opt-in CUDA backend check")
def test_gpu_compaction_preserves_results_and_reduces_wasted_physics(tmp_path, monkeypatch):
    from copy import deepcopy
    from line_follower import gpu_backend

    config, fixtures = fixed_genomes()
    genomes = []
    for key in range(64):
        genome = deepcopy(fixtures[0])
        genome.key = key
        for node in genome.nodes.values():
            node.bias = -1000 if key < 48 else float(np.log(.8/.2)/5)
            node.activation = "sigmoid"
        for connection in genome.connections.values():
            connection.weight = 0
        genomes.append(genome)
    with Evaluator(RobotConfig(), EpisodeConfig(stall_seconds=.3), config,
                   backend="gpu", gpu_batch_size=64, time_limit=.8) as evaluator:
        monkeypatch.setattr(gpu_backend, "COMPACTION_MIN_WORLDS", 1000)
        started = time.perf_counter()
        expected = evaluator.evaluate(genomes, [straight_track(.4)])
        baseline_wall = time.perf_counter()-started
        baseline_steps = evaluator.gpu.executed_world_steps
        baseline_padding = evaluator.gpu.padding_world_steps
        monkeypatch.setattr(gpu_backend, "COMPACTION_MIN_WORLDS", 16)
        started = time.perf_counter()
        actual = evaluator.evaluate(genomes, [straight_track(.4)])
        compact_wall = time.perf_counter()-started
        compact_steps = evaluator.gpu.executed_world_steps-baseline_steps
        compact_padding = evaluator.gpu.padding_world_steps-baseline_padding
    assert compact_steps < baseline_steps*.65
    assert compact_padding < baseline_padding*.1
    for key in expected:
        a, b = expected[key][0], actual[key][0]
        assert (a.termination, a.control_steps, a.completed) == (b.termination, b.control_steps, b.completed)
        assert a.score == pytest.approx(b.score, abs=1e-5)
        assert a.progress_m == pytest.approx(b.progress_m, abs=1e-5)
        assert a.average_forward_speed_m_s == pytest.approx(b.average_forward_speed_m_s, abs=1e-4)
    write_json(tmp_path / "compaction-benchmark.json", {
        "fixture_only_no_evolution": True, "worlds": 64, "baseline_wall_s": baseline_wall,
        "timing_caveat": "Single diagnostic; includes first graph setup for each batch shape, not a controlled training speedup estimate",
        "compacted_wall_s": compact_wall, "baseline_world_steps": baseline_steps,
        "compacted_world_steps": compact_steps, "baseline_padding_steps": baseline_padding,
        "compacted_padding_steps": compact_padding})


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GUI_TEST") != "1", reason="Opt-in native window smoke check")
def test_live_preview_with_fixture_network_without_training(tmp_path):
    from line_follower.viewer import training_preview

    config, genomes = fixed_genomes()
    straight_track().save(tmp_path / "track.json")
    save_model(tmp_path / "fixture.pkl", genomes[0], config,
               {"robot": asdict(RobotConfig()), "episode": asdict(EpisodeConfig())})
    write_json(tmp_path / "preview.json", {"generation": "fixture-only", "model": "fixture.pkl", "tracks": ["track.json"]})
    with training_preview(tmp_path, enabled=True) as process:
        time.sleep(2)
        assert process.is_alive()
        time.sleep(2)
        assert process.is_alive()
    assert process.exitcode == 0
