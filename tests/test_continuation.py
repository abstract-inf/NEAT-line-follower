"""Continuation checks use saved populations; never execute evolution."""

from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import pytest

from continue_training import PROJECT_ROOT, phase_checkpoint, prepare_continuation
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.experiments import TrainConfig, _save_checkpoint, environment_record
from line_follower.fitness import progress_fitness
from line_follower.suites import load_split, read_suite, training_batch
from line_follower.tracks import GeneratorConfig, straight_track, validate_track

PARENT = PROJECT_ROOT / "runs/28-10-2026"


def test_speed_bonus_requires_completion_and_keeps_completion_dominant():
    def score(p, completed, time, beta=.2):
        return progress_fitness(p, completed, time, 60, 0, 60, beta)
    assert score(.7, False, 5) == score(.7, False, 5, beta=0)
    assert score(0, False, 5) < 0
    assert score(1, True, 20) - score(1, True, 40) == pytest.approx(.1)
    assert score(1, True, 60) > score(.999, False, 0)
    with pytest.raises(ValueError):
        EpisodeConfig(completion_speed_bonus=-.1)
    with pytest.raises(ValueError):
        EpisodeConfig(completion_speed_bonus=1)


def test_checkpoint_retry_keeps_last_committed_state_if_replacement_stays_locked(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl.gz"
    _save_checkpoint(path, {"committed": 1})
    original = Path.replace
    attempts = []
    def briefly_locked(temporary, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("Windows reader lock")
        return original(temporary, target)
    monkeypatch.setattr(Path, "replace", briefly_locked)
    monkeypatch.setattr("line_follower.experiments.time.sleep", lambda seconds: None)
    _save_checkpoint(path, {"committed": 2})
    assert len(attempts) == 3
    import gzip, pickle
    with gzip.open(path, "rb") as stream:
        assert pickle.load(stream) == {"committed": 2}
    def permanently_locked(*args):
        raise PermissionError("Persistent lock")
    monkeypatch.setattr(Path, "replace", permanently_locked)
    with pytest.raises(PermissionError):
        _save_checkpoint(path, {"committed": 3})
    with gzip.open(path, "rb") as stream:
        assert pickle.load(stream) == {"committed": 2}


def test_continuation_profile_is_shared_reproducible_and_has_fresh_zigzags():
    fixed = straight_track()
    first = training_batch(0, "multi", 100, 200, fixed, set(), set(), GeneratorConfig(), 4, "zigzag")
    repeated = training_batch(0, "multi", 100, 200, fixed, set(), set(), GeneratorConfig(), 4, "zigzag")
    assert first == repeated
    assert [t.difficulty for t in first] == ["beginner", "intermediate", "advanced", "advanced"]
    assert all("zigzag" in first[i].features for i in (1, 3))
    # Each motif has three 90-degree turns and two 45-degree entrance/exit turns.
    assert sum(s.kind == "corner" and s.feature == "zigzag" for s in first[3].sections) >= 10
    assert all(not validate_track(t) for t in first)
    forbidden = {t.geometry_hash for t in first}
    second = training_batch(0, "multi", 101, 200, fixed, forbidden, set(), GeneratorConfig(), 4, "zigzag")
    assert not forbidden & {t.geometry_hash for t in second}
    with pytest.raises(ValueError, match="multiple of four"):
        TrainConfig(strategy="multi", track_profile="zigzag", episodes_per_genome=3)


def test_standard_generation_still_matches_original_saved_batch():
    if not PARENT.exists():
        pytest.skip("Local historical artifact is optional")
    record = json.loads((PARENT / "run.json").read_text())
    suite = Path(record["suite_path"])
    manifest = read_suite(suite)
    forbidden = {e["geometry_hash"] for split in ("validation", "test") for e in manifest["splits"][split]}
    actual = training_batch(0, "curriculum", 0, 100, load_split(suite, "training")[0],
                            forbidden, set(), GeneratorConfig())
    expected = json.loads((PARENT / "tracks/generation-0000/batch.json").read_text())["hashes"]
    assert [t.geometry_hash for t in actual] == expected


def test_dense_zigzag_generation_retries_failed_seeds():
    failures = []
    tracks = training_batch(0, "multi", 104, 200, straight_track(), set(), set(),
                            GeneratorConfig(), 4, "zigzag", generation_failures=failures)
    assert len(tracks) == 4
    assert all(not validate_track(t) for t in tracks)
    assert any(f["seed"] == 16842410141962136783 for f in failures)


def fixture_parent(tmp_path):
    """Serializable state-shaped fixture: no NEAT population is initialized."""
    from line_follower.config import freeze_geometry
    from line_follower.experiments import snapshot_sources, write_json
    from line_follower.suites import create_suite

    parent = tmp_path / "parent"
    (parent / "checkpoints").mkdir(parents=True)
    suite = create_suite(tmp_path / "suite", validation_count=1, test_count=1)
    robot, episode = RobotConfig(), EpisodeConfig()
    freeze_geometry(parent / "geometry-freeze.json", robot)
    import neat
    from importlib.resources import files
    from line_follower.controllers import load_neat_config
    config = load_neat_config(files("line_follower").joinpath("configs/neat.txt"))
    previous = random.getstate()
    random.seed(42)
    genomes = []
    for key in (1, 2):
        genome = neat.DefaultGenome(key)
        genome.configure_new(config.genome_config)
        genomes.append(genome)
    random.setstate(previous)
    species = SimpleNamespace(fitness_history=[1.9], last_improved=1, fitness=1.9, adjusted_fitness=.5,
                              members={g.key: g for g in genomes})
    population = SimpleNamespace(generation=100, population={g.key: g for g in genomes}, config=config,
                                 species=SimpleNamespace(species={1: species}), best_genome=genomes[0])
    state = {"checkpoint_version": 2, "neat_version": "0.92", "population": population,
             "settings": TrainConfig(generations=100), "robot": robot, "episode": episode,
             "suite_hash": hashlib.sha256(suite.read_bytes()).hexdigest(), "best_key": (.5, .6),
             "source_sha256": environment_record()["source_sha256"],
             "dependencies": environment_record()["dependencies"], "best_model_bytes": b"parent artifact",
             "seen_hashes": {"prior-track"}, "total_physics_steps": 1234, "random_state": random.getstate()}
    _save_checkpoint(parent / "checkpoints/latest.pkl.gz", state)
    snapshot_sources(parent / "source.zip")
    write_json(parent / "run.json", {"suite_path": str(suite), "settings": asdict(state["settings"]),
               "geometry_freeze": json.loads((parent / "geometry-freeze.json").read_text())})
    (parent / "frozen-validation").mkdir()
    load_split(suite, "validation")[0].save(parent / "frozen-validation/000.json")
    straight_track().save(parent / "single-training-track.json")
    return parent, state


def test_preparation_preserves_population_rng_parent_and_resumes_at_generation_100(tmp_path):
    parent, before = fixture_parent(tmp_path)
    original = {p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    output = tmp_path / "phase"
    report = prepare_continuation(parent, output, dry_run=True)
    assert report["settings"]["generations"] == 200
    assert not output.exists()
    prepare_continuation(parent, output)
    saved = phase_checkpoint(output)
    assert saved["population"].generation == 100
    assert saved["random_state"] == before["random_state"]
    assert saved["seen_hashes"] == before["seen_hashes"]
    for key, genome in saved["population"].population.items():
        old = before["population"].population[key]
        assert {i: vars(n) for i, n in genome.nodes.items()} == {i: vars(n) for i, n in old.nodes.items()}
        assert {i: vars(c) for i, c in genome.connections.items()} == {i: vars(c) for i, c in old.connections.items()}
    assert saved["population"].species.species[1].fitness_history == []
    assert saved["best_model_bytes"] is None
    assert saved["robot"].max_speed == .35
    assert saved["episode"].completion_speed_bonus == 0
    assert saved["settings"].episodes_per_genome == 10
    assert saved["settings"].time_limit is None
    assert saved["curriculum_state"].stage == 0  # Fixture has no training history.
    assert original == {p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_continuation(parent, output)


def test_preparation_refuses_unfinished_parent_and_source_tampering(tmp_path):
    parent, state = fixture_parent(tmp_path)
    state["population"].generation = 99
    _save_checkpoint(parent / "checkpoints/latest.pkl.gz", state)
    with pytest.raises(ValueError, match="not finished"):
        prepare_continuation(parent, tmp_path / "phase")
    state["population"].generation = 100
    state["source_sha256"] = "wrong"
    _save_checkpoint(parent / "checkpoints/latest.pkl.gz", state)
    with pytest.raises(ValueError, match="source.zip"):
        prepare_continuation(parent, tmp_path / "phase")


def test_phase_resume_retains_saved_research_controls_and_rejects_source_change(tmp_path, monkeypatch):
    parent, _ = fixture_parent(tmp_path)
    output = tmp_path / "phase"
    prepare_continuation(parent, output, additional_generations=7, max_speed=.32, completion_speed_bonus=.1)
    saved = phase_checkpoint(output)
    assert saved["settings"].generations == 107
    assert saved["robot"].max_speed == .32
    monkeypatch.setattr("continue_training.environment_record", lambda: {"source_sha256": "changed", "dependencies": {}})
    with pytest.raises(ValueError, match="source/dependencies changed"):
        phase_checkpoint(output)


def test_real_checkpoint_requires_matching_dependencies_before_evolution(tmp_path, monkeypatch):
    """Exercise the dependency gate and, on a matching setup, the train resume path."""
    if not PARENT.exists():
        pytest.skip("Local historical artifact is optional")
    import neat
    from contextlib import nullcontext
    from line_follower.experiments import train

    output = tmp_path / "real-phase"
    with gzip.open(PARENT / "checkpoints/latest.pkl.gz", "rb") as stream:
        parent = pickle.load(stream)
    if parent["dependencies"] != environment_record()["dependencies"]:
        with pytest.raises(ValueError, match="Dependencies changed since parent checkpoint"):
            prepare_continuation(PARENT, output)
        assert not output.exists()
        return
    prepare_continuation(PARENT, output)
    saved = phase_checkpoint(output)
    class EvaluationBoundaryReached(Exception):
        pass
    def no_evolution(population, evaluate, count):
        assert population.generation == 100
        assert count == 1
        batch = json.loads((output / "tracks/generation-0100/batch.json").read_text())
        assert batch["track_profile"] == "progressive"
        assert len(batch["tracks"]) == 10
        assert batch["curriculum_stage"] == 1
        raise EvaluationBoundaryReached
    monkeypatch.setattr(neat.Population, "run", no_evolution)
    monkeypatch.setattr("line_follower.experiments.Evaluator", lambda *args: nullcontext(SimpleNamespace(gpu=None)))
    record = json.loads((output / "run.json").read_text())
    with pytest.raises(EvaluationBoundaryReached):
        train(output, record["suite_path"], saved["settings"], saved["robot"], saved["episode"],
              resume=output / "checkpoints/latest.pkl.gz", geometry_lock=output / "geometry-freeze.json")


def test_training_records_and_curriculum_survive_resume_with_mocked_evolution(tmp_path, monkeypatch):
    """Synthetic episode fixtures exercise I/O; no physics or NEAT evolution runs."""
    import neat
    from dataclasses import fields
    from line_follower.episodes import EpisodeResult
    from line_follower.experiments import train

    parent, state = fixture_parent(tmp_path)
    population = neat.Population.__new__(neat.Population)  # No constructor/population creation.
    vars(population).update(vars(state["population"]))
    state["population"] = population
    _save_checkpoint(parent / "checkpoints/latest.pkl.gz", state)
    output = tmp_path / "phase"
    prepare_continuation(parent, output)
    def fixture_results(tracks):
        results = []
        for track in tracks:
            values = {f.name: 0 for f in fields(EpisodeResult)}
            values.update(track_name=track.name, track_hash=track.geometry_hash, track_seed=track.seed,
                split=track.split, difficulty=track.difficulty, features=";".join(track.features),
                completed=True, termination="completed", error="", elapsed_s=.1, completion_time_s=.1,
                time_limit_s=45, progress=1.0, route_length_m=track.route.length, score=1.99)
            results.append(EpisodeResult(**values))
        return results
    class FixtureEvaluator:
        gpu = None
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def evaluate(self, genomes, tracks, *args, **kwargs):
            return {g.key: fixture_results(tracks) for g in genomes}
        def validate(self, genome, tracks, *args):
            return fixture_results(tracks)
    def mocked_evolution(self, callback, count):
        callback(list(self.population.items()), self.config)
        self.generation += 1  # No selection, reproduction, mutation or weight changes.
    monkeypatch.setattr(neat.Population, "run", mocked_evolution)
    monkeypatch.setattr("line_follower.experiments.Evaluator", FixtureEvaluator)
    for expected in (101, 102):
        saved = phase_checkpoint(output)
        record = json.loads((output / "run.json").read_text())
        train(output, record["suite_path"], saved["settings"], saved["robot"], saved["episode"],
              resume=output / "checkpoints/latest.pkl.gz", geometry_lock=output / "geometry-freeze.json", stop_after=1)
        checkpoint = phase_checkpoint(output)
        assert checkpoint["population"].generation == expected
        assert checkpoint["curriculum_state"].age == expected-100
        assert len(checkpoint["seen_hashes"]) == 1+10*(expected-100)
    assert (output / "diagnostics.csv").exists() and (output / "curriculum.csv").exists()


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GPU_TEST") != "1", reason="Opt-in CUDA backend check")
def test_continuation_speed_and_bonus_use_the_same_cpu_and_gpu_accounting():
    import math
    import neat
    from importlib.resources import files
    from line_follower.controllers import load_neat_config
    from line_follower.experiments import Evaluator

    config = load_neat_config(files("line_follower").joinpath("configs/neat.txt"))
    previous = random.getstate()
    random.seed(42)
    genome = neat.DefaultGenome(1)
    genome.configure_new(config.genome_config)
    random.setstate(previous)
    for node in genome.nodes.values():
        node.bias, node.activation = math.log(.6/.4)/5, "sigmoid"
    for connection in genome.connections.values():
        connection.weight = 0
    robot, episode = RobotConfig(max_speed=.35), EpisodeConfig(completion_speed_bonus=.2)
    with Evaluator(robot, episode, config, time_limit=2) as cpu:
        expected = cpu.validate(genome, [straight_track(.03)])[0]
    with Evaluator(robot, episode, config, backend="gpu", gpu_batch_size=1, time_limit=2) as gpu:
        actual = gpu.validate(genome, [straight_track(.03)])[0]
    assert expected.completed and actual.completed
    assert actual.score == pytest.approx(expected.score, abs=.004)
    assert actual.score == pytest.approx(progress_fitness(actual.progress, True, actual.elapsed_s,
        actual.time_limit_s, actual.legacy_return, actual.control_steps, .2))
