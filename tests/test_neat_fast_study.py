from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import run_neat_fast_study as launcher
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import read_model_artifact
from line_follower.episodes import EpisodeResult, run_episode
from line_follower.experiments import TrainConfig, write_json
from line_follower.fast_study import fast_training_batch, prepare_suite, quality_gate, speed_bank_slots, speed_course
from line_follower.fitness import fitness_components, progress_fitness
from line_follower.simulation import Simulation
from line_follower.tracks import straight_track, validate_track


class FixedPopulation:
    """Exercise the generation scheduler with one fixed genome and no reproduction."""

    def __init__(self, artifact):
        self.config = artifact["config"]
        self.population = {artifact["genome"].key: artifact["genome"]}
        self.generation = 0
        self.best_genome = None
        self.species = SimpleNamespace(species={
            1: SimpleNamespace(members=self.population, fitness_history=[], last_improved=0)})

    def run(self, evaluate, generations):
        assert generations == 1
        evaluate(list(self.population.items()), self.config)
        self.generation += 1


class FixedEvaluator:
    gpu = None

    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def validate(self, genome, tracks, progress=None, detailed=False):
        return [result(track.geometry_hash, track_name=track.name, route_length_m=track.route.length,
                       score=2.1, physics_steps=6000, control_steps=600) for track in tracks]

    def evaluate(self, genomes, tracks, progress=None, live=None):
        return {genome.key: self.validate(genome, tracks) for genome in genomes}


def result(track_hash, **changes):
    values = {name: 0 for name in EpisodeResult.__dataclass_fields__}
    values.update(track_name=track_hash, track_hash=track_hash, difficulty="speed", features="",
                  split="validation", completed=True, termination="completed", error="", progress=1,
                  elapsed_s=10, completion_time_s=10, time_limit_s=45, route_length_m=4,
                  geometry_lateral_rmse_m=.01, command_jitter=.005, straight_speed_m_s=1)
    values.update(changes)
    return EpisodeResult(**values)


def quality_fixture():
    rows = [result("mixed"), result("speed"), result("corner")]
    reference = {row.track_hash: asdict(row) for row in rows}
    reference["speed"]["dedicated_straight"] = True
    groups = {row.track_hash: row.track_hash for row in rows}
    return rows, reference, groups


def test_route_speed_fitness_rewards_only_completed_progress_and_logs_exact_terms():
    def score(t, completed=True, jitter=0):
        return progress_fitness(1 if completed else .99, completed, t, 45, 0, 100,
                                command_jitter=jitter, command_jitter_penalty=.2,
                                route_length=4, fitness_version="route-speed-1.0")
    assert score(4) > score(8) > score(16)
    assert score(4) > score(4, jitter=.5)
    assert score(45, jitter=1) > score(.5, completed=False)
    parts = fitness_components(.99, False, 10, 45, 0, 100, .5, .2, .2, 4, "route-speed-1.0")
    assert parts["speed_term"] == 0
    completed = fitness_components(1, True, 4, 45, 0, 100, .5, 0, .2, 4, "route-speed-1.0")
    assert completed["speed_term"] == .5 and sum(completed.values()) == score(4)
    with pytest.raises(ValueError):
        EpisodeConfig(fitness_version="unknown")
    assert EpisodeConfig().fitness_version == "progress-1.2"


def test_quality_gate_rejects_tradeoffs_and_tracks_missing_straight_goal():
    rows, reference, groups = quality_fixture()
    key, info = quality_gate(rows, reference, groups)
    assert key == (0, 3, -.01) and info["eligible"] and info["straight_goal"]
    cases = [
        ({"completed": False, "completion_time_s": None}, "speed_completion"),
        ({"geometry_lateral_rmse_m": .011}, "all_geometry_lateral_rmse_m"),
        ({"command_jitter": .006}, "all_command_jitter"),
    ]
    for changes, reason in cases:
        modified = [rows[0], replace(rows[1], **changes), rows[2]]
        key, info = quality_gate(modified, reference, groups)
        assert key is None and reason in info["reasons"]
    faster = [replace(row, completion_time_s=8) for row in rows]
    key, _ = quality_gate(faster, reference, groups)
    assert key[0] == 2
    slow_straight = [rows[0], replace(rows[1], straight_speed_m_s=.9), rows[2]]
    assert not quality_gate(slow_straight, reference, groups)[1]["straight_goal"]
    reference["speed"].update(straight_speed_m_s=None, completed=False, completion_time_s=None)
    assert not quality_gate(slow_straight, reference, groups)[1]["straight_goal"]
    with pytest.raises(ValueError):
        quality_gate(rows[:2], reference, groups)


def test_shared_success_metrics_cannot_hide_worse_tracking():
    rows, reference, groups = quality_fixture()
    # A failed reference track with high error must not hide degradation on shared finishes.
    reference["mixed"].update(completed=False, completion_time_s=None, geometry_lateral_rmse_m=1, command_jitter=1)
    rows[1] = replace(rows[1], geometry_lateral_rmse_m=.02, command_jitter=.01)
    key, info = quality_gate(rows, reference, groups)
    assert key is None and "shared_geometry_lateral_rmse_m" in info["reasons"]
    assert "shared_command_jitter" in info["reasons"]


def test_speed_bank_cycles_and_fresh_six_course_batches():
    bank = [speed_course(10000+i, i, "training") for i in range(20)]
    assert all(not validate_track(track) and track.resolution == .002 for track in bank)
    assert len({track.geometry_hash for track in bank}) == 20
    slots = [slot for generation in range(10) for slot in speed_bank_slots(99, generation)]
    assert sorted(slots) == list(range(20))
    assert speed_bank_slots(99, 10) == speed_bank_slots(99, 10)
    assert speed_bank_slots(100, 0) != speed_bank_slots(99, 0)
    seen = set()
    first = fast_training_batch(10, 0, bank, set(), seen)
    second = fast_training_batch(10, 1, bank, set(), seen)
    assert len(first) == len(second) == 6 and len(seen) == 8
    assert [track.difficulty for track in first[2:]] == ["intermediate", "advanced", "advanced", "advanced"]
    assert all(not validate_track(track) for track in first + second)
    for batch in (first, second):
        turns = [s.angle for track in batch[4:] for s in track.sections if s.kind == "corner"]
        assert sum(a > 0 for a in turns) == sum(a < 0 for a in turns) == 3
    with pytest.raises(ValueError):
        TrainConfig(strategy="multi", track_profile="fast", episodes_per_genome=4)


def test_detailed_episode_reports_straight_speed_and_fitness():
    class Constant:
        def reset(self):
            pass

        def act(self, observation, dt):
            return np.array([.7, .7])

    track = straight_track(4.1)
    robot = RobotConfig(max_speed=1.5)
    rules = EpisodeConfig(fitness_version="route-speed-1.0", command_jitter_penalty=.2)
    measured = run_episode(Simulation(robot), track, Constant(), rules, detailed=True)
    assert measured.completed and .98 < measured.straight_speed_m_s < 1.1
    assert measured.peak_forward_speed_m_s >= measured.average_forward_speed_m_s
    assert measured.geometry_lateral_rmse_m < .001 and measured.command_jitter == 0
    expected = progress_fitness(1, True, measured.elapsed_s, measured.time_limit_s, measured.legacy_return,
                                measured.control_steps, command_jitter_penalty=.2,
                                route_length=4.1, fitness_version="route-speed-1.0")
    assert measured.score == pytest.approx(expected)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    output = tmp_path / "study"
    p = output / "prepared"
    plan = {"software": {"source_sha256": "fixed"}, "robot": asdict(RobotConfig()),
            "episode": asdict(EpisodeConfig(fitness_version="route-speed-1.0")),
            "settings": {str(seed): launcher.normalized(asdict(launcher.settings(seed))) for seed in launcher.SEEDS},
            "inputs": {f"seed-{seed:02d}-low.pkl": f"parent-{seed}" for seed in launcher.SEEDS}}
    write_json(p / "protocol.json", plan)
    for seed in launcher.SEEDS:
        write_json(p / f"references/seed-{seed:02d}.json", {})
    calls = []
    monkeypatch.setattr(launcher, "verify", lambda path: plan)
    monkeypatch.setattr(launcher, "finalize", lambda *args: {"selected": None})

    def train_stub(*args, **kwargs):
        calls.append((args, kwargs))
        directory = args[0]
        checkpoint = directory / "checkpoints/latest.pkl.gz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"stub checkpoint")
        write_json(directory / "status.json", {"complete": False})
        raise RuntimeError("stub interruption")

    monkeypatch.setattr(launcher, "train", train_stub)
    return output, p, plan, calls


def test_user_commands_warm_start_own_seed_and_resume(prepared):
    output, p, plan, calls = prepared
    with pytest.raises(RuntimeError, match="stub interruption"):
        launcher.main(["--output", str(output), "train", "--seed", "3"])
    args, options = calls[-1]
    assert args[2].population == 100 and args[2].episodes_per_genome == 6 and args[2].generations == 300
    assert options["initial_model"] == p / "inputs/seed-03-low.pkl"
    assert options["quality_reference"] == p / "references/seed-03.json"
    with pytest.raises(RuntimeError, match="stub interruption"):
        launcher.main(["--output", str(output), "train", "--seed", "3", "--resume"])
    assert calls[-1][1]["resume"] == output / "seed-03/training/checkpoints/latest.pkl.gz"
    assert calls[-1][1]["initial_model"] is None
    start = output / "seed-03/start.json"
    write_json(start, {**launcher.read_json(start), "parent_sha256": "changed"})
    with pytest.raises(ValueError, match="Stage inputs changed"):
        launcher.main(["--output", str(output), "train", "--seed", "3", "--resume"])


def test_assessment_requires_finished_seeds_before_loading_test(prepared, monkeypatch):
    output, p, plan, _ = prepared
    monkeypatch.setattr(launcher, "load_split", lambda *args: pytest.fail("Test courses opened too early"))
    with pytest.raises(ValueError, match="Seed 1 must finish"):
        launcher.assess(output, p, plan)


def test_engine_checkpoints_before_any_qualified_winner_then_resumes(tmp_path, monkeypatch):
    import line_follower.experiments as engine
    from line_follower.tracks import GeneratorConfig
    from line_follower.suites import load_split
    parent = launcher.PREVIOUS / "assessment/frozen/seed-01-low.pkl"
    artifact = read_model_artifact(parent)
    suite = tmp_path / "suite/suite.json"
    entries = {"training": [], "validation": [], "test": []}
    for index in range(20):
        track = replace(straight_track(1 + index * .01), split="training")
        path = f"training/{index}.json"
        track.save(suite.parent / path)
        entries["training"].append({"path": path, "geometry_hash": track.geometry_hash, "group": "speed"})
    for index, group in enumerate(("mixed", "speed", "corner")):
        track = replace(straight_track(2 + index * .1), split="validation")
        path = f"validation/{index}.json"
        track.save(suite.parent / path)
        entries["validation"].append({"path": path, "geometry_hash": track.geometry_hash, "group": group})
    write_json(suite, {"suite_version": 1, "generator_version": "1.0",
                       "generator_config": asdict(GeneratorConfig()), "splits": entries})
    validation = load_split(suite, "validation")
    refs = {track.geometry_hash: asdict(result(track.geometry_hash)) for track in validation}
    reference = tmp_path / "reference.json"
    write_json(reference, refs)
    monkeypatch.setattr(engine, "warm_start_population", lambda *args: FixedPopulation(deepcopy(artifact)))
    monkeypatch.setattr(engine, "Evaluator", FixedEvaluator)
    monkeypatch.setattr(engine, "verify_geometry", lambda *args: {})
    training = replace(launcher.settings(1), generations=2, workers=1, speed_schedule=((0, .65), (1, 1.50)))
    rules = EpisodeConfig(fitness_version="route-speed-1.0")
    output = tmp_path / "training"
    engine.train(output, suite, training, RobotConfig(), rules, initial_model=parent,
                 quality_reference=reference, stop_after=1)
    assert not (output / "best.pkl").exists()
    checkpoint = output / "checkpoints/latest.pkl.gz"
    assert checkpoint.is_file()
    # A modified quality reference must be rejected before any further generation.
    altered = deepcopy(refs)
    altered[next(iter(altered))]["command_jitter"] = 1
    write_json(reference, altered)
    with pytest.raises(ValueError, match="Quality reference changed"):
        engine.train(output, suite, training, RobotConfig(), rules, resume=checkpoint, quality_reference=reference)
    write_json(reference, refs)
    engine.train(output, suite, training, RobotConfig(), rules, resume=checkpoint, quality_reference=reference)
    assert read_model_artifact(output / "best.pkl")["metadata"]["generation"] == 1
    assert launcher.read_json(output / "status.json")["complete"]
    assert [int(row["generation"]) for row in launcher.read_csv(output / "quality.csv")] == [0, 1]
    assert len(launcher.read_csv(output / "fitness-components.csv")) == 4


def test_protocol_schedule_and_prepared_file_integrity(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    sample = prepared / "input.json"
    write_json(sample, {"value": 1})
    software = {"source_sha256": "fixed"}
    plan = {"software": software, "prepared_files": {sample.name: launcher.digest(sample)},
            "settings": {str(seed): launcher.normalized(asdict(launcher.settings(seed))) for seed in launcher.SEEDS}}
    write_json(prepared / "protocol.json", plan)
    monkeypatch.setattr(launcher, "software", lambda: software)
    assert launcher.verify(prepared) == plan
    schedule = [launcher.settings(1).speed_at(g, 0) for g in range(300)]
    assert schedule == [.65]*50 + [.8]*50 + [1.]*50 + [1.25]*50 + [1.5]*100
    write_json(sample, {"value": 2})
    with pytest.raises(ValueError, match="Frozen study file changed"):
        launcher.verify(prepared)
    monkeypatch.setattr(launcher, "software", lambda: {"source_sha256": "changed"})
    with pytest.raises(ValueError, match="Source or dependencies changed"):
        launcher.verify(prepared)


def test_directory_publication_survives_windows_locks(tmp_path, monkeypatch):
    from pathlib import Path
    source, destination = tmp_path / "source", tmp_path / "published"
    source.mkdir()
    (source / "payload.txt").write_text("frozen input")
    write_json(source / "protocol.json", {"payload.txt": launcher.digest(source / "payload.txt")})

    def locked(*args):
        raise PermissionError("Windows folder lock")

    real_copy = launcher.shutil.copy2

    def copy_with_commit_check(src, dst, *args, **kwargs):
        if Path(src).name == "protocol.json":
            assert (destination / "payload.txt").read_text() == "frozen input"
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", locked)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(launcher.shutil, "copy2", copy_with_commit_check)
    launcher.publish_directory(source, destination, "protocol.json")
    assert launcher.read_json(destination / "protocol.json")["payload.txt"] == launcher.digest(destination / "payload.txt")


def test_completed_run_can_have_no_qualified_winner(tmp_path):
    directory = tmp_path / "training"
    write_json(directory / "status.json", {"complete": True, "next_generation": 300})
    launcher.append_csv(directory / "generations.csv",
                        [{"generation": g, "wheel_speed_cap_m_s": launcher.settings(1).speed_at(g, 0)} for g in range(300)],
                        ["generation", "wheel_speed_cap_m_s"])
    quality = [{"generation": g, "eligible": False, "reasons": "all_command_jitter", "completed": 60,
                "geometric_rmse_m": .01, "paired_time_change_s": -1} for g in range(300)]
    launcher.append_csv(directory / "quality.csv", quality, list(quality[0]))
    outcome = launcher.finalize(directory, tmp_path, {}, 1)
    assert outcome["selected"] is None
    assert int(outcome["diagnostic_tradeoff"]["generation"]) == 200
    (directory / "best.pkl").write_bytes(b"unqualified")
    with pytest.raises(ValueError, match="unqualified"):
        launcher.finalize(directory, tmp_path, {}, 1)


def test_suite_separation_assessment_freeze_summaries_and_plots(tmp_path, monkeypatch):
    from collections import Counter
    from line_follower.suites import read_suite, load_split
    output = tmp_path / "study"
    prepared = output / "prepared"
    protected = {speed_course(123, 0, "training").geometry_hash}
    suite = prepare_suite(prepared / "suite", protected)
    manifest = read_suite(suite)
    assert Counter(r["group"] for r in manifest["splits"]["validation"]) == {"mixed": 30, "speed": 20, "corner": 10}
    assert Counter(r["group"] for r in manifest["splits"]["test"]) == {"mixed": 90, "speed": 30, "corner": 30}
    assert not protected.intersection(r["geometry_hash"] for rows in manifest["splits"].values() for r in rows)
    for split in manifest["splits"]:
        assert all(not validate_track(track) for track in load_split(suite, split))
    plan = {"robot": asdict(RobotConfig()), "episode": asdict(EpisodeConfig())}
    write_json(prepared / "protocol.json", plan)
    inputs = prepared / "inputs"
    inputs.mkdir()
    for seed in launcher.SEEDS:
        (inputs / f"seed-{seed:02d}-ramp.pkl").write_bytes(b"frozen reference")
        directory = output / f"seed-{seed:02d}/training"
        write_json(directory / "selection.json", {"selected": None})
        launcher.append_csv(directory / "generations.csv", [{"generation": 0, "training_mean": 1}], ["generation", "training_mean"])
        launcher.append_csv(directory / "quality.csv", [{"generation": 0, "geometric_rmse_m": .01}], ["generation", "geometric_rmse_m"])
        parts = [{"generation": 0, "role": role, **fitness_components(1, True, 4, 45, 0, 100)} for role in ("training", "validation")]
        launcher.append_csv(directory / "fitness-components.csv", parts, list(parts[0]))
    for cap in launcher.CAPS:
        write_json(inputs / f"pid-{int(cap*100):03d}.json", {})
    monkeypatch.setattr(launcher, "finalize", lambda *args: {"selected": None})

    def check_test_load(*args):
        assert (output / "assessment/frozen/selection-freeze.json").is_file()
        return load_split(*args)

    def fixed_results(tasks, workers, expected):
        records = []
        for task in tasks:
            track = launcher.TrackSpec.load(task["tracks"][0])
            row = asdict(result(track.geometry_hash, features=";".join(track.features)))
            row.update(policy_id=task["candidate"], evaluation_speed_limit_m_s=json.loads(task["robot_json"])["max_speed"])
            # A missing straight segment must fail the speed goal even if others are fast.
            if task["slot_offset"] == 90:
                row.update(completed=False, completion_time_s=None, straight_speed_m_s=None)
            records.append(row)
        assert len(records) == expected and workers == 8
        return records

    monkeypatch.setattr(launcher, "load_split", check_test_load)
    monkeypatch.setattr(launcher, "collect", fixed_results)
    launcher.assess(output, prepared, plan)
    summaries = launcher.read_csv(output / "assessment/results/test-summary.csv")
    assert all(row["straight_goal"] == "False" for row in summaries if row["group"] in ("all", "speed"))
    launcher.assess(output, prepared, plan)  # Completed assessment is verified and reused.
    launcher.plot(output)
    assert len(list((output / "plots").glob("*.svg"))) == len(list((output / "plots").glob("*.png"))) == 4
