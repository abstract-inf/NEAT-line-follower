from collections import Counter
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

import run_neat_speed_study as study
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.experiments import append_csv, write_json
from line_follower.pid import ResearchPIDConfig
from line_follower.pid_experiments import evaluate_candidate, geometric_tracking_error
from line_follower.tracks import straight_track


@pytest.fixture
def prepared_study(tmp_path, monkeypatch):
    output = tmp_path / "study"
    prepared = output / "prepared"
    suite = prepared / "suite/suite.json"
    track = replace(straight_track(.4), split="test")
    track.save(suite.parent / "test/000.json")
    write_json(suite, {"suite_version": 1, "generator_version": "1.0", "splits": {
        "test": [{"path": "test/000.json", "geometry_hash": track.geometry_hash}],
    }})
    pid_names = ("pid-050.json", "pid-150.json")
    for name in pid_names:
        write_json(prepared / "inputs" / name, asdict(ResearchPIDConfig()))
    plan = {
        "robot": asdict(RobotConfig()),
        "episode": asdict(EpisodeConfig(corridor_half_width=.10, departure_seconds=2)),
        "software": {"source_sha256": "example"},
        "suite_sha256": study.digest(suite),
        "inputs": {name: study.digest(prepared / "inputs" / name) for name in pid_names},
        "training": {str(seed): {phase: asdict(study.training_settings(seed, phase))
                                for phase in ("low", "ramp")} for seed in study.SEEDS},
    }
    write_json(prepared / "protocol.json", plan)
    monkeypatch.setattr(study, "software_fingerprints", lambda: plan["software"])
    monkeypatch.setattr(study, "read_model_artifact", lambda path: {"metadata": study.read_json(path)})
    calls = []

    def train_stub(directory, suite_path, settings, robot, episode, **kwargs):
        calls.append((directory, settings, kwargs))
        assert suite_path == suite and episode.departure_seconds == 2
        ramp = settings.track_profile == "zigzag"
        rows = []
        for generation in range(settings.generations):
            cap = settings.speed_at(generation, robot.max_speed)
            completion = .95 if ramp and generation == 10 else .2
            if generation == (141 if ramp else 120):
                completion = .7
            rows.append({"generation": generation, "wheel_speed_cap_m_s": cap,
                         "validation_completion_rate": completion, "validation_score": 1.3})
        winner = rows[141 if ramp else 120]
        (directory / "generations.csv").unlink(missing_ok=True)
        append_csv(directory / "generations.csv", rows, list(rows[0]))
        write_json(directory / "run.json", {"settings": asdict(settings), "robot": asdict(robot),
                                            "episode": asdict(episode), "suite_sha256": study.digest(suite)})
        write_json(directory / "best.pkl", {
            "generation": winner["generation"], "evolution_seed": settings.seed,
            "robot": {**asdict(robot), "max_speed": winner["wheel_speed_cap_m_s"]},
            "validation_key": [winner["validation_completion_rate"], winner["validation_score"]],
            "episode": asdict(episode), "suite_sha256": study.digest(suite), "fitness_version": study.FITNESS_VERSION,
            "controller_semantics": "mirrored-sensor-history-v1", "training_backend": "cpu", "validation_backend": "cpu",
        })
        checkpoint = directory / "checkpoints/latest.pkl.gz"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b"fake checkpoint")
        write_json(directory / "status.json", {"complete": True, "next_generation": settings.generations,
                                               "target_generations": settings.generations})

    monkeypatch.setattr(study, "train", train_stub)

    def command(*args):
        study.main(["--output", str(output), *args])

    return command, output, prepared, track, calls


def test_stages_use_distinct_seeds_and_exact_caps():
    phases = [study.training_settings(seed, phase) for seed in study.SEEDS for phase in ("low", "ramp")]
    assert len({phase.seed for phase in phases}) == 10
    for index, settings in enumerate(phases):
        assert (settings.generations, settings.population, settings.workers, settings.episodes_per_genome,
                settings.backend) == (150, 100, 8, 4, "cpu")
        caps = Counter(settings.speed_at(g, 0.35) for g in range(settings.generations))
        if index % 2 == 0:
            assert settings.track_profile == "corner"
            assert caps == {.50: 150}
        else:
            assert settings.track_profile == "zigzag"
            assert caps == {.65: 10, .80: 10, 1.00: 10, 1.25: 10, 1.50: 110}


def test_stage_dispatch_freezes_winners_and_uses_same_seed_parent(prepared_study):
    command, output, _, _, calls = prepared_study
    with pytest.raises(ValueError, match="Seed 1 low training is incomplete"):
        command("train-ramp", "--seed", "1")
    assert not calls

    command("train-low", "--seed", "1")
    low, settings, kwargs = calls[-1]
    assert len(calls) == 1 and kwargs["initial_model"] is None
    assert kwargs["neat_path"].name == "neat-corner.txt"
    assert study.read_json(low / "selection.json")["generation"] == 120
    with pytest.raises(ValueError, match="Seed 2 low training is incomplete"):
        command("train-ramp", "--seed", "2")

    command("train-ramp", "--seed", "1")
    ramp, ramp_settings, kwargs = calls[-1]
    assert len(calls) == 2 and kwargs["initial_model"] == low / "best.pkl"
    assert ramp_settings.seed != settings.seed
    assert study.read_json(ramp / "selection.json")["generation"] == 141
    assert study.read_json(output / "seed-01/ramp-start.json")["parent_sha256"] == study.digest(low / "best.pkl")

    command("train-ramp", "--seed", "1", "--resume")
    assert len(calls) == 2
    start = output / "seed-01/ramp-start.json"
    write_json(start, {**study.read_json(start), "parent_sha256": "changed"})
    with pytest.raises(ValueError, match="saved stage inputs differ"):
        command("train-ramp", "--seed", "1", "--resume")
    assert len(calls) == 2


def test_training_resume_uses_committed_checkpoint(prepared_study, monkeypatch):
    command, output, _, _, calls = prepared_study
    real_stub = study.train

    def interrupted(*args, **kwargs):
        real_stub(*args, **kwargs)
        write_json(args[0] / "status.json", {"complete": False, "next_generation": 3})
        raise RuntimeError("interrupted")

    monkeypatch.setattr(study, "train", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        command("train-low", "--seed", "1")
    monkeypatch.setattr(study, "train", real_stub)
    command("train-low", "--seed", "1", "--resume")
    assert calls[-1][2]["resume"] == output / "seed-01/low/checkpoints/latest.pkl.gz"
    assert (output / "seed-01/low/selection.json").is_file()


def test_assessment_freezes_all_policies_before_opening_test(prepared_study, monkeypatch):
    command, output, _, track, calls = prepared_study
    load = study.load_split
    frozen = output / "assessment/frozen"

    def load_test(suite, split):
        record = study.read_json(frozen / "selection-freeze.json")
        assert record["frozen_before_test_geometry_load"] and len(record["policies"]) == 12
        return load(suite, split)

    monkeypatch.setattr(study, "load_split", load_test)
    with pytest.raises(ValueError, match="Seed 1 low training is incomplete"):
        command("assess")
    assert not calls and not frozen.exists()
    for seed in study.SEEDS:
        command("train-low", "--seed", str(seed))
        command("train-ramp", "--seed", str(seed))

    class LocalPool:
        def Pool(self, workers):
            assert workers == 8
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def imap(self, evaluate, jobs):
            return map(evaluate, jobs)

    def evaluate_stub(job):
        assert job["geometric_tracking"] and len(job["tracks"]) == 1
        completed = job["candidate"] != "pid-150"
        return [{"track_hash": track.geometry_hash, "evaluation_speed_limit_m_s": json.loads(job["robot_json"])["max_speed"],
                 "completed": completed, "completion_time_s": 3.0 if completed else None, "error": "",
                 "termination": "completed" if completed else "off_route", "progress": float(completed),
                 "geometry_lateral_rmse_m": 0.0, "geometry_lateral_mae_m": 0.0,
                 "average_forward_speed_m_s": 0.3, "command_jitter": 0.05}]

    monkeypatch.setattr(study.mp, "get_context", lambda _: LocalPool())
    monkeypatch.setattr(study, "evaluate_candidate", evaluate_stub)
    command("assess")
    results = output / "assessment/results"
    assert study.read_json(results / "status.json")["episodes"] == 24
    summaries = study.read_csv(results / "test-summary.csv")
    assert len(summaries) == 24 and {float(row["command_jitter"]) for row in summaries} == {.05}
    assert len(study.read_csv(results / "seed-spread.csv")) == 4
    primary = next(row for row in study.read_csv(results / "paired-comparisons.csv")
                   if (row["cap_m_s"], row["policy"], row["reference"]) == ("1.5", "seed-01-ramp", "pid-150"))
    assert primary["both_finished"] == "0" and primary["policy_mean_shared_time_s"] == ""


def test_geometric_error_uses_nearest_segment_and_endpoints():
    points = np.array([[0, 0], [1, 0], [1, 1]])
    positions = np.array([[-.3, 0], [.5, .4], [1.2, .5]])
    assert geometric_tracking_error(positions, points) == pytest.approx((math.sqrt(.29 / 3), .3))


def test_geometric_error_is_opt_in_for_fixed_policy_evaluation(tmp_path):
    path = tmp_path / "track.json"
    straight_track(.4).save(path)
    task = {"robot_json": json.dumps(asdict(RobotConfig(max_speed=.5))),
            "episode": asdict(EpisodeConfig(corridor_half_width=.10, departure_seconds=2)),
            "parameters": asdict(ResearchPIDConfig()), "candidate": "fixed-pid", "seed": 1,
            "tracks": [str(path)], "phase": "check"}
    regular, = evaluate_candidate(task)
    geometric, = evaluate_candidate({**task, "geometric_tracking": True})
    assert regular["completed"] and geometric["completed"]
    assert "geometry_lateral_rmse_m" not in regular
    assert geometric["geometry_lateral_mae_m"] <= geometric["geometry_lateral_rmse_m"] < .001
