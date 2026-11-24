"""Evaluate validation-selected speed checkpoints and two PID baselines on matched tracks."""

from collections import Counter
import csv
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model
from line_follower.episodes import Episode, TELEMETRY_FIELDS
from line_follower.experiments import append_csv, environment_record, write_json
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.simulation import Simulation
from line_follower.suites import read_suite
from line_follower.tracks import TrackMask, TrackSpec

RUN = ROOT / "runs/neat-speed-100-20261104"
CORNER = ROOT / "runs/neat-corner-100-20261104"
OUT = ROOT / "research/artifacts/neat-speed-results"
TRACE_SLOTS = (1, 2, 31, 32)


class PlainPID:
    """Standard PID on the normalized centroid with a fixed forward command."""

    def __init__(self, kp, ki, kd, base):
        self.kp, self.ki, self.kd, self.base = kp, ki, kd, base
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def act(self, readings, dt):
        active = np.flatnonzero(readings > 0)
        if not len(active):
            self.previous_error = None
            return np.array((self.base, self.base))
        error = float(np.mean(active)/7 - 1)
        derivative = (error - self.previous_error)/dt if self.previous_error is not None else 0.0
        self.integral += error * dt
        self.previous_error = error
        turn = self.kp*error + self.ki*self.integral + self.kd*derivative
        return np.clip((self.base + turn, self.base - turn), -1, 1)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def mask_for(path):
    return TrackMask(TrackSpec.load(path))


@lru_cache(maxsize=6)
def simulator(robot):
    return Simulation(robot)


def evaluate_task(task):
    policy, model, settings_json, robot, episode, path, slot, expected, trace = task
    # Controller state is reset by Episode. Config JSON makes the worker cache key immutable.
    controller = controller_for(policy, model, settings_json, robot)
    mask = mask_for(path)
    if mask.track.geometry_hash != expected:
        raise ValueError(f"Track geometry changed: {path}")
    trial = Episode(simulator(robot), mask.track, controller, episode, mask)
    absolute_error = 0.
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)
    stream = trace.open("w", newline="", encoding="utf-8") if trace else None
    try:
        writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS) if stream else None
        if writer:
            writer.writeheader()
        while not trial.done:
            sample = trial.advance(telemetry=writer is not None)
            absolute_error += abs(trial.route.lateral_error)
            if writer and sample:
                writer.writerow(sample)
    finally:
        if stream:
            stream.close()
    result = trial.result()
    return {"policy_id": policy, "evaluation_speed_limit_m_s": robot.max_speed, "track_slot": slot,
            "backend": "cpu", "mean_absolute_lateral_error_m": absolute_error/max(1, result.control_steps),
            "telemetry": trace.relative_to(ROOT).as_posix() if trace else "", **asdict(result)}


@lru_cache(maxsize=24)
def controller_for(policy, model, settings_json, robot):
    if model:
        return load_model(model)[0]
    settings = json.loads(settings_json)
    if policy == "pid-filtered":
        return ResearchPIDController(ResearchPIDConfig(**settings), robot)
    return PlainPID(**settings)


def main():
    status = read_json(RUN / "training/status.json")
    assessment = read_json(RUN / "assessment/status.json")
    freeze = read_json(RUN / "assessment/selection-freeze.json")
    plan = read_json(RUN / "protocol.json")
    suite = RUN / "suite/suite.json"
    original = RUN / "assessment/test-episodes.csv"
    parent = RUN / "initial-g94.pkl"
    final = RUN / "assessment/frozen-best.pkl"
    assert status["complete"] and assessment["complete"]
    assert digest(original) == assessment["results_sha256"]
    assert digest(suite) == freeze["suite_sha256"] == plan["suite_sha256"]
    assert digest(parent) == freeze["g94_sha256"] and digest(final) == freeze["final_sha256"]

    generations = read_csv(RUN / "training/generations.csv")
    assert [int(r["generation"]) for r in generations] == list(range(100))
    choices = {}
    for _, cap in plan["training"]["speed_schedule"]:
        candidates = [r for r in generations if float(r["wheel_speed_cap_m_s"]) == cap]
        best = max(candidates, key=lambda r: (float(r["validation_completion_rate"]), float(r["validation_score"])))
        gen = int(best["generation"])
        path = RUN / "training/champions" / f"generation-{gen:04d}.pkl"
        if cap == 1.5:
            assert gen == freeze["selected_generation"]
            path = final
        _, metadata = load_model(path)
        assert metadata["generation"] == gen and metadata["robot"]["max_speed"] == cap
        choices[f"speed-g{gen:02d}"] = {"generation": gen, "training_cap_m_s": cap,
                                       "validation_completion": float(best["validation_completion_rate"]),
                                       "validation_score": float(best["validation_score"]),
                                       "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
    baseline = read_json(CORNER / "assessment/selection-freeze.json")["pid"]["pid-filtered"]
    assert baseline["corner_supervision"] is False
    pid = {"pid-plain": {key: baseline[key] for key in ("kp", "ki", "kd", "base")},
           "pid-filtered": baseline}
    cases = [(policy, cap, ROOT / item["path"], {})
             for policy, item in choices.items() if item["training_cap_m_s"] != 1.5
             for cap in sorted({.50, 1.50, item["training_cap_m_s"]})]
    cases += [(policy, cap, None, config) for policy, config in pid.items() for cap in (.50, 1.50)]
    supplemental = OUT / "supplemental-episodes.csv"
    analysis_plan = OUT / "comparison-freeze.json"
    record = {"suite_sha256": digest(suite), "original_assessment_sha256": digest(original),
              "checkpoints_selected_by": "Within-speed validation completion then validation fitness. Earliest breaks exact ties.",
              "phase_checkpoints": choices, "pid_settings": pid,
              "conditions": [{"policy_id": p, "wheel_speed_cap_m_s": cap} for p, cap, _, _ in cases],
              "pid_no_line": "Plain: keep equal base wheel commands, hold integral, reset derivative history.",
              "retrospective": True, "backend": "cpu", "trace_slots": list(TRACE_SLOTS),
              "evaluator_sha256": digest(Path(__file__)),
              "source_sha256": environment_record()["source_sha256"]}
    if analysis_plan.exists() and read_json(analysis_plan) != record:
        raise ValueError("Comparison differs from its frozen inputs")
    if supplemental.exists():
        print(f"Saved supplemental assessment: {supplemental}")
        return
    write_json(analysis_plan, record)

    # The recorded evaluation plan precedes opening test geometry.
    entries = read_suite(suite)["splits"]["test"]
    assert len(entries) == 90 and [r["geometry_hash"] for r in entries] == freeze["test_hashes"]
    assert Counter(r["difficulty"] for r in entries) == dict.fromkeys(("beginner", "intermediate", "advanced"), 30)
    robot = RobotConfig(**plan["base_robot"])
    episode = EpisodeConfig(**plan["episode"])
    tasks = []
    for slot, entry in enumerate(entries):
        for policy, cap, model, settings in cases:
            trace = None
            if cap == 1.5 and policy in pid and slot in TRACE_SLOTS:
                trace = OUT / "telemetry" / f"{policy}-{slot:03d}.csv"
            tasks.append((policy, model, json.dumps(settings, sort_keys=True), replace(robot, max_speed=cap), episode,
                          suite.parent / entry["path"], slot, entry["geometry_hash"], trace))
    print(f"Evaluating {len(tasks)} supplemental matched episodes", flush=True)
    records = []
    with mp.get_context("spawn").Pool(8) as pool:
        for index, row in enumerate(pool.imap(evaluate_task, tasks, chunksize=3), 1):
            records.append(row)
            if index % 180 == 0:
                print(f"{index}/{len(tasks)} evaluated", flush=True)
    assert len({(r["policy_id"], r["evaluation_speed_limit_m_s"], r["track_hash"]) for r in records}) == len(tasks)
    temporary = supplemental.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    append_csv(temporary, records, list(records[0]))
    if any(r["error"] for r in records):
        temporary.replace(OUT / "technical-failures.csv")
        raise RuntimeError("At least one supplemental evaluation reported a technical error")
    temporary.replace(supplemental)
    write_json(OUT / "supplemental-status.json", {"episodes": len(records), "sha256": digest(supplemental)})
    print(supplemental, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
