"""Rerun frozen controllers on the speed cohort with corrected evaluation."""

from dataclasses import asdict, replace
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from evaluate_speed_comparison import (
    ROOT, RUN, append_csv, digest, environment_record, evaluate_task,
    mask_for, read_csv, read_json, write_json,
)
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.experiments import snapshot_sources
from line_follower.fitness import FITNESS_VERSION

OUT = ROOT / "research/artifacts/neat-speed-results-corrected"


def evaluate_corrected(task):
    row = evaluate_task(task)
    if row["error"]:
        return row
    xy = np.loadtxt(ROOT / row["telemetry"], delimiter=",", skiprows=1, usecols=(2, 3), ndmin=2)
    route = mask_for(task[5]).track.route
    starts = route.points[:-1]
    delta = np.diff(route.points, axis=0)
    length2 = np.sum(delta*delta, axis=1)
    distances = []
    for start in range(0, len(xy), 64):
        block = xy[start:start+64]
        dx = block[:, None, 0]-starts[:, 0]
        dy = block[:, None, 1]-starts[:, 1]
        fraction = np.clip((dx*delta[:, 0]+dy*delta[:, 1])/length2, 0, 1)
        squared = (dx-fraction*delta[:, 0])**2+(dy-fraction*delta[:, 1])**2
        distances.extend(np.sqrt(np.min(squared, axis=1)).tolist())
    row["geometry_lateral_rmse_m"] = float(np.sqrt(np.mean(np.square(distances))))
    row["geometry_lateral_mae_m"] = float(np.mean(distances))
    return row


def main():
    started = time.perf_counter()
    plan = read_json(RUN / "protocol.json")
    previous = read_json(ROOT / "research/artifacts/neat-speed-results/comparison-freeze.json")
    suite = RUN / "suite/suite.json"
    assert digest(suite) == previous["suite_sha256"]
    entries = read_json(suite)["splits"]["test"]
    pid_path = ROOT / "runs/pid-gains-20261029/selected-pid_filtered.json"
    pid = read_json(pid_path)
    assert pid == previous["pid_settings"]["pid-filtered"] and not pid["corner_supervision"]
    parent, final = RUN / "initial-g94.pkl", RUN / "assessment/frozen-best.pkl"
    early = ROOT / previous["phase_checkpoints"]["speed-g00"]["path"]
    assert digest(final) == previous["phase_checkpoints"]["speed-g98"]["sha256"]
    assert digest(early) == previous["phase_checkpoints"]["speed-g00"]["sha256"]
    assert digest(parent) == plan["warm_start"]["sha256"]
    cases = [("g94-parent", .50, parent, {}), ("speed-g98", 1.50, final, {}),
             ("pid-filtered", .50, None, pid), ("pid-filtered", 1.50, None, pid),
             ("speed-g00", .50, early, {}), ("speed-g98", .50, final, {})]
    episode = replace(EpisodeConfig(**plan["episode"]), corridor_half_width=.10, departure_seconds=2.0)
    freeze = {"evaluator_version": FITNESS_VERSION, "episode": asdict(episode),
              "suite_sha256": digest(suite), "pid_sha256": digest(pid_path), "pid_settings": pid,
              "pid_tuning_cap_m_s": .35, "source": environment_record(), "retrospective": True,
              "metric": "Exact minimum chassis-center distance to any segment of the intended route polyline.",
              "conditions": [{"policy": p, "cap_m_s": c, "model_sha256": digest(m) if m else None}
                             for p, c, m, _ in cases]}
    results = OUT / "test-episodes.csv"
    rows = []
    if results.exists():
        status = read_json(OUT / "status.json")
        assert digest(results) == status["sha256"]
        saved = read_json(OUT / "evaluation-freeze.json")
        for key in ("evaluator_version", "episode", "suite_sha256", "pid_sha256", "pid_settings"):
            assert saved[key] == freeze[key], f"Assessment inputs changed: {key}"
        assert saved["source"]["source_sha256"] == freeze["source"]["source_sha256"]
        rows = read_csv(results)
        freeze["previous_results_sha256"] = status["sha256"]
    existing = {(r["policy_id"], float(r["evaluation_speed_limit_m_s"]), r["track_hash"]) for r in rows}
    robot = RobotConfig(**plan["base_robot"])
    tasks = [(p, m, json.dumps(settings, sort_keys=True), replace(robot, max_speed=c), episode,
              suite.parent / entry["path"], slot, entry["geometry_hash"],
              OUT / "telemetry" / f"{slot:03d}-{c:.2f}-{p}.csv")
             for slot, entry in enumerate(entries) for p, c, m, settings in cases
             if (p, c, entry["geometry_hash"]) not in existing]
    if not tasks:
        print("Using verified corrected assessment", flush=True)
        return
    write_json(OUT / "evaluation-freeze.json", freeze)
    if not (OUT / "source.zip").exists():
        snapshot_sources(OUT / "source.zip")
    with mp.get_context("spawn").Pool(8) as pool:
        for i, row in enumerate(pool.imap(evaluate_corrected, tasks, chunksize=2), 1):
            rows.append(row)
            if i % 90 == 0:
                print(f"Corrected assessment {i}/{len(tasks)}", flush=True)
    expected = len(entries)*len(cases)
    assert len(rows) == expected and not any(r["error"] for r in rows)
    assert len({(r["policy_id"], float(r["evaluation_speed_limit_m_s"]), r["track_hash"]) for r in rows}) == expected
    temporary = results.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    append_csv(temporary, rows, list(rows[0]))
    temporary.replace(results)
    write_json(OUT / "status.json", {"complete": True, "episodes": len(rows),
                                    "sha256": digest(results), "wall_s": time.perf_counter()-started})
    for p, cap, _, _ in cases:
        matched = [r for r in rows if r["policy_id"] == p and float(r["evaluation_speed_limit_m_s"]) == cap]
        print(p, cap, sum(str(r["completed"]) == "True" for r in matched), "/90",
              "RMSE mm", round(float(np.mean([float(r["geometry_lateral_rmse_m"]) for r in matched]))*1000, 2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
