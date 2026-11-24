"""Measure missing MAE by frozen-policy CPU validation replay, without training.

These new measurements are labelled separately from historical GPU results.
No test tracks are loaded and no model is selected or updated here.
"""

from dataclasses import asdict
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model
from line_follower.episodes import run_episode
from line_follower.simulation import Simulation
from line_follower.suites import load_split
from research.analyze_neat import write_csv

WORKERS = 4


@lru_cache(maxsize=2)
def context(run_name: str):
    run = ROOT / "runs" / run_name
    record = json.loads((run / "run.json").read_text())
    controller, metadata = load_model(run / "best.pkl")
    robot = RobotConfig(**record["robot"])
    episode = EpisodeConfig(**record["episode"])
    tracks = load_split(record["suite_path"], "validation")
    return Simulation(robot), controller, episode, tracks, metadata, record


def replay(task: tuple[str, int]) -> dict:
    name, slot = task
    simulation, controller, episode, tracks, metadata, record = context(name)
    absolute_errors = []
    def observe(current, force):
        if current.steps:
            absolute_errors.append(abs(current.route.lateral_error))
    result = run_episode(simulation, tracks[slot], controller, episode,
                         time_limit=record["settings"].get("time_limit"), observer=observe)
    return {"run": name, "condition": "neat_frozen_posthoc_cpu_validation", "generation": metadata["generation"],
            "backend": "cpu", "metric_origin": "new fixed-policy replay; not historical GPU telemetry",
            "mean_absolute_lateral_error_m": sum(absolute_errors)/len(absolute_errors), **asdict(result)}


def main() -> None:
    tasks = [(name, slot) for name in ("28-10-2026", "28-10-2026-hard") for slot in range(30)]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        rows = list(pool.imap(replay, tasks))
    write_csv(ROOT / "research/artifacts/neat/posthoc-cpu-validation.csv", rows)
    for name in ("28-10-2026", "28-10-2026-hard"):
        selected = [r for r in rows if r["run"] == name]
        print(name, "CPU completion", sum(r["completed"] for r in selected), "/", len(selected),
              "MAE", sum(r["mean_absolute_lateral_error_m"] for r in selected)/len(selected))


if __name__ == "__main__":
    mp.freeze_support()
    main()
