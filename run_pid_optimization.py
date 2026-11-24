"""Run a bounded, parallel, headless PID study; never start NEAT training.

Edit the globals below, or select another fresh output with --output. The final
90-track assessment occurs only after validation selection and policy freezing.
"""

import argparse
import multiprocessing
from pathlib import Path

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.pid_experiments import PIDStudyConfig, run_study

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = "runs/pid-research-20261029"
TRACK_SUITE = "benchmarks/suite/suite.json"
ROBOT_CONFIG = "runs/28-10-2026-hard/robot.json"
NEAT_BASELINE = "runs/28-10-2026-hard/best.pkl"
TUNING_SEED = 20261029
WORKERS = 4
INITIAL_CANDIDATES = 96
LOCAL_REFINEMENTS = 24
PROMOTED_PER_FAMILY = 6
VALIDATION_PER_FAMILY = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / OUTPUT_DIRECTORY)
    args = parser.parse_args()
    settings = PIDStudyConfig(TUNING_SEED, WORKERS, INITIAL_CANDIDATES, LOCAL_REFINEMENTS,
                              PROMOTED_PER_FAMILY, VALIDATION_PER_FAMILY)
    robot = RobotConfig.load(PROJECT_ROOT / ROBOT_CONFIG)
    episode = EpisodeConfig(maximum_time_limit=120, time_reference_speed=.10)
    print(run_study(args.output.resolve(), PROJECT_ROOT / TRACK_SUITE, robot, episode,
                    PROJECT_ROOT / NEAT_BASELINE, settings))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
