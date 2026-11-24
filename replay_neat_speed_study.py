"""Open a saved study model in MuJoCo with its recorded episode rules."""

import argparse

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model
from line_follower.tracks import DIFFICULTIES, TrackSpec, generate_track
from line_follower.viewer import view_tracks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the study stage's best.pkl")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--track", help="Path to a saved track JSON")
    source.add_argument("--seed", type=int, help="Seed for a fresh practice track")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="advanced")
    args = parser.parse_args(argv)

    controller, metadata = load_model(args.model)
    if "episode" not in metadata or "robot" not in metadata:
        parser.error("The model must contain saved episode and robot settings")
    episode = EpisodeConfig(**metadata["episode"])
    robot = RobotConfig(**metadata["robot"])
    track = (TrackSpec.load(args.track) if args.track
             else generate_track(args.seed, args.difficulty))
    view_tracks([track], robot, controller, episode_config=episode)


if __name__ == "__main__":
    main()
