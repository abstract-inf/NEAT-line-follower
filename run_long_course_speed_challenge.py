"""Build and evaluate the long course at a 1.50 m/s wheel-speed cap."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import load_model
from line_follower.episodes import Episode
from line_follower.experiments import append_csv, write_json
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.simulation import Simulation
from line_follower.tracks import Section, TrackMask, TrackSpec, sample_route, validate_track
from run_neat_speed_study import OUTPUT as STUDY, SEEDS, digest, read_json, verify_prepared

ROOT = Path(__file__).resolve().parent
PID_RUN = ROOT / "runs/pid-speed-followup-20261116"
OUTPUT = ROOT / "runs/long-course-speed-challenge"
CAP = 1.50


def make_track():
    sections = [
        Section("straight", length=5, feature="opening-straight"),
        Section("arc", radius=2, angle=math.pi/4, feature="left-curve"),
        Section("arc", radius=2, angle=-math.pi/4, feature="right-curve"),
        Section("corner", angle=math.pi/4, feature="45-degree-turn"),
        Section("straight", length=2, feature="two-meter-straight"),
        Section("corner", angle=math.pi/2, feature="90-degree-turn"),
        Section("straight", length=3, feature="middle-straight"),
    ]
    turns = (-1, 1, -1, 1)
    for index, turn in enumerate(turns):
        sections.append(Section("corner", angle=turn*math.pi/2, feature="zigzag"))
        if index < len(turns) - 1:
            sections.append(Section("straight", length=.8, feature="zigzag"))
    sections.append(Section("straight", length=5, feature="closing-straight"))
    sections = tuple(sections)
    route = sample_route(sections, (0, 0, 0))
    heading = route.heading_at(route.length)
    runout_end = route.points[-1] + .25 * np.array([math.cos(heading), math.sin(heading)])
    bounds = np.vstack((route.points, runout_end))
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    center = (lower + upper) / 2
    return TrackSpec("long-course-speed-challenge", sections, (-center[0], -center[1], 0.0),
                     arena_size=math.ceil(max(upper - lower) + 1), resolution=.002,
                     generator_version="2.0")


def draw_course(track, destination):
    fig, axis = plt.subplots(figsize=(8.2, 6.6), layout="constrained")
    route = track.route
    axis.plot(*route.points.T, color="#1F4B6E", linewidth=2)
    axis.scatter(*route.points[0], color="#008663", zorder=3)
    axis.scatter(*route.points[-1], color="#C45322", marker="s", zorder=3)
    landmarks = [
        (0, "Start", (8, -12)),
        (2.5, "5 m opening straight", (0, -19)),
        (5 + math.pi/2, "Opposing 2 m-radius curves", (0, -34)),
        (5 + math.pi, "45°", (-12, 12)),
        (6 + math.pi, "2 m straight", (12, -10)),
        (7 + math.pi, "90°", (10, 1)),
        (8.5 + math.pi, "3 m straight", (-12, -17)),
        (11.5 + math.pi, "Four alternating 90° turns", (-16, 8)),
        (route.length - 2.5, "5 m final straight", (-19, 8)),
        (route.length, "Finish", (10, -3)),
    ]
    for distance, label, offset in landmarks:
        xy = route.point_at(distance)
        alignment = "right" if offset[0] < 0 else "center" if offset[0] == 0 else "left"
        axis.annotate(label, xy, xytext=offset, textcoords="offset points", ha=alignment,
                      color="#243645", fontsize=9)
    axis.set(title=f"Long-course speed challenge  ·  {route.length:.2f} m", xlabel="X (m)", ylabel="Y (m)",
             aspect="equal")
    axis.margins(.15)
    axis.grid(color="#E4EAF0")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("svg", "png"):
        fig.savefig(destination / f"course-map.{suffix}", dpi=220)
    plt.close(fig)


def evaluate(sim, mask, policy, rules):
    episode = Episode(sim, mask.track, policy, rules, mask)
    peak = 0.0
    opening_time = None
    while not episode.done:
        speed_before = episode.speed_sum
        episode.advance()
        peak = max(peak, episode.speed_sum - speed_before)
        if opening_time is None and episode.route.distance >= 5:
            opening_time = episode.steps * sim.config.dt
    result = episode.result()
    if result.error:
        raise RuntimeError(result.error)
    return {"finished": result.completed, "termination": result.termination,
            "progress_m": result.progress_m, "progress_percent": 100 * result.progress,
            "average_forward_m_s": result.average_forward_speed_m_s, "peak_forward_m_s": peak,
            "opening_5m_s": opening_time, "elapsed_s": result.elapsed_s, "finish_time_s": result.completion_time_s}


def run(output):
    prepared = STUDY / "prepared"
    plan = verify_prepared(prepared)
    frozen = STUDY / "assessment/frozen"
    records = read_json(frozen / "selection-freeze.json")["policies"]
    pid_freeze = read_json(PID_RUN / "selection-freeze.json")["selections"]["1.50"]
    pid_file = PID_RUN / pid_freeze["config"]
    configs = {"Tuned PID": (pid_file, pid_freeze["sha256"])}
    for seed in SEEDS:
        record = records[f"seed-{seed:02d}-ramp"]
        configs[f"Ramp NEAT seed {seed}"] = frozen / record["file"], record["sha256"]
    if any(digest(path) != sha for path, sha in configs.values()):
        raise ValueError("A frozen policy fingerprint changed")
    robot = RobotConfig(**{**plan["robot"], "max_speed": CAP})
    rules = EpisodeConfig(**plan["episode"])
    track = make_track()
    errors = validate_track(track)
    if errors:
        raise ValueError(", ".join(errors))
    mask = TrackMask(track)
    output.mkdir(parents=True)
    track.save(output / "track.json")
    draw_course(track, output)
    write_json(output / "protocol.json", {
        "purpose": "Exploratory comparison on one custom course",
        "robot": asdict(robot), "episode": asdict(rules), "route_length_m": track.route.length,
        "track_geometry_sha256": track.geometry_hash, "meters_per_pixel": mask.meters_per_pixel,
        "policies": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha}
                     for name, (path, sha) in configs.items()},
    })
    sim = Simulation(robot)
    for name, (path, _) in configs.items():
        policy = (ResearchPIDController(ResearchPIDConfig.load(path), robot) if path.suffix == ".json"
                  else load_model(path)[0])
        row = {"policy": name, **evaluate(sim, mask, policy, rules)}
        append_csv(output / "results.csv", [row], list(row))
        print(row, flush=True)


def run_lower_pid_caps(course):
    destination = course / "pid-lower-caps"
    if destination.exists():
        raise FileExistsError(f"Lower-cap PID results already exist: {destination}")
    prepared = STUDY / "prepared"
    study = verify_prepared(prepared)
    protocol = read_json(course / "protocol.json")
    if protocol["episode"] != study["episode"] or protocol["robot"] != {**study["robot"], "max_speed": CAP}:
        raise ValueError("The saved course used different episode or robot settings")
    original_csv = course / "results.csv"
    original_hash = digest(original_csv)
    track = TrackSpec.load(course / "track.json")
    if track.geometry_hash != protocol["track_geometry_sha256"]:
        raise ValueError("The saved course geometry changed")
    mask = TrackMask(track)
    if mask.meters_per_pixel != protocol["meters_per_pixel"]:
        raise ValueError("The saved floor resolution changed")

    one_meter = read_json(PID_RUN / "selection-freeze.json")["selections"]["1.00"]
    configs = {
        .50: (prepared / "inputs/pid-050.json", study["inputs"]["pid-050.json"]),
        1.00: (PID_RUN / one_meter["config"], one_meter["sha256"]),
    }
    rules = EpisodeConfig(**protocol["episode"])
    rows = []
    for cap, (path, sha) in configs.items():
        if digest(path) != sha:
            raise ValueError(f"The frozen PID gains changed at {cap:.2f} m/s")
        robot = RobotConfig(**{**protocol["robot"], "max_speed": cap})
        controller = ResearchPIDController(ResearchPIDConfig.load(path), robot)
        row = {"policy": "Tuned PID", "wheel_speed_cap_m_s": cap,
               **evaluate(Simulation(robot), mask, controller, rules)}
        rows.append(row)
        print(row, flush=True)
    if digest(original_csv) != original_hash:
        raise ValueError("The original course results changed during this evaluation")
    destination.mkdir()
    write_json(destination / "protocol.json", {
        "course_protocol_sha256": digest(course / "protocol.json"),
        "original_results_sha256": original_hash,
        "track_geometry_sha256": track.geometry_hash,
        "policies": {f"{cap:.2f}": {"path": str(path.relative_to(ROOT)), "sha256": sha}
                     for cap, (path, sha) in configs.items()},
    })
    append_csv(destination / "results.csv", rows, list(rows[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--lower-pid-caps", action="store_true",
                        help="Evaluate lower-cap PID on the existing course in --output")
    args = parser.parse_args()
    if args.lower_pid_caps:
        run_lower_pid_caps(args.output.resolve())
    else:
        run(args.output.resolve())
