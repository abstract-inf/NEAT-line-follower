"""A reproducible PID-only sensor-offset pilot. This module never evolves NEAT.

Edit PilotConfig defaults or pass a JSON configuration. All workers own their
MuJoCo state; cached simulations are reset for each deterministic trial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
from functools import lru_cache
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .config import EpisodeConfig, RobotConfig
from .controllers import PIDConfig, PIDController
from .episodes import Episode, TELEMETRY_FIELDS
from .experiments import append_csv, environment_record, write_json, snapshot_sources
from .simulation import Simulation, sensor_locations
from .tracks import Section, TrackMask, TrackSpec, validate_track


@dataclass(frozen=True)
class PilotConfig:
    """SI parameters for the engineering comparison, separate from main study."""

    offsets_m: tuple[float, ...] = (0.090, 0.120, 0.140, 0.160)
    target_speeds_m_s: tuple[float, ...] = (0.10, 0.18, 0.26)
    tuning_speed_m_s: float = 0.18
    proportional_gains: tuple[float, ...] = (0.06, 0.12, 0.20)
    derivative_gains: tuple[float, ...] = (0.001, 0.004, 0.010)
    integral_gain: float = 0.0001
    workers: int = 4

    def __post_init__(self) -> None:
        values = (*self.offsets_m, *self.target_speeds_m_s, self.tuning_speed_m_s,
                  *self.proportional_gains, *self.derivative_gains)
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Pilot distances, speeds, and gains must be finite and positive")
        if not all((self.offsets_m, self.target_speeds_m_s, self.proportional_gains, self.derivative_gains)):
            raise ValueError("Pilot parameter grids cannot be empty")
        if self.workers < 1 or not math.isfinite(self.integral_gain) or self.integral_gain < 0:
            raise ValueError("Need at least one worker and a nonnegative integral gain")


def pilot_tracks(stage: str) -> list[TrackSpec]:
    """Dedicated, disjoint courses; never reuse the research test manifests."""
    def straight(length, width=.020):
        return Section("straight", length=length, width=width)

    def corner(angle):
        return Section("corner", angle=angle)

    def arc(radius, angle):
        return Section("arc", radius=radius, angle=angle)

    quarter = math.pi/2
    if stage == "tuning":
        recipes = [
            ("straight", [straight(1.4)]),
            ("sharp", [straight(.55), corner(quarter), straight(.65)]),
            ("zigzag", [straight(.60), corner(-quarter), straight(.30), corner(quarter), straight(.60)]),
            ("curve", [straight(.50), arc(.25, quarter), straight(.50)]),
        ]
    elif stage == "evaluation":
        recipes = [
            ("straight", [straight(1.4, .012)]),
            ("straight", [straight(1.8, .020)]),
            ("straight", [straight(2.2, .028)]),
            ("sharp", [straight(.70), corner(quarter), straight(.60)]),
            ("sharp", [straight(.90), corner(-quarter), straight(.70)]),
            ("sharp", [straight(.65), corner(quarter), straight(.40), corner(quarter), straight(.55)]),
            ("zigzag", [straight(.55), corner(quarter), straight(.35), corner(-quarter), straight(.65)]),
            ("zigzag", [straight(.60), corner(-quarter/2), straight(.35), corner(quarter), straight(.35), corner(-quarter/2), straight(.50)]),
            ("zigzag", [straight(.55), corner(-quarter), straight(.35), corner(quarter), straight(.45), corner(-quarter), straight(.35), corner(quarter), straight(.60)]),
            ("curve", [straight(.60), arc(.30, quarter), straight(.50)]),
            ("curve", [straight(.60), arc(.18, -quarter), straight(.55)]),
            ("curve", [straight(.55), arc(.20, .9), arc(.20, -1.8), arc(.20, .9), straight(.60)]),
        ]
    else:
        raise ValueError("Pilot stage must be tuning or evaluation")
    tracks = [TrackSpec(f"{stage}-{index:02d}-{feature}", tuple(sections), (-1.1, 0, 0),
                        split=f"geometry_{stage}", difficulty=feature)
              for index, (feature, sections) in enumerate(recipes)]
    for track in tracks:
        errors = validate_track(track)
        if errors:
            raise ValueError(f"{track.name}: {'; '.join(errors)}")
    return tracks


@lru_cache(maxsize=4)
def _simulation(config_json: str) -> Simulation:
    return Simulation(RobotConfig(**json.loads(config_json)))


@lru_cache(maxsize=4)
def _mask(track_json: str) -> TrackMask:
    return TrackMask(TrackSpec.from_dict(json.loads(track_json)))


def _trial(task: dict) -> dict:
    simulation = _simulation(task["robot_json"])
    mask = _mask(task["track_json"])
    episode = Episode(simulation, mask.track, PIDController(PIDConfig(**task["pid"])), mask=mask)
    maximum_tilt = 0.0
    minimum_clearance = simulation.board_ground_clearance()
    # Coplanar rigid PCB pieces have the same lowest Z. Use vertices directly
    # instead of dozens of expensive geometry-distance calls at every frame.
    outline = simulation.model.mesh("integrated_pcb")
    geom = simulation.model.geom("chassis_geom")
    vertices = simulation.model.mesh_vert[outline.vertadr[0]:outline.vertadr[0]+outline.vertnum[0]]
    writer = None
    stream = None
    if task.get("telemetry_path"):
        path = Path(task["telemetry_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS + ["tilt_rad", "pcb_clearance_m"])
        writer.writeheader()
    try:
        while not episode.done:
            row = episode.advance(telemetry=writer is not None)
            rotation = simulation.data.xmat[simulation.chassis_id].reshape(3, 3)
            tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
            geom_rotation = simulation.data.geom_xmat[geom.id].reshape(3, 3)
            clearance = float(np.min(vertices @ geom_rotation[2]) + simulation.data.geom_xpos[geom.id, 2])
            maximum_tilt = max(maximum_tilt, tilt)
            minimum_clearance = min(minimum_clearance, clearance)
            if writer is not None and row is not None:
                writer.writerow({**row, "tilt_rad": tilt, "pcb_clearance_m": clearance})
    finally:
        if stream is not None:
            stream.close()
    result = asdict(episode.result())
    if result["error"]:
        raise RuntimeError(f"Technical pilot failure: {result}")
    passed = sum(episode.route.distance >= at + .050 for at in mask.track.route.corners)
    return {"stage": task["stage"], "candidate": task["candidate"], "offset_m": simulation.config.sensor_offset,
            "nominal_straight_target_m_s": task["speed"], "kp": task["pid"]["kp"],
            "ki": task["pid"]["ki"], "kd": task["pid"]["kd"],
            "maximum_tilt_rad": maximum_tilt, "minimum_pcb_clearance_m": minimum_clearance,
            "sharp_turns_total": len(mask.track.route.corners), "sharp_turns_completed": int(passed), **result}


def _rank(rows: list[dict]) -> tuple:
    return (float(np.mean([row["completed"] for row in rows])),
            float(np.mean([row["progress"] for row in rows])),
            -float(np.mean([row["lateral_rmse_m"] for row in rows])))


def _summary(rows: list[dict]) -> list[dict]:
    groups = sorted({(row["offset_m"], row["nominal_straight_target_m_s"], feature)
                     for row in rows for feature in ("all", row["difficulty"])})
    summaries = []
    for offset, speed, feature in groups:
        subset = [r for r in rows if r["offset_m"] == offset and r["nominal_straight_target_m_s"] == speed
                  and (feature == "all" or r["difficulty"] == feature)]
        turns = sum(r["sharp_turns_total"] for r in subset)
        summaries.append({"offset_m": offset, "nominal_straight_target_m_s": speed, "feature": feature,
            "episodes": len(subset), "completed": sum(r["completed"] for r in subset),
            "completion_fraction": _rank(subset)[0], "mean_progress": _rank(subset)[1],
            "mean_observed_lateral_rmse_m": -_rank(subset)[2],
            "mean_line_loss_events": float(np.mean([r["line_loss_events"] for r in subset])),
            "line_loss_fraction": sum(r["line_loss_seconds"] for r in subset)/sum(r["elapsed_s"] for r in subset),
            "mean_observed_forward_speed_m_s": float(np.mean([r["average_forward_speed_m_s"] for r in subset])),
            "maximum_tilt_rad": max(r["maximum_tilt_rad"] for r in subset),
            "minimum_pcb_clearance_m": min(r["minimum_pcb_clearance_m"] for r in subset),
            "sharp_turns_total": turns, "sharp_turns_completed": sum(r["sharp_turns_completed"] for r in subset),
            "technical_failures": sum(bool(r["error"]) for r in subset),
            "tipped_episodes": sum(r["termination"] == "tipped" for r in subset)})
    return summaries


def run_pilot(output: str | Path, config: PilotConfig | None = None) -> Path:
    """Tune one PID configuration, then evaluate the complete offset/speed grid."""
    config = config or PilotConfig()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Pilot output must be empty; preserve previous observations")
    robot = RobotConfig()
    if max((*config.target_speeds_m_s, config.tuning_speed_m_s)) > robot.max_speed:
        raise ValueError("Nominal target exceeds the unchanged physical wheel-speed ceiling")
    tracks = {stage: pilot_tracks(stage) for stage in ("tuning", "evaluation")}
    hashes = {stage: {t.geometry_hash for t in courses} for stage, courses in tracks.items()}
    if hashes["tuning"] & hashes["evaluation"]:
        raise ValueError("Tuning/evaluation pilot tracks overlap")
    for stage, courses in tracks.items():
        for track in courses:
            track.save(output / "tracks" / stage / f"{track.name}.json")
    robots = {offset: replace(robot, sensor_offset=offset) for offset in config.offsets_m}
    for offset, variant in robots.items():
        Simulation(variant)  # Fail geometry validation before any controller tuning.
        write_json(output / "robots" / f"offset-{offset:.3f}.json", asdict(variant))
    write_json(output / "protocol.json", {"pilot": asdict(config), "robot_base": asdict(robot),
               "episode": asdict(EpisodeConfig()), "environment": environment_record(),
               "scope": "Deterministic PID engineering pilot; not independent research trials; no NEAT training",
               "ranking": "completion fraction, valid progress, lower observed-duration RMSE",
               "sensor_measurements": [{"index": i, "x_m": float(p[0]), "y_m": float(p[1])}
                                       for i, p in enumerate(sensor_locations(robot))]})
    snapshot_sources(output / "source.zip")
    gains = [PIDConfig(kp=kp, ki=config.integral_gain, kd=kd)
             for kp, kd in itertools.product(config.proportional_gains, config.derivative_gains)]
    gains.append(PIDConfig())

    def tasks(stage, candidates, speeds):
        for candidate, pid in enumerate(candidates):
            for offset, speed, track in itertools.product(config.offsets_m, speeds, tracks[stage]):
                task = {"stage": stage, "candidate": candidate, "speed": speed,
                        "robot_json": json.dumps(asdict(robots[offset]), sort_keys=True),
                        "track_json": json.dumps(asdict(track), sort_keys=True),
                        "pid": asdict(replace(pid, base=speed/robot.max_speed))}
                if stage == "evaluation":
                    task["telemetry_path"] = str(output / "telemetry" / f"offset-{offset:.3f}-speed-{speed:.2f}-{track.name}.csv")
                yield task

    started = time.perf_counter()
    with mp.get_context("spawn").Pool(config.workers) as pool:
        tuning = []
        for row in pool.imap(_trial, tasks("tuning", gains, [config.tuning_speed_m_s])):
            tuning.append(row)
            append_csv(output / "tuning.csv", [row], list(row))
            if len(tuning) % (len(config.offsets_m)*len(tracks["tuning"])) == 0:
                print(f"PID pilot tuning: {len(tuning)} episodes saved", flush=True)
        best = max(range(len(gains)), key=lambda i: _rank([r for r in tuning if r["candidate"] == i]))
        chosen = gains[best]
        write_json(output / "pid-selected.json", asdict(replace(chosen, base=config.tuning_speed_m_s/robot.max_speed)))
        print(f"Shared PID selected: {asdict(chosen)}; starting separate evaluation courses", flush=True)
        evaluated = []
        for row in pool.imap(_trial, tasks("evaluation", [chosen], config.target_speeds_m_s)):
            evaluated.append(row)
            append_csv(output / "evaluation.csv", [row], list(row))
            if len(evaluated) % len(tracks["evaluation"]) == 0:
                print(f"PID pilot evaluation: {len(evaluated)} episodes saved", flush=True)
    summaries = _summary(evaluated)
    append_csv(output / "summary.csv", summaries, list(summaries[0]))
    recommended = max(config.offsets_m, key=lambda offset: _rank([r for r in evaluated if r["offset_m"] == offset]))
    write_json(output / "robot-recommended.json", asdict(robots[recommended]))
    sharp = [r for r in evaluated if r["offset_m"] == recommended and r["difficulty"] in {"sharp", "zigzag"}]
    write_json(output / "result.json", {"recommended_starting_offset_m": recommended,
               "geometry_finalized": False, "selected_shared_pid": asdict(chosen),
               "pid_base_semantics": "Base is overridden per trial by nominal straight target / 0.30 m/s",
               "recommended_configuration": "robot-recommended.json",
               "recommended_sharp_course_completions": sum(r["completed"] for r in sharp),
               "recommended_sharp_course_count": len(sharp),
               "caution": "User must select final geometry. Pilot findings do not establish NEAT performance or statistical significance.",
               "tuning_episodes": len(tuning), "evaluation_episodes": len(evaluated),
               "physics_steps": sum(r["physics_steps"] for r in tuning+evaluated),
               "wall_seconds": time.perf_counter()-started})
    return output / "summary.csv"


def report_pilot(directory: str | Path) -> Path:
    """Regenerate figures and a factual report from saved CSVs; never rerun trials."""
    directory = Path(directory)
    with (directory / "evaluation.csv").open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError("Pilot has no evaluation episodes")
    rows = [{**r, **{key: float(r[key]) for key in ("offset_m", "nominal_straight_target_m_s", "progress",
             "lateral_rmse_m", "maximum_tilt_rad", "minimum_pcb_clearance_m", "elapsed_s", "line_loss_seconds")},
             "completed": r["completed"] == "True", "line_loss_events": int(r["line_loss_events"])} for r in raw]
    offsets = sorted({r["offset_m"] for r in rows})
    speeds = sorted({r["nominal_straight_target_m_s"] for r in rows})
    result = json.loads((directory / "result.json").read_text())
    gain = result["selected_shared_pid"]
    lines = ["# Sensor-offset engineering pilot", "",
             f"One shared PID: Kp={gain['kp']}, Ki={gain['ki']}, Kd={gain['kd']}. "
             "Gains selected on separate pilot tuning tracks; base = nominal speed / 0.30 m/s.", "",
             "| Center offset (m) | Outer-sensor X (m) | Completed / attempted | Mean observed RMSE (m) | Line-loss events | Min PCB clearance (m) | Max tilt (deg) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for offset in offsets:
        subset = [r for r in rows if r["offset_m"] == offset]
        outer_x = sensor_locations(replace(RobotConfig(), sensor_offset=offset))[0, 0]
        lines.append(f"| {offset:.3f} | {outer_x:.6f} | {sum(r['completed'] for r in subset)} / {len(subset)} | "
                     f"{np.mean([r['lateral_rmse_m'] for r in subset]):.6f} | {sum(r['line_loss_events'] for r in subset)} | "
                     f"{min(r['minimum_pcb_clearance_m'] for r in subset):.6f} | {max(r['maximum_tilt_rad'] for r in subset)*180/math.pi:.3f} |")
    lines += ["", "| Center offset (m) | " + " | ".join(f"{speed:.2f} m/s target" for speed in speeds) + " |",
              "| ---: | " + " | ".join("---:" for _ in speeds) + " |"]
    for offset in offsets:
        counts = []
        for speed in speeds:
            subset = [r for r in rows if r["offset_m"] == offset and r["nominal_straight_target_m_s"] == speed]
            counts.append(f"{sum(r['completed'] for r in subset)} / {len(subset)}")
        lines.append(f"| {offset:.3f} | " + " | ".join(counts) + " |")
    recommended = result["recommended_starting_offset_m"]
    lines += ["", f"The prespecified ranking recommends **{recommended:.3f} m as a starting point**. "
              "Geometry remains unfinalized. The best offset depends on speed and controller; the pooled ranking is not a universal optimum.",
              "", "Errors and tilt/clearance describe the observed duration, including unsuccessful episodes. "
              "A zero all-zero-sensor count does not establish correct route following: the axle may cut a corner while the forward array still sees paint. "
              "All failures remain in evaluation.csv; summary.csv separates feature types and speeds.",
              "", "This is a deterministic engineering pilot, not repeated independent research evidence. "
              "It provides no NEAT performance, statistical-significance, hardware-calibration, or robustness claim.",
              "", "Reproduction inputs: protocol.json, source.zip, robots/, tracks/, pid-selected.json. "
              "Raw outcomes: tuning.csv, evaluation.csv, summary.csv, telemetry/. No main study test tracks were used."]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        lines += ["", "Install the plots extra and run geometry-report again to generate figures."]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
        for offset in offsets:
            subsets = [[r for r in rows if r["offset_m"] == offset and r["nominal_straight_target_m_s"] == s] for s in speeds]
            label = f"{offset*100:.0f} cm"
            axes[0,0].plot(speeds, [np.mean([r["completed"] for r in subset]) for subset in subsets], "o-", label=label)
            axes[0,1].plot(speeds, [np.mean([r["lateral_rmse_m"] for r in subset])*1000 for subset in subsets], "o-", label=label)
            axes[1,0].plot(speeds, [max(r["maximum_tilt_rad"] for r in subset)*180/math.pi for subset in subsets], "o-", label=label)
        for axis in axes.flat:
            axis.grid(alpha=.2)
        for axis in (axes[0,0],axes[0,1],axes[1,0]):
            axis.set_xlabel("Nominal straight target (m/s)")
        axes[0,0].set(ylabel="Completion fraction (12 courses)", ylim=(-.03,1.03))
        axes[0,0].legend(title="Center offset", frameon=False, fontsize=8)
        axes[0,1].set_ylabel("Mean observed lateral RMSE (mm)")
        axes[1,0].set_ylabel("Maximum observed tilt (degrees)")
        course_path = directory / "tracks/evaluation/evaluation-03-sharp.json"
        if course_path.exists():
            track = TrackSpec.load(course_path)
            axis = axes[1,1]
            axis.plot(*track.route.points.T, color=".3", linewidth=4, alpha=.3, label="Reference route")
            for offset in (.09,.14):
                trace = directory / "telemetry" / f"offset-{offset:.3f}-speed-0.18-evaluation-03-sharp.csv"
                if not trace.exists():
                    continue
                with trace.open(newline="") as stream:
                    points = np.array([[float(r["x_m"]),float(r["y_m"])] for r in csv.DictReader(stream)])
                axis.plot(*points.T, linewidth=1.5, label=f"{offset*100:.0f} cm axle path")
            vertex = track.route.point_at(track.route.corners[0])
            axis.set(xlim=(vertex[0]-.25,vertex[0]+.08), ylim=(vertex[1]-.08,vertex[1]+.38),
                     xlabel="World X (m)", ylabel="World Y (m)", title="90° corner, 0.18 m/s target")
            axis.set_aspect("equal")
            axis.legend(frameon=False, fontsize=7)
        fig.suptitle("PID sensor-offset pilot: 12 evaluation courses per speed and offset", fontsize=11)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(directory / f"offset-comparison.{suffix}", dpi=300)
        plt.close(fig)
        lines += ["", "![Offset comparison](offset-comparison.png)",
                  "", "Points are deterministic summaries over the same 12 courses. No confidence intervals are implied."]
    destination = directory / "report.md"
    destination.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return destination
