"""MuJoCo viewing and rendering; imported only by visual commands."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import numpy as np
from PIL import Image

from .config import EpisodeConfig, RobotConfig
from .controllers import Controller, PIDConfig, PIDController, load_model
from .episodes import Episode
from .simulation import Simulation
from .tracks import TrackSpec, straight_track
from .population_view import POPULATION_DISPLAY_HZ, STALE_EXPORT_SECONDS, PopulationScene, read_snapshot

SENSOR_ACTIVE_RGBA = np.array([0.0, 0.35, 1.0, 1.0], dtype=np.float32)
SENSOR_INACTIVE_RGBA = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)


def _line(scene, start: np.ndarray, end: np.ndarray, rgba: list[float], width: float = 0.001) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    scene.ngeom += 1


def add_overlays(scene, simulation: Simulation, reference: bool = False,
                 trajectory: list[np.ndarray] | None = None, landmarks: bool = True) -> None:
    """Graphics-only markers never touch the floor mask or physical contacts."""
    for hit, reading in zip(simulation.sensor_hits, simulation.observation):
        if not np.all(np.isfinite(hit)) or scene.ngeom >= scene.maxgeom:
            continue
        color = SENSOR_ACTIVE_RGBA if reading else SENSOR_INACTIVE_RGBA
        point = hit + [0, 0, 0.002]
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.002, 0, 0]), point, np.eye(3).ravel(), color)
        scene.ngeom += 1
    if reference:
        route = simulation.track.route.points
        points = route[::max(1, len(route) // 200)]
        for first, second in zip(points[:-1], points[1:]):
            _line(scene, np.r_[first, 0.001], np.r_[second, 0.001], [0.2, 0.75, 0.3, 0.7], 0.0007)
    if trajectory and len(trajectory) > 1:
        points = trajectory[::max(1, len(trajectory) // 180)]
        for first, second in zip(points[:-1], points[1:]):
            _line(scene, np.r_[first[:2], 0.002], np.r_[second[:2], 0.002], [0.9, 0.4, 0.05, 1])
    if not landmarks:
        return
    half = simulation.track.arena_size / 2
    corners = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    for first, second in zip(corners[:-1], corners[1:]):
        _line(scene, np.r_[first, 0.001], np.r_[second, 0.001], [0.6, 0.6, 0.6, 1], 0.002)
    # Start and finish posts sit beside the paint and are not observations.
    for arclength, color in ((0.0, [0.2, 0.7, 0.3, 1]), (simulation.track.route.length, [0.8, 0.2, 0.2, 1])):
        heading = simulation.track.route.heading_at(arclength)
        center = simulation.track.route.point_at(arclength)
        normal = np.array([-math.sin(heading), math.cos(heading)])
        for sign in (-1, 1):
            base = np.r_[center + normal * sign * 0.06, 0.001]
            _line(scene, base, base + [0, 0, 0.025], color, 0.003)


def set_transparency(simulation: Simulation, transparent: bool) -> None:
    simulation.model.geom("chassis_geom").rgba[3] = 0.25 if transparent else 1.0
    if simulation.config.design == "paper":
        simulation.model.geom("sensor_board").rgba[3] = 0.25 if transparent else 0.7


def camera_for(simulation: Simulation, view: str = "robot") -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if view == "track":
        camera.lookat[:] = [0, 0, 0]
        camera.distance = simulation.track.arena_size * 1.5
        camera.azimuth, camera.elevation = 90, -89.9
    else:
        competition = simulation.config.design == "competition"
        center_x = (simulation.config.sensor_offset + 0.008 - simulation.config.chassis_length/3) / 2 if competition else -0.015
        camera.lookat[:] = simulation.state.position + np.array([center_x, 0, -0.015])
        span = max(simulation.config.sensor_offset + 0.008 + simulation.config.chassis_length/3,
                   simulation.config.wheel_spacing + simulation.config.wheel_width,
                   2 * (simulation.config.sensor_radius + 0.008) * math.sin(simulation.config.sensor_angle / 2))
        camera.distance = ((1.85 if view == "robot" else 1.65) * span) if competition else (0.37 if view == "robot" else 0.32)
        camera.azimuth = simulation.state.yaw * 180/math.pi + {"robot": 135, "front": 180, "top": 90}[view]
        camera.elevation = {"robot": -32, "front": -8, "top": -89.9}[view]
    return camera


def render_track(track: TrackSpec, path: str | Path, robot: RobotConfig | None = None,
                 view: str = "robot", transparent: bool = False) -> Path:
    """Produce a real MuJoCo render, not a drawing or generated mockup."""
    simulation = Simulation(robot)
    simulation.reset(track)
    simulation.enable_rendering()
    set_transparency(simulation, transparent)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(simulation.model, height=900, width=1200) as renderer:
        renderer.update_scene(simulation.data, camera_for(simulation, view))
        if view == "track":
            add_overlays(renderer.scene, simulation)
        Image.fromarray(renderer.render()).save(destination)
    return destination


def view_tracks(tracks: list[TrackSpec], robot: RobotConfig | None = None,
                controller: Controller | None = None, episode_config: EpisodeConfig | None = None,
                auto_close: float | None = None, pid_config: PIDConfig | None = None) -> None:
    """Interactive reset/switch/pause with the exact headless episode loop."""
    import mujoco.viewer

    if not tracks:
        raise ValueError("Viewer needs at least one track")
    robot = robot or RobotConfig()
    simulation = Simulation(robot)
    simulation.reset(tracks[0])
    simulation.enable_rendering()
    pid = PIDController(pid_config or PIDConfig(sensor_layout=robot.sensor_layout))
    controllers = [pid] if controller is None else [pid, controller]
    controller_index = len(controllers) - 1
    track_index = 0
    episode = Episode(simulation, tracks[0], controllers[controller_index], episode_config)
    keys: SimpleQueue[int] = SimpleQueue()
    trajectory: deque[np.ndarray] = deque(maxlen=1200)
    paused, reference, transparent = False, False, False
    print("MuJoCo controls: SPACE pause | R reset | N/P track | C PID/model | T reference | B transparent chassis", flush=True)
    print(f"Track: {tracks[0].name}", flush=True)
    started = time.perf_counter()
    with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put) as viewer:
        camera = camera_for(simulation, "track")
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        while viewer.is_running():
            tick = time.perf_counter()
            if auto_close is not None and tick - started >= auto_close:
                break
            reset = False
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key in (ord("N"), ord("P")):
                    track_index = (track_index + (1 if key == ord("N") else -1)) % len(tracks)
                    reset = True
                elif key == ord("R"):
                    reset = True
                elif key == ord("C"):
                    controller_index = (controller_index + 1) % len(controllers)
                    reset = True
                elif key == ord("T"):
                    reference = not reference
                elif key == ord("B"):
                    transparent = not transparent
                    set_transparency(simulation, transparent)
            if reset:
                episode = Episode(simulation, tracks[track_index], controllers[controller_index], episode_config)
                simulation.update_visual_track(viewer)
                trajectory.clear()
                paused = False
                print(f"Track: {tracks[track_index].name}; controller={type(controllers[controller_index]).__name__}", flush=True)
            if not paused and not episode.done:
                episode.advance()
                trajectory.append(simulation.state.position.copy())
                if episode.done:
                    print(asdict(episode.result()), flush=True)
                    paused = True
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_overlays(viewer.user_scn, simulation, reference, list(trajectory))
            viewer.sync()
            time.sleep(max(0, robot.dt - (time.perf_counter() - tick)))


def export_studio_model(track: TrackSpec, path: str | Path, robot: RobotConfig | None = None) -> Path:
    """Save a standalone model with its track texture for MuJoCo Live."""
    destination = Path(path)
    if destination.suffix.lower() != ".mjb":
        raise ValueError("MuJoCo Studio exports require an .mjb file")
    simulation = Simulation(robot)
    simulation.reset(track)
    simulation.enable_rendering()
    simulation.model.qpos0[:] = simulation.data.qpos
    mujoco.mj_setConst(simulation.model, simulation.data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(simulation.model, str(destination))
    return destination


def _training_preview(output: str, stop, robot: RobotConfig | None = None,
                      auto_close: float | None = None, replay_speed: int = 10, grid: bool = True) -> None:
    """Observe live population snapshots, with replay fallback for older runs.

    Workers never wait for the window and closing it does not stop evolution.
    Display meshes are instances of independent worlds, without added contacts.
    """
    import mujoco.viewer

    output = Path(output)
    record_path = output / "preview.json"
    run_path = output / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    robot = robot or RobotConfig(**run.get("robot", {}))
    simulation = Simulation(robot)
    tracks = [straight_track()]
    simulation.reset(tracks[0])
    simulation.enable_rendering()
    settings = EpisodeConfig(**run.get("episode", {}))
    episode = None
    controller = None
    keys = SimpleQueue()
    index, paused, overview, reference = 0, False, False, False
    camera_request = "robot"
    revision = None
    record = {}
    next_poll = 0.0
    started = time.perf_counter()
    progress = {}
    population = None
    population_scene = None
    snapshots = {}
    initial_pose = simulation.data.qpos.copy()
    live_revisions = {}
    selected = 0
    follow = False
    auto_track, auto_select = True, True
    newest_export = None
    progress_updated = None
    run_status = {}
    previous_position = simulation.state.position.copy()
    if replay_speed < 1:
        raise ValueError("Replay speed must be a positive integer")
    print("Opening population observer; waiting for live evaluation snapshots (older runs use policy replay).", flush=True)
    print("Population observer: A automatic live view | F follow + orbit/zoom | U free camera | C overview | [/] select robot | N track | SPACE pause display", flush=True)
    with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        print("Training preview window is open.", flush=True)
        while viewer.is_running() and not stop.is_set():
            tick = time.perf_counter()
            if auto_close is not None and tick-started >= auto_close:
                break
            reset = False
            if tick >= next_poll:
                next_poll = tick + 1 / POPULATION_DISPLAY_HZ
                progress_path = output / "progress.json"
                progress_update = read_snapshot(progress_path)
                if progress_update is not None:
                    progress = progress_update
                    try:
                        progress_updated = progress_path.stat().st_mtime
                    except FileNotFoundError:
                        pass
                run_status = read_snapshot(output / "status.json") or {}
                roster_path = output / "live" / "population.json"
                roster = read_snapshot(roster_path)
                if roster is not None and not paused:
                    changed_population = population is None or roster["token"] != population["token"]
                    if changed_population:
                        population = roster
                        snapshots, live_revisions = {}, {}
                        newest_export = None
                        tracks = [TrackSpec.load(path) for path in population["tracks"]]
                        index, selected = 0, 0
                        simulation.reset(tracks[0])
                        initial_pose = simulation.data.qpos.copy()
                        simulation.update_visual_track(viewer)
                        population_scene = PopulationScene(simulation)
                        # Hide the base robot; all individuals are real mesh
                        # instances in user_scn, including the selected genome.
                        simulation.model.geom_group[simulation.model.geom_bodyid != 0] = 5
                        viewer.opt.geomgroup[5] = 0
                        population_scene.option.geomgroup[5] = 1
                        camera_request = "track"
                        overview, follow = True, False
                    dirty = changed_population
                    for path in (output / "live").glob("*.json"):
                        if path.name == "population.json":
                            continue
                        try:
                            stamp = path.stat().st_mtime_ns
                        except FileNotFoundError:
                            continue
                        if live_revisions.get(path.name) == stamp:
                            continue
                        update = read_snapshot(path)
                        if update is None:
                            continue
                        live_revisions[path.name] = stamp
                        if update.get("token") == population["token"]:
                            newest_export = max(newest_export or 0, stamp/1e9)
                            for item in update["robots"]:
                                snapshots[item["genome"], item["track"]] = item
                            dirty = True
                    # A finished batch retains its terminal poses. Default to
                    # the track with the most live worlds rather than making
                    # the observer look frozen on the generation's first track.
                    running = {}
                    for (genome, slot), item in snapshots.items():
                        if item["status"] == "running":
                            running.setdefault(slot, []).append(genome)
                    if auto_track and running:
                        active = max(running, key=lambda slot: (len(running[slot]), slot == index))
                        if active != index:
                            index = active
                            simulation.reset(tracks[index])
                            initial_pose = simulation.data.qpos.copy()
                            simulation.update_visual_track(viewer)
                            dirty = True
                    if auto_select and index in running:
                        selected_genome = population["genomes"][selected]
                        if selected_genome not in running[index]:
                            selected = population["genomes"].index(running[index][0])
                            dirty = True
                    if dirty:
                        with viewer.lock():
                            poses = []
                            for genome in population["genomes"]:
                                item = snapshots.get((genome, index))
                                poses.append((item["qpos"] if item else initial_pose,
                                              item["status"] if item else "waiting"))
                            population_scene.draw(viewer.user_scn, poses, grid, selected)
                        selected_item = snapshots.get((population["genomes"][selected], index))
                        if selected_item:
                            simulation.data.qpos[:] = selected_item["qpos"]
                            mujoco.mj_kinematics(simulation.model, simulation.data)
                changed = record_path.stat().st_mtime_ns if record_path.exists() and population is None else None
                if changed is not None and changed != revision:
                    record = json.loads(record_path.read_text())
                    controller, metadata = load_model(output / record["model"])
                    replay_robot = RobotConfig(**metadata["robot"])
                    scheduled_speeds = {speed for _, speed in run.get("settings", {}).get("speed_schedule", [])}
                    if (replay_robot != robot and
                            (replay_robot.max_speed not in scheduled_speeds
                             or replace(replay_robot, max_speed=robot.max_speed) != robot)):
                        raise ValueError("Preview robot differs from run metadata; restart the observer for this run")
                    if replay_robot != simulation.config:
                        simulation.config = replay_robot
                        angular_limit = replay_robot.max_speed / replay_robot.wheel_radius
                        simulation.model.actuator_ctrlrange[:] = (-angular_limit, angular_limit)
                    settings = EpisodeConfig(**metadata["episode"])
                    tracks = [TrackSpec.load(output / path) for path in record["tracks"]]
                    index, reset, revision = 0, True, changed
                    print(f"Preview generation {record['generation']}", flush=True)
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key == ord("R"):
                    reset = True
                elif key == ord("N"):
                    index, reset = (index+1) % len(tracks), True
                    auto_track = False
                elif key == ord("A"):
                    auto_track, auto_select = True, True
                    live_revisions.clear()
                elif key == ord("C"):
                    overview = not overview
                    camera_request = "track" if overview else "robot"
                    follow = not overview
                elif key == ord("F"):
                    overview = False
                    camera_request = "robot"
                    follow = True
                elif key == ord("U"):
                    follow = False
                elif key == ord("G") and population:
                    grid = not grid
                    live_revisions.clear()
                    camera_request = "track"
                    overview, follow = True, False
                elif key in (ord("["), ord("]")) and population:
                    selected = (selected + (1 if key == ord("]") else -1)) % len(population["genomes"])
                    reset = True
                    auto_select = False
                elif key == ord("T"):
                    reference = not reference
            if population and reset:
                simulation.reset(tracks[index])
                initial_pose = simulation.data.qpos.copy()
                simulation.update_visual_track(viewer)
                # Invalidate file stamps so the selected track/robot redraws.
                live_revisions.clear()
            if population is None and controller is not None and (reset or episode is None or episode.done):
                episode = Episode(simulation, tracks[index], controller, settings)
                simulation.update_visual_track(viewer)
            if population is None and episode is not None and not paused:
                for _ in range(replay_speed):
                    if episode.done:
                        break
                    episode.advance()
            with viewer.lock():
                position = simulation.state.position.copy()
                offset = population_scene.offset(selected, len(population["genomes"]), grid) if population else np.zeros(3)
                position += offset
                if camera_request is not None:
                    camera = camera_for(simulation, camera_request)
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.trackbodyid = viewer.cam.fixedcamid = -1
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
                    if population and grid:
                        if camera_request == "track":
                            viewer.cam.distance *= math.ceil(math.sqrt(len(population["genomes"])))
                        else:
                            viewer.cam.lookat[:] += offset
                    camera_request = None
                elif follow:
                    # Translation only: preserve mouse orbit, zoom and pan offset.
                    viewer.cam.lookat[:] += position - previous_position
                previous_position = position
                if population is None:
                    viewer.user_scn.ngeom = 0
                    add_overlays(viewer.user_scn, simulation, reference, landmarks=overview)
            title = (f"{record.get('phase', 'Champion')} | generation {record['generation']}"
                     if controller is not None else "Waiting for first policy | stationary preview")
            detail = "Free camera | F: frame robot | C: overview/robot | policy replay, independent of workers."
            if population:
                exports_age = time.time()-max(newest_export or 0, progress_updated or 0)
                stale = newest_export is not None and exports_age > STALE_EXPORT_SECONDS
                statuses = [snapshots.get((key, index), {}).get("status", "waiting") for key in population["genomes"]]
                title = f"Live population: {len(statuses)} | generation {population['generation']} | {population['backend']}"
                if "wheel_speed_cap_m_s" in population:
                    title += f" | cap {population['wheel_speed_cap_m_s']:.2f} m/s"
                detail = (f"Track {index+1}/{len(tracks)} | running {statuses.count('running')} | waiting {statuses.count('waiting')} | "
                          f"finished {len(statuses)-statuses.count('running')-statuses.count('waiting')}\n"
                          f"Genome {population['genomes'][selected]} | {'auto live' if auto_track and auto_select else 'manual selection'} | "
                          "A auto | F follow | U free | G grid | [/] select | N track")
                if paused:
                    detail += "\nDisplay PAUSED: SPACE to resume (training continues)."
                elif run_status.get("complete"):
                    title = f"Training complete | {len(statuses)} saved robot poses"
                elif stale:
                    title = f"Saved population: {len(statuses)} | waiting for training exports"
                    detail += f"\nNo fresh exports for {exports_age:.0f}s. Check the training terminal; this window only observes."
                elif statuses.count("running") == 0:
                    phase = progress.get("phase", "")
                    detail += ("\nTraining poses held during champion validation." if phase == "validation" else
                               "\nNo active robots on this track; showing latest poses. A: auto live view.")
            else:
                detail += f" Replay target {replay_speed}x; training is unpaced."
            if progress:
                detail += (f"\n{progress['phase']}: {progress['completed_episodes']}/{progress['total_episodes']} episodes")
                if progress.get("batch_wall_seconds", 0) > 0:
                    wall = progress["batch_wall_seconds"]
                    detail += (f" | world speed {progress['simulated_seconds']/wall:.2f}x"
                               f" | aggregate {progress['aggregate_simulated_seconds']/wall:.1f} world-s/s")
            viewer.set_texts((None, None, title, detail))
            viewer.sync()
            stop.wait(max(0, robot.dt-(time.perf_counter()-tick)))


@contextmanager
def training_preview(output: str | Path, enabled: bool, robot: RobotConfig | None = None,
                     replay_speed: int = 10, grid: bool = True):
    """Keep rendering out of the persistent physics workers."""
    process = None
    if enabled:
        context = mp.get_context("spawn")
        stop = context.Event()
        process = context.Process(target=_training_preview, args=(str(Path(output).resolve()), stop, robot),
                                  kwargs={"replay_speed": replay_speed, "grid": grid}, name="MuJoCo-preview")
        process.start()
    try:
        yield process
    finally:
        if process is not None:
            stop.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
