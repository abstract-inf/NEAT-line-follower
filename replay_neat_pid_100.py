"""Replay the two frozen comparison policies on ten prespecified tracks."""

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path
from queue import SimpleQueue
import time

import numpy as np

from compare_neat_pid_100 import OUTPUT, replay_setup
from line_follower.controllers import NEATController
from line_follower.episodes import Episode
from line_follower.pid import ResearchPIDController
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask
from line_follower.viewer import add_overlays, camera_for, set_transparency


def upload_display(simulation, track, viewer, arena, resolution):
    """The common visual canvas never replaces the original physics sensor mask."""
    display = TrackMask(replace(track, arena_size=arena, resolution=resolution))
    texture = simulation.model.texture("track_texture")
    if texture.width[0] != display.pixels or texture.height[0] != display.pixels:
        raise ValueError("Unexpected replay canvas size")
    simulation.model.tex_data[:] = np.asarray(display.image.convert("RGB")).reshape(-1)
    viewer.update_texture(texture.id)


def view(output):
    import mujoco.viewer
    tracks, robot, config, controllers = replay_setup(output)
    if type(controllers[0]) is not NEATController or type(controllers[1]) is not ResearchPIDController:
        raise ValueError("Frozen replay controller roles are inconsistent")
    manifest = json.loads((output/"frozen/manifest.json").read_text(encoding="utf-8"))
    identities = {
        NEATController: ("Frozen NEAT generation 134", manifest["policies"]["neat"]["file"]),
        ResearchPIDController: ("Frozen tuned filtered PID", manifest["policies"]["pid"]["file"]),
    }
    arena = max(track.arena_size for track in tracks)
    resolution = max(max(track.resolution for track in tracks), arena/8192)
    simulation = Simulation(robot)
    simulation.reset(replace(tracks[0], arena_size=arena, resolution=resolution))
    simulation.enable_rendering()
    keys = SimpleQueue()
    trajectory = deque(maxlen=1200)
    policy = track_index = 0
    paused = reference = transparent = False
    episode = None
    print("C: switch frozen NEAT/PID | N/P: next/previous track | R: reset | Space: pause | T: route", flush=True)
    print(f"Shared wheel-speed cap: {robot.max_speed:.2f} m/s | frozen PID base: {controllers[1].config.base:.2f} | "
          f"PID nominal straight target: {controllers[1].config.base*robot.max_speed:.2f} m/s", flush=True)
    with mujoco.viewer.launch_passive(simulation.model, simulation.data, key_callback=keys.put) as viewer:
        reset = True
        while viewer.is_running():
            started = time.perf_counter()
            while not keys.empty():
                key = keys.get()
                if key == 32:
                    paused = not paused
                elif key in (ord("N"), ord("P")):
                    track_index = (track_index+(1 if key == ord("N") else -1)) % len(tracks)
                    reset = True
                elif key == ord("C"):
                    policy, reset = 1-policy, True
                elif key == ord("R"):
                    reset = True
                elif key == ord("T"):
                    reference = not reference
                elif key == ord("B"):
                    transparent = not transparent
                    set_transparency(simulation, transparent)
            if reset:
                episode = Episode(simulation, tracks[track_index], controllers[policy], config, detailed=True)
                upload_display(simulation, tracks[track_index], viewer, arena, resolution)
                camera = camera_for(simulation, "track")
                with viewer.lock():
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
                trajectory.clear()
                paused, reset = False, False
                active = episode.controller
                label, filename = identities[type(active)]
                throttle = f"base {active.config.base:.2f}" if isinstance(active, ResearchPIDController) else "learned throttle"
                print(f"Track {track_index+1}/10: {tracks[track_index].name} | {label} | "
                      f"controller={type(active).__name__} | file={filename} | "
                      f"cap {robot.max_speed:.2f} m/s | {throttle}", flush=True)
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
            time.sleep(max(0, robot.dt-(time.perf_counter()-started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    view(args.run.resolve())


if __name__ == "__main__":
    main()
