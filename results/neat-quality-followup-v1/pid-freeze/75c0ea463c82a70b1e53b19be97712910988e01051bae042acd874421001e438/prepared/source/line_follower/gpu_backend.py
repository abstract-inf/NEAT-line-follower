"""Opt-in batched CUDA physics with the same sensor/episode accounting as CPU.

NEAT inference, reflectance lookup, and route evaluation remain on CPU. This is
an experimental backend: float32 contact dynamics can change later binary
observations. Never pool CPU/GPU results without recording the backend.
"""

from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import math
import time

import mujoco
import numpy as np

from .config import EpisodeConfig, RobotConfig
from .controllers import NEATController
from .episodes import Episode
from .simulation import Simulation, sample_sensor_rays
from .tracks import TrackMask, TrackSpec
from .population_view import SnapshotPublisher, robot_snapshot

# A lowered PCB can generate >300 contacts and >900 solver rows in one
# world. Reserve headroom for tipping/board contact, including 1-4 world tails.
# Contact slots form a shared batch pool; constraint rows are per world.
CONTACTS_PER_WORLD = 512
CONSTRAINTS_PER_WORLD = 2048
COMPACTION_MIN_WORLDS = 16  # Smaller tails are cheaper than another graph/buffer transition.
# These fields are required by sensor rays and mj_objectVelocity. All derive
# from the captured GPU forward pass; CPU mj_forward would repeat the physics.
STATE_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "xpos", "xmat", "xipos", "subtree_com", "cvel",
                "site_xpos", "site_xmat", "actuator_force")


class GPUEvaluator:
    """Batch independent episodes; retain one resettable CPU data view per world."""

    def __init__(self, robot: RobotConfig, episode: EpisodeConfig, neat_config,
                 time_limit: float | None, batch_size: int):
        os.environ.setdefault("WARP_CACHE_PATH", str(Path(".cache/warp").resolve()))
        try:
            import warp as wp
            import mujoco_warp as mjw
        except ImportError as error:
            raise RuntimeError("Install the gpu extra before selecting --backend gpu") from error
        wp.init()
        if any(importlib.metadata.version(name) != "4.0.0" for name in ("mujoco", "mujoco-warp")):
            raise RuntimeError("The experimental GPU adapter requires mujoco and mujoco-warp 4.0.0")
        if not wp.is_cuda_available():
            raise RuntimeError("GPU backend explicitly requested but no CUDA device is available")
        self.wp, self.mjw = wp, mjw
        self.robot, self.episode, self.neat_config = robot, episode, neat_config
        self.time_limit, self.batch_size = time_limit, batch_size
        self.template = Simulation(robot)
        probe = deepcopy(self.template.model)
        probe.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        with wp.ScopedDevice("cuda:0"):
            self.model = mjw.put_model(probe)
        self.cpu_model = probe
        self.buffers = OrderedDict()
        self.masks: OrderedDict[str, TrackMask] = OrderedDict()
        self.executed_world_steps = 0
        self.padding_world_steps = 0
        self.batch_profiles: list[dict] = []

    def mask(self, track: TrackSpec) -> TrackMask:
        identity = hashlib.sha256(json.dumps(asdict(track), sort_keys=True).encode()).hexdigest()
        if identity not in self.masks:
            self.masks[identity] = TrackMask(track)
            while len(self.masks) > 4:
                self.masks.popitem(last=False)
        self.masks.move_to_end(identity)
        return self.masks[identity]

    def _buffer(self, episodes: list[Episode]):
        count = len(episodes)
        wp, mjw = self.wp, self.mjw
        if count not in self.buffers:
            with wp.ScopedDevice("cuda:0"):
                data = mjw.put_data(self.cpu_model, episodes[0].simulation.data, nworld=count,
                                    nconmax=CONTACTS_PER_WORLD, njmax=CONSTRAINTS_PER_WORLD)
                mjw.step(self.model, data)  # Compile before CUDA graph capture.
                wp.synchronize()
                with wp.ScopedCapture() as capture:
                    for _ in range(self.robot.physics_hz // self.robot.control_hz):
                        mjw.step(self.model, data)
                    mjw.forward(self.model, data)
                self.buffers[count] = data, capture.graph
                # Compaction sizes vary by generation. Bound retained device buffers.
                while len(self.buffers) > 3:
                    oldest = next(key for key in self.buffers if key != self.batch_size)
                    del self.buffers[oldest]
        self.buffers.move_to_end(count)
        data, graph = self.buffers[count]
        for name in ("qpos", "qvel", "act", "qacc_warmstart"):
            target = getattr(data, name)
            values = np.stack([getattr(e.simulation.data, name) for e in episodes]).astype(np.float32)
            target.assign(values)
        data.time.assign(np.array([e.simulation.data.time for e in episodes], dtype=np.float32))
        data.ctrl.zero_()
        data.overflow.zero_()
        return data, graph

    def _batch(self, tasks: list[tuple], progress=None, publisher=None, track_index=0) -> list:
        batch_started = time.perf_counter()
        episodes = [Episode(Simulation(self.robot, model=self.template.model), track,
                            NEATController(genome, self.neat_config), self.episode,
                            self.mask(track), self.time_limit) for genome, track in tasks]
        data, graph = self._buffer(episodes)
        profile = {"batch": len(self.batch_profiles)+1, "worlds": len(tasks),
                   "setup_wall_s": time.perf_counter()-batch_started, "inference_wall_s": 0.0,
                   "physics_transfer_wall_s": 0.0, "accounting_observer_wall_s": 0.0,
                   "compaction_wall_s": 0.0, "max_control_block_wall_s": 0.0}
        starting_steps, starting_padding = self.executed_world_steps, self.padding_world_steps
        all_episodes = episodes[:]
        all_tasks = tasks[:]
        track_indices = track_index if isinstance(track_index, list) else [track_index]*len(tasks)
        def publish(force=False):
            if publisher is not None and publisher.context is not None and (force or publisher.due):
                publisher.publish([robot_snapshot(genome.key, slot, episode)
                                   for (genome, _), slot, episode in zip(all_tasks, track_indices, all_episodes)], force)
        publish(True)
        ratio = self.robot.physics_hz // self.robot.control_hz
        last_report = time.perf_counter()
        while any(not episode.done for episode in episodes):
            block_started = time.perf_counter()
            compaction_wall = 0.0
            observations = [episode.observation.copy() for episode in episodes]
            commands = np.zeros((len(episodes), 2), dtype=np.float32)
            for index, episode in enumerate(episodes):
                if episode.done:
                    continue
                action = np.asarray(episode.controller.act(observations[index], self.robot.dt))
                if action.shape != (2,) or not np.all(np.isfinite(action)):
                    raise ValueError("Non-finite or malformed GPU episode action")
                commands[index] = np.clip(action, -1, 1)
                episode.simulation.last_action = commands[index].astype(float)
            inference_finished = time.perf_counter()
            profile["inference_wall_s"] += inference_finished-block_started
            data.ctrl.assign(commands * np.float32(self.robot.max_speed/self.robot.wheel_radius))
            self.wp.capture_launch(graph)
            if np.any(data.overflow.numpy()):
                raise RuntimeError("GPU contact/constraint/solver capacity overflow; no fitness may be reported from this batch")
            self.executed_world_steps += len(episodes)*ratio
            self.padding_world_steps += sum(e.done for e in episodes)*ratio
            # Copy only the physical state required by the shared observation
            # and metrics code. No floor texture or rendering goes to workers.
            states = {name: getattr(data, name).numpy() for name in STATE_FIELDS}
            physics_finished = time.perf_counter()
            profile["physics_transfer_wall_s"] += physics_finished-inference_finished
            sites = self.template.site_ids
            readings = np.zeros((len(episodes), len(sites)))
            hits = np.full((len(episodes), len(sites), 3), np.nan)
            groups = {}
            for index, episode in enumerate(episodes):
                groups.setdefault(id(episode.simulation.mask), []).append(index)
            for indices in groups.values():
                mask = episodes[indices[0]].simulation.mask
                readings[indices], hits[indices] = sample_sensor_rays(states["site_xpos"][indices][:, sites],
                    states["site_xmat"][indices][:, sites].reshape(len(indices), len(sites), 3, 3),
                    mask, self.robot.sensor_max_distance)
            for index, episode in enumerate(episodes):
                if episode.done:
                    continue
                simulation = episode.simulation
                for name, values in states.items():
                    target = getattr(simulation.data, name)
                    target[:] = values[index].reshape(target.shape)
                # Episode time is an integer control clock. Float32 CUDA time
                # accumulation must not add a spurious step to timeout rules.
                simulation.data.time = simulation._start_time + (episode.steps+1)*self.robot.dt
                simulation.observation = readings[index]
                simulation.sensor_hits[:] = hits[index]
                episode.observe_step(observations[index])
            publish()
            if progress is not None and time.perf_counter()-last_report >= 2:
                last_report = time.perf_counter()
                progress({"completed_episodes": sum(e.done for e in all_episodes),
                          "simulated_seconds": max(e.steps*self.robot.dt for e in all_episodes),
                          "batch_wall_seconds": time.perf_counter()-batch_started,
                          "active_worlds": sum(not e.done for e in all_episodes),
                          "aggregate_simulated_seconds": sum(e.steps*self.robot.dt for e in all_episodes)})
            # Stop spending most CUDA work on already terminated worlds. Restore
            # continuing qpos/qvel, actuator filter and solver warm-start state;
            # control frequency, observations and episode accounting stay identical.
            alive = [episode for episode in episodes if not episode.done]
            if alive and len(episodes) > COMPACTION_MIN_WORLDS and len(alive) <= len(episodes)//2:
                episodes = alive
                compact_started = time.perf_counter()
                data, graph = self._buffer(episodes)
                compaction_wall = time.perf_counter()-compact_started
                profile["compaction_wall_s"] += compaction_wall
            block_wall = time.perf_counter()-block_started
            profile["accounting_observer_wall_s"] += time.perf_counter()-physics_finished-compaction_wall
            profile["max_control_block_wall_s"] = max(profile["max_control_block_wall_s"], block_wall)
        publish(True)
        if progress is not None:
            progress({"completed_episodes": len(all_episodes),
                      "simulated_seconds": max(e.steps*self.robot.dt for e in all_episodes),
                      "batch_wall_seconds": time.perf_counter()-batch_started,
                      "active_worlds": 0,
                      "aggregate_simulated_seconds": sum(e.steps*self.robot.dt for e in all_episodes)})
        results = [episode.result() for episode in all_episodes]
        profile.update(wall_seconds=time.perf_counter()-batch_started,
                       simulated_world_seconds=sum(r.elapsed_s for r in results),
                       executed_world_steps=self.executed_world_steps-starting_steps,
                       padding_world_steps=self.padding_world_steps-starting_padding)
        self.batch_profiles.append(profile)
        if any(result.error for result in results):
            raise RuntimeError(f"GPU evaluation saved explicit technical failures: {[r.error for r in results if r.error]}")
        return results

    def evaluate(self, genomes: list, tracks: list[TrackSpec], progress=None, live=None) -> dict:
        # One population on one track per batch improves mask locality and makes
        # the display an honest simultaneous population view. Scatter back into
        # the original genome/track order before calculating any fitness.
        self.batch_profiles = []
        results = {genome.key: [None]*len(tracks) for genome in genomes}
        tasks = [(genome, track, index) for index, track in enumerate(tracks) for genome in genomes]
        total = len(tasks)
        with self.wp.ScopedDevice("cuda:0"):
            for first in range(0, total, self.batch_size):
                batch = tasks[first:first+self.batch_size]
                publisher = SnapshotPublisher(live, f"gpu-batch-{first}")
                def report(update):
                    if progress is not None:
                        progress({**update, "completed_episodes": first+update["completed_episodes"],
                                  "total_episodes": total, "batch": first//self.batch_size+1,
                                  "batches": math.ceil(total/self.batch_size)})
                report({"completed_episodes": 0, "simulated_seconds": 0})
                evaluated = self._batch([(genome, track) for genome, track, _ in batch],
                                        report if progress else None, publisher, [slot for _, _, slot in batch])
                for (genome, _, slot), result in zip(batch, evaluated):
                    results[genome.key][slot] = result
        return results

    def close(self) -> None:
        self.buffers.clear()
