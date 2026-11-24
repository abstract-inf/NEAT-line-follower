"""Contact-buffer stress tests without population creation or NEAT evolution."""

import json
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.gpu_backend import GPUEvaluator
from line_follower.simulation import Simulation
from line_follower.tracks import straight_track


@pytest.mark.skipif(os.environ.get("RUN_MUJOCO_GPU_TEST") != "1", reason="Opt-in CUDA capacity stress check")
@pytest.mark.parametrize("world_count", [1, 4, 200])
def test_gpu_capacity_handles_contact_heavy_poses_and_small_batches(world_count, tmp_path):
    robot = RobotConfig()
    evaluator = GPUEvaluator(robot, EpisodeConfig(), None, None, 200)
    track = straight_track()
    first = Simulation(robot, model=evaluator.template.model)
    first.reset(track)
    worlds = [SimpleNamespace(simulation=first)]
    for _ in range(world_count-1):
        simulation = Simulation(robot, model=evaluator.template.model)
        simulation.reset(track, first.mask)
        worlds.append(SimpleNamespace(simulation=simulation))
    base = first.data.qpos.copy()
    measurements = []
    try:
        with evaluator.wp.ScopedDevice("cuda:0"):
            evaluator._buffer(worlds)  # Compile graphs from a stable starting pose.
            # The lowered PCB intentionally contacts many collision pieces at
            # once. Tilted/side/inverted poses cover fallen-robot configurations.
            for name, height, roll in [("lowered", .018, 0), ("tilted", .04, .5),
                                       ("side", .092, np.pi/2), ("inverted", -.017, np.pi)]:
                for world in worlds:
                    data = world.simulation.data
                    data.qpos[:] = base
                    data.qpos[2] = height
                    data.qpos[3:7] = [np.cos(roll/2), np.sin(roll/2), 0, 0]
                    data.qvel[:] = 0
                    data.act[:] = 0
                    data.qacc_warmstart[:] = 0
                    mujoco.mj_forward(world.simulation.model, data)
                data, graph = evaluator._buffer(worlds)
                peak_pairs = peak_contacts = peak_constraints = 0
                for _ in range(3):
                    evaluator.wp.capture_launch(graph)
                    assert not np.any(data.overflow.numpy()), f"GPU capacity overflow: {name}, {world_count} worlds"
                    assert np.all(np.isfinite(data.qpos.numpy()))
                    peak_pairs = max(peak_pairs, int(np.max(data.ncollision.numpy())))
                    peak_contacts = max(peak_contacts, int(np.max(data.nacon.numpy())))
                    peak_constraints = max(peak_constraints, int(np.max(data.nefc.numpy())))
                measurements.append({"pose": name, "worlds": world_count, "contact_capacity": data.naconmax,
                                     "constraints_per_world": data.njmax, "peak_candidate_pairs": peak_pairs,
                                     "peak_contacts": peak_contacts, "peak_constraints": peak_constraints})
    finally:
        evaluator.close()
    (tmp_path / "gpu-capacity.json").write_text(json.dumps(measurements, indent=2))
