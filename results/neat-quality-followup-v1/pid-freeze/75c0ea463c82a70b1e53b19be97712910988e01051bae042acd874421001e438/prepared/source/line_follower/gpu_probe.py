"""Optional GPU physics/parity benchmark; never runs a NEAT training loop.

Benchmark the actual robot against persistent CPU workers before choosing a
backend. Physics-only throughput excludes policy inference, sensing and route
evaluation, so it is not an estimate of complete evolutionary training speed.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from .experiments import environment_record, write_json
from .simulation import Simulation
from .tracks import straight_track


@lru_cache(maxsize=1)
def _cpu_initial() -> Simulation:
    simulation = Simulation()
    simulation.reset(straight_track())
    return simulation


def _cpu_rollout(actions: np.ndarray) -> np.ndarray:
    simulation = _cpu_initial()
    data = deepcopy(simulation.data)
    ratio = simulation.config.physics_hz // simulation.config.control_hz
    for action in actions:
        data.ctrl[:] = action * simulation.config.max_speed / simulation.config.wheel_radius
        mujoco.mj_step(simulation.model, data, nstep=ratio)
        mujoco.mj_forward(simulation.model, data)
    return data.qpos.copy()


def benchmark_gpu(output: str | Path, worlds: int = 32, control_steps: int = 60, workers: int = 4) -> Path:
    """Measure a captured CUDA graph and an equivalent CPU multiprocessing job."""
    if min(worlds, control_steps, workers) < 1:
        raise ValueError("World count, step count and worker count must be positive")
    # Set the cache before Warp initializes: its own Windows path handling adds
    # the extended-path prefix needed for kernels in long workspace directories.
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(".cache/warp").resolve()))
    try:
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as error:
        raise RuntimeError("Install the optional gpu extra: pip install -e '.[gpu]'") from error
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("No CUDA device detected; CPU workers remain available")
    simulation = _cpu_initial()
    # MJWarp 3.17 rejects this diagnostic flag. Change only a separate probe
    # model and record the difference; production CPU physics is untouched.
    probe_model = deepcopy(simulation.model)
    probe_model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    actions = np.array([[.6, .6] if i < control_steps/3 else
                        [.35, -.35] if i < 2*control_steps/3 else [0, 0]
                        for i in range(control_steps)], dtype=float)
    started = time.perf_counter()
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(probe_model)
        data = mjw.put_data(probe_model, simulation.data, nworld=worlds, nconmax=32, njmax=128)
        mjw.step(model, data)
        wp.synchronize()
        # Restore after JIT warmup, then capture one complete 60 Hz physics block.
        data = mjw.put_data(probe_model, simulation.data, nworld=worlds, nconmax=32, njmax=128)
        with wp.ScopedCapture() as capture:
            for _ in range(simulation.config.physics_hz // simulation.config.control_hz):
                mjw.step(model, data)
            mjw.forward(model, data)
        wp.synchronize()
        setup_seconds = time.perf_counter()-started
        started = time.perf_counter()
        for action in actions:
            targets = np.tile(action * simulation.config.max_speed/simulation.config.wheel_radius, (worlds, 1)).astype(np.float32)
            wp.copy(data.ctrl, wp.array(targets, dtype=wp.float32))
            wp.capture_launch(capture.graph)
        wp.synchronize()
        gpu_qpos = data.qpos.numpy()
        gpu_seconds = time.perf_counter()-started
    with mp.get_context("spawn").Pool(workers, initializer=_cpu_initial) as pool:
        pool.map(_cpu_rollout, [actions[:1]]*workers)  # Exclude worker setup/JIT from steady-state comparison.
        started = time.perf_counter()
        cpu_qpos = np.array(pool.map(_cpu_rollout, [actions]*worlds))
        cpu_seconds = time.perf_counter()-started
    position_error = float(np.max(np.linalg.norm(cpu_qpos[:, :3]-gpu_qpos[:, :3], axis=1)))
    quaternion_dot = np.abs(np.sum(cpu_qpos[:, 3:7]*gpu_qpos[:, 3:7], axis=1))
    orientation_error = float(np.max(2*np.arccos(np.clip(quaternion_dot, 0, 1))))
    record = {"environment": environment_record(), "device": str(wp.get_device("cuda:0")),
              "gpu_name": wp.get_device("cuda:0").name,
              "mujoco_warp_version": importlib.metadata.version("mujoco-warp"),
              "warp_version": importlib.metadata.version("warp-lang"),
              "worlds": worlds, "control_steps_per_world": control_steps, "cpu_workers": workers,
              "gpu_setup_and_compilation_s": setup_seconds, "gpu_steady_state_s": gpu_seconds,
              "cpu_parallel_steady_state_s": cpu_seconds, "physics_only_speed_ratio_cpu_over_gpu": cpu_seconds/gpu_seconds,
              "maximum_final_position_difference_m": position_error,
              "maximum_final_orientation_difference_rad": orientation_error,
              "finite_state": bool(np.all(np.isfinite(gpu_qpos))),
              "probe_parity_passed": bool(position_error < .002 and orientation_error < .02),
              "model_difference": "AUTORESET disable flag unsupported by MJWarp and omitted only in probe clone",
              "scope": "Open-loop physics benchmark only. No NEAT training or full controller/backend equivalence claim.",
              "production_backend": "CPU MuJoCo with persistent multiprocessing"}
    destination = Path(output)
    write_json(destination, record)
    return destination
