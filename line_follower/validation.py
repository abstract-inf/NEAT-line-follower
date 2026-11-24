"""Privileged geometry feasibility checks, excluded from controller comparisons."""

from __future__ import annotations

import math
import numpy as np

from .config import EpisodeConfig, RobotConfig
from .episodes import EpisodeResult, run_episode
from .simulation import Simulation
from .tracks import TrackSpec, validate_track


class ReferenceDriver:
    """Slow waypoint driver for checking physical traversability only.

    This explicitly uses true pose and the route. It is not PID, a learned policy,
    or an admissible controller in the scientific performance comparison.
    """

    def __init__(self, simulation: Simulation, track: TrackSpec):
        self.simulation = simulation
        samples = np.unique(np.r_[np.arange(0.06, track.route.length, 0.06), track.route.corners, track.route.length])
        self.waypoints = np.array([track.route.point_at(s) for s in samples if s > 0])
        self.reset()

    def reset(self) -> None:
        self.target_index = 0

    def act(self, _observation: np.ndarray, _dt: float) -> np.ndarray:
        state = self.simulation.state
        distance = np.linalg.norm(self.waypoints[self.target_index] - state.position[:2])
        if distance < 0.008 and self.target_index < len(self.waypoints) - 1:
            self.target_index += 1
        direction = self.waypoints[self.target_index] - state.position[:2]
        error = math.remainder(math.atan2(direction[1], direction[0]) - state.yaw, 2 * math.pi)
        speed = min(0.06, float(np.linalg.norm(direction)) * 2) if abs(error) < 0.25 else 0.0
        turn = float(np.clip(error * 3.0, -1.2, 1.2))
        half_spacing = self.simulation.config.wheel_spacing / 2
        return np.clip([(speed - half_spacing * turn) / self.simulation.config.max_speed,
                        (speed + half_spacing * turn) / self.simulation.config.max_speed], -1, 1)


def validate_physics(track: TrackSpec, robot: RobotConfig | None = None) -> EpisodeResult:
    """A passing result establishes nominal low-speed feasibility, not learnability."""
    errors = validate_track(track)
    if errors:
        raise ValueError("; ".join(errors))
    simulation = Simulation(robot)
    driver = ReferenceDriver(simulation, track)
    return run_episode(simulation, track, driver, EpisodeConfig(stall_seconds=10),
                       time_limit=30 + track.route.length / 0.03)
