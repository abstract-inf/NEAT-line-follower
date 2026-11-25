"""Ordered-route evaluation and separately versioned legacy/progress fitness."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import EpisodeConfig, RobotConfig
from .simulation import RobotState
from .tracks import TrackSpec

FITNESS_VERSION = "progress-1.2"  # Corridor-consistent gates and sensor-aware departure.
LEGACY_VERSION = "headless-a883c3a7"
HISTORICAL_PIXELS_PER_METER = 1000.0
HISTORICAL_WHEEL_RADIUS = 0.0165


@dataclass
class LegacyScorer:
    """Original headless arithmetic, including its known signs and unit errors.

    A compatibility adapter reconstructs the old get_velocity values from wheel
    rim speeds. They are *not* physical velocities and are never used as metrics.
    Source: train_neat_multiprocess.py at a883c3a7ce4daed75836cf58688b748151d04248.
    """

    off_track_steps: int = 0
    total: float = 0.0

    def update(self, sensors: np.ndarray, state: RobotState, config: RobotConfig,
               zone: str | None = None) -> float:
        old_wheels = state.wheel_speeds * HISTORICAL_PIXELS_PER_METER
        velocity = float(np.mean(old_wheels)) * HISTORICAL_WHEEL_RADIUS
        angular = float(old_wheels[1] - old_wheels[0]) * HISTORICAL_WHEEL_RADIUS / (config.wheel_spacing * HISTORICAL_PIXELS_PER_METER)
        angular_deg = math.degrees(angular)
        absolute_speed = abs(velocity)
        center = float(np.sum(sensors[6:9]))
        minimum_speed = config.max_speed / 2
        if zone == "green":
            value = 10000 + absolute_speed * 100
        elif zone == "red":
            value = -500.0
        else:
            speed_reward = absolute_speed * (10.0 if center >= 2 else 0.5)
            if abs(angular_deg) > 20:
                speed_reward *= 0.5
            middle_bonus = 50 if sensors[7] == 1 else 0
            center_bonus = center ** 1.5 * 2 if velocity > 3 else 0
            smooth_bonus = max(0, 2 - abs(float(old_wheels[0] - old_wheels[1]))) * 0.5
            off_track_penalty = 0.0
            if not np.any(sensors):
                off_track_penalty = 50 + self.off_track_steps * 50
                self.off_track_steps += 1
            else:
                self.off_track_steps = 0
            if velocity <= 0:
                speed_penalty = 1000 * (absolute_speed + 0.05)
            elif velocity < minimum_speed:
                speed_penalty = -100 * (minimum_speed - absolute_speed)
            else:
                speed_penalty = -100 * (absolute_speed - minimum_speed)
            steering_penalty = min(abs(angular_deg) * 0.8, 30)
            zone_bonus = (500 if velocity >= minimum_speed else -1000) if zone == "yellow" else 0
            value = speed_reward + middle_bonus + center_bonus + smooth_bonus - off_track_penalty - speed_penalty - steering_penalty + zone_bonus
        self.total += value
        return value


def progress_fitness(progress: float, completed: bool, elapsed: float, time_limit: float,
                     legacy_return: float, control_steps: int,
                     completion_speed_bonus: float = 0.0, *, command_jitter: float = 0.0,
                     command_jitter_penalty: float = 0.0, route_length: float = 0.0,
                     fitness_version: str = "progress-1.2") -> float:
    """Reward progress and completion, with optional elapsed-time and jitter terms.

    F = p + completed - 0.1*t/T + 0.01*p*tanh(legacy/steps)
        + speed_bonus - lambda*jitter.
    The version chooses timeout-relative or route-speed completion bonus.
    Jitter is the mean squared wheel command change, normalized to [0, 1].
    """
    return float(sum(fitness_components(progress, completed, elapsed, time_limit, legacy_return, control_steps,
                                       completion_speed_bonus, command_jitter, command_jitter_penalty,
                                       route_length, fitness_version).values()))


def fitness_components(progress, completed, elapsed, time_limit, legacy_return, control_steps,
                       completion_speed_bonus=0.0, command_jitter=0.0, command_jitter_penalty=0.0,
                       route_length=0.0, fitness_version="progress-1.2"):
    if fitness_version == "route-speed-1.0":
        speed = min(1.0, route_length / elapsed) if completed and elapsed > 0 else 0.0
        bonus = .50 * int(completed) * speed
    elif fitness_version == "progress-1.2":
        bonus = completion_speed_bonus * int(completed) * max(0.0, min(1.0, 1 - elapsed/time_limit))
    else:
        raise ValueError("Unknown fitness version")
    return {"progress_term": float(progress), "completion_term": float(completed),
            "elapsed_term": -.1 * elapsed / time_limit,
            "legacy_term": .01 * progress * math.tanh(legacy_return / max(1, control_steps)),
            "speed_term": bonus, "jitter_term": -command_jitter_penalty * command_jitter}


def aggregate_fitness(scores: list[float] | np.ndarray, method: str = "mean_min", tail_fraction: float = .3) -> float:
    """A bad track contributes directly instead of being hidden by the mean."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Fitness aggregation needs finite episode scores")
    if method not in {"mean_min", "mean_cvar"} or not 0 < tail_fraction <= 1:
        raise ValueError("Use mean_min or mean_cvar, with tail_fraction in (0, 1]")
    tail = float(np.min(values))
    if method == "mean_cvar":
        ordered = np.sort(values)
        mass = len(values)*tail_fraction
        full = math.floor(mass)
        partial = (mass-full)*ordered[full] if full < len(values) else 0.0
        tail = float((ordered[:full].sum()+partial)/mass)
    return float(0.5 * np.mean(values) + 0.5 * tail)


class RouteProgress:
    """Evaluator-only gate state. Nearby branches cannot advance its cursor."""

    def __init__(self, track: TrackSpec, config: EpisodeConfig):
        self.route = track.route
        self.config = config
        self.gates = np.unique(np.concatenate((np.arange(0.04, self.route.length, 0.04),
                                               self.route.corners, [self.route.length])))
        self.gates = self.gates[self.gates > 0]
        self.next_gate = 0
        self.distance = 0.0
        self.lateral_error = 0.0
        self.inside_corridor = True
        self.completed = False

    def update(self, position: np.ndarray) -> None:
        """Project only onto the current gate interval, with a short recovery tail."""
        previous_gate = self.gates[self.next_gate - 1] if self.next_gate else 0.0
        next_gate = self.gates[min(self.next_gate, len(self.gates) - 1)]
        lower = max(0, int(np.searchsorted(self.route.distance, previous_gate - 0.06)) - 1)
        upper = min(len(self.route.points) - 1, int(np.searchsorted(self.route.distance, next_gate + 0.04)))
        start = self.route.points[lower:upper]
        delta = self.route.points[lower + 1:upper + 1] - start
        fraction = np.clip(np.sum((position[:2] - start) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1)
        projections = start + fraction[:, None] * delta
        index = int(np.argmin(np.linalg.norm(position[:2] - projections, axis=1)))
        vector = position[:2] - projections[index]
        distance = float(np.linalg.norm(vector))
        tangent = delta[index] / np.linalg.norm(delta[index])
        cross = float(tangent[0] * vector[1] - tangent[1] * vector[0])
        self.lateral_error = math.copysign(distance, cross if abs(cross) > 1e-12 else 1.0)
        self.inside_corridor = distance <= self.config.corridor_half_width
        if not self.inside_corridor:
            return
        segment_index = lower + index
        projected_s = float(self.route.distance[segment_index] + fraction[index] * (self.route.distance[segment_index + 1] - self.route.distance[segment_index]))
        # Gates are dense enough that a valid physical step cannot skip a section.
        # At a right-angle turn, valid positions in the adjoining corridor
        # bands can be sqrt(2) times the half-width from the corner gate.
        gate_radius = max(self.config.gate_radius, math.sqrt(2)*self.config.corridor_half_width)
        while self.next_gate < len(self.gates):
            gate_s = self.gates[self.next_gate]
            if projected_s < gate_s - 0.003 or np.linalg.norm(position[:2] - self.route.point_at(gate_s)) > gate_radius:
                break
            self.next_gate += 1
        authenticated_s = min(projected_s, float(self.gates[min(self.next_gate, len(self.gates) - 1)]))
        self.distance = max(self.distance, authenticated_s)
        self.completed = self.next_gate == len(self.gates) and projected_s >= self.route.length - 0.003
        if self.completed:
            self.distance = self.route.length

    @property
    def fraction(self) -> float:
        return min(1.0, self.distance / self.route.length)
