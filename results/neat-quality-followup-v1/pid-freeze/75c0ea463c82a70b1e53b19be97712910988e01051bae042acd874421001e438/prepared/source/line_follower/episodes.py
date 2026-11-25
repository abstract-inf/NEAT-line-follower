"""One episode loop shared by headless evaluation and the interactive viewer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path

import mujoco
import numpy as np

from .config import EpisodeConfig
from .controllers import Controller
from .fitness import LegacyScorer, RouteProgress, progress_fitness
from .simulation import Simulation
from .tracks import SENSOR_COUNT, TrackMask, TrackSpec

TELEMETRY_FIELDS = ["step", "time_s", "x_m", "y_m", "yaw_rad", "progress", "lateral_error_m",
                    "forward_speed_m_s", "yaw_rate_rad_s", "action_L", "action_R",
                    "wheel_L_m_s", "wheel_R_m_s", "torque_L_Nm", "torque_R_Nm",
                    "line_lost", "planned_gap", "legacy_increment", "observation_time_s"] + [f"sensor{i}" for i in range(SENSOR_COUNT)] + [f"next_sensor{i}" for i in range(SENSOR_COUNT)]


@dataclass(frozen=True)
class EpisodeResult:
    """All failures remain rows. completion_time_s is absent for incomplete runs."""

    track_name: str
    track_hash: str
    track_seed: int | None
    split: str
    difficulty: str
    features: str
    completed: bool
    termination: str
    error: str
    elapsed_s: float
    completion_time_s: float | None
    time_limit_s: float
    progress: float
    progress_m: float
    route_length_m: float
    lateral_rmse_m: float
    sensor_index_rmse: float
    average_forward_speed_m_s: float
    line_loss_events: int
    line_loss_seconds: float
    unplanned_line_loss_events: int
    mean_action_L: float
    mean_action_R: float
    motor_asymmetry: float
    control_steps: int
    physics_steps: int
    legacy_return: float
    score: float
    command_jitter: float = 0.0
    peak_forward_speed_m_s: float = 0.0
    geometry_lateral_rmse_m: float | None = None
    straight_speed_m_s: float | None = None


class Episode:
    """Explicit episode state; route data never enters the controller call."""

    def __init__(self, simulation: Simulation, track: TrackSpec, controller: Controller,
                 config: EpisodeConfig | None = None, mask: TrackMask | None = None,
                 time_limit: float | None = None, *, detailed: bool = False):
        self.simulation = simulation
        self.track = track
        self.controller = controller
        self.config = config or EpisodeConfig()
        self.time_limit = time_limit or self.config.time_limit(track.route.length, simulation.config.max_speed)
        if self.time_limit <= 0:
            raise ValueError("Episode time limit must be positive")
        self.observation = simulation.reset(track, mask)
        controller.reset()
        self.route = RouteProgress(track, self.config)
        self.legacy = LegacyScorer()
        self.steps = 0
        self.termination = ""
        self.error = ""
        self.last_progress_time = 0.0
        self.progress_anchor = 0.0
        self.departure_time = 0.0
        self.squared_error = 0.0
        self.sensor_squared_error = 0.0
        self.last_sensor_error = 0.0
        self.speed_sum = 0.0
        self.actions_sum = np.zeros(2)
        self.asymmetry_sum = 0.0
        self.previous_left_command = self.previous_right_command = 0.0
        self.squared_command_change = 0.0
        self.line_loss_events = 0
        self.unplanned_line_loss_events = 0
        self.line_loss_steps = 0
        self.was_lost = False
        self.peak_forward_speed = 0.0
        self.positions = [] if detailed else None
        self.straight_entry_time = self.straight_exit_time = None
        self.previous_progress = self.previous_time = 0.0
        self.straight_window = None
        if detailed and self.track.route.length >= 4 and all(s.kind == "straight" for s in self.track.sections):
            middle = self.track.route.length / 2
            self.straight_window = (middle - 1, middle + 1)

    @property
    def done(self) -> bool:
        return bool(self.termination)

    def advance(self, telemetry: bool = False) -> dict | None:
        """Advance exactly one control step; optionally return an auditable trace."""
        if self.done:
            return None
        controller_observation = self.observation.copy()
        try:
            action = self.controller.act(controller_observation.copy(), self.simulation.config.dt)
            self.observation = self.simulation.step(action)
        except (ValueError, FloatingPointError, mujoco.FatalError) as error:
            self.termination = "controller_or_simulation_error"
            self.error = f"{type(error).__name__}: {error}"
            return None
        return self.observe_step(controller_observation, telemetry)

    def observe_step(self, controller_observation: np.ndarray, telemetry: bool = False) -> dict | None:
        """Account for one completed physics block, also used by batched GPU physics.

        External steppers must update Simulation data, observation, and the last
        action first. Controller input remains sensors and dt only.
        """
        if self.done:
            return None
        self.observation = self.simulation.observation.copy()
        self.steps += 1
        state = self.simulation.state
        if not np.all(np.isfinite(self.simulation.data.qpos)) or not np.all(np.isfinite(self.simulation.data.qvel)):
            self.termination = "invalid_physics"
            self.error = "Non-finite physical state"
            return None
        self.route.update(state.position)
        self._check_termination(state.time)
        zone = "green" if self.termination == "completed" else "red" if self.termination == "off_route" else None
        legacy_increment = self.legacy.update(self.observation, state, self.simulation.config, zone)
        planned_gap = any(begin <= self.route.distance <= end for begin, end in self.track.gaps)
        lost = not bool(np.any(self.observation))
        if lost and not self.was_lost:
            self.line_loss_events += 1
            self.unplanned_line_loss_events += int(not planned_gap)
        self.line_loss_steps += int(lost)
        self.was_lost = lost
        if np.any(self.observation):
            self.last_sensor_error = float(np.dot(np.arange(SENSOR_COUNT), self.observation) / np.sum(self.observation) - 7)
        self.sensor_squared_error += self.last_sensor_error ** 2
        self.squared_error += self.route.lateral_error ** 2
        self.speed_sum += state.forward_speed
        self.peak_forward_speed = max(self.peak_forward_speed, state.forward_speed)
        if self.positions is not None:
            self.positions.append(state.position[:2].copy())
        if self.straight_window:
            for boundary, field in zip(self.straight_window, ("straight_entry_time", "straight_exit_time")):
                if getattr(self, field) is None and self.previous_progress < boundary <= self.route.distance:
                    fraction = (boundary - self.previous_progress) / (self.route.distance - self.previous_progress)
                    setattr(self, field, self.previous_time + fraction * (state.time - self.previous_time))
        self.previous_progress, self.previous_time = self.route.distance, state.time
        self.actions_sum += self.simulation.last_action
        left, right = self.simulation.last_action
        self.asymmetry_sum += abs(float(left - right))
        if self.steps > 1:
            self.squared_command_change += (left - self.previous_left_command) ** 2 + (right - self.previous_right_command) ** 2
        self.previous_left_command, self.previous_right_command = left, right
        if not telemetry:
            return None
        row = dict(zip(TELEMETRY_FIELDS[:18], [self.steps, state.time, *state.position[:2], state.yaw,
                   self.route.fraction, self.route.lateral_error, state.forward_speed, state.yaw_rate,
                   *self.simulation.last_action, *state.wheel_speeds, *state.torques,
                   int(lost), int(planned_gap), legacy_increment]))
        row["observation_time_s"] = state.time - self.simulation.config.dt
        row.update({f"sensor{i}": int(value) for i, value in enumerate(controller_observation)})
        row.update({f"next_sensor{i}": int(value) for i, value in enumerate(self.observation)})
        return row

    def _check_termination(self, elapsed: float) -> None:
        if self.route.distance - self.progress_anchor >= self.config.progress_increment:
            self.progress_anchor = self.route.distance
            self.last_progress_time = elapsed
        tracking_line = self.route.inside_corridor or bool(np.any(self.observation))
        self.departure_time = 0.0 if tracking_line else self.departure_time + self.simulation.config.dt
        upright = self.simulation.data.xmat[self.simulation.chassis_id].reshape(3, 3)[2, 2]
        if upright < 0.5 or self.simulation.data.xpos[self.simulation.chassis_id, 2] < 0:
            self.termination = "tipped"
        elif self.route.completed:
            self.termination = "completed"
        elif self.departure_time >= self.config.departure_seconds:
            self.termination = "off_route"
        elif elapsed - self.last_progress_time >= self.config.stall_seconds:
            self.termination = "stalled"
        elif elapsed >= self.time_limit - 1e-10:
            self.termination = "timeout"

    def result(self) -> EpisodeResult:
        if not self.done:
            raise RuntimeError("An episode result requires an explicit termination")
        state = self.simulation.state
        count = max(1, self.steps)
        completed = self.termination == "completed"
        means = self.actions_sum / count
        # Both clipped wheel commands can change by at most 2 in one control step.
        jitter = float(self.squared_command_change / (8 * max(1, self.steps - 1)))
        score = progress_fitness(self.route.fraction, completed, state.time, self.time_limit,
                                 self.legacy.total, self.steps, self.config.completion_speed_bonus,
                                 command_jitter=jitter, command_jitter_penalty=self.config.command_jitter_penalty,
                                 route_length=self.track.route.length, fitness_version=self.config.fitness_version)
        geometric_rmse = straight_speed = None
        if self.positions and not self.error:
            from .pid_experiments import geometric_tracking_error
            geometric_rmse, _ = geometric_tracking_error(np.asarray(self.positions), self.track.route.points)
        if self.straight_exit_time is not None:
            straight_speed = 2 / (self.straight_exit_time - self.straight_entry_time)
        if self.error:
            score = -1.0
        return EpisodeResult(self.track.name, self.track.geometry_hash, self.track.seed, self.track.split,
                             self.track.difficulty, ";".join(self.track.features), completed, self.termination, self.error,
                             state.time, state.time if completed else None, self.time_limit,
                             self.route.fraction, self.route.distance, self.track.route.length,
                             float(np.sqrt(self.squared_error / count)), float(np.sqrt(self.sensor_squared_error / count)),
                             self.speed_sum / count, self.line_loss_events, self.line_loss_steps * self.simulation.config.dt,
                             self.unplanned_line_loss_events, float(means[0]), float(means[1]), self.asymmetry_sum / count,
                             self.steps, self.steps * (self.simulation.config.physics_hz // self.simulation.config.control_hz),
                             self.legacy.total, score, jitter, self.peak_forward_speed, geometric_rmse, straight_speed)


def run_episode(simulation: Simulation, track: TrackSpec, controller: Controller,
                config: EpisodeConfig | None = None, mask: TrackMask | None = None,
                telemetry_path: str | Path | None = None, time_limit: float | None = None,
                observer=None, detailed: bool = False) -> EpisodeResult:
    """Run unpaced CPU physics; a rendering clock is never consulted."""
    episode = Episode(simulation, track, controller, config, mask, time_limit, detailed=detailed)
    if observer is not None:
        observer(episode, True)
    if telemetry_path:
        path = Path(telemetry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS)
            writer.writeheader()
            while not episode.done:
                row = episode.advance(telemetry=True)
                if observer is not None:
                    observer(episode, episode.done)
                if row is not None:
                    writer.writerow(row)
    else:
        while not episode.done:
            episode.advance()
            if observer is not None:
                observer(episode, episode.done)
    return episode.result()
