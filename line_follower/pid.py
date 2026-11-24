"""Research PID variants using only physical binary sensors and control time.

The historical PID in controllers.py remains unchanged. Corner supervision is
optional and explicitly identifies a different controller family. Command-based
odometry below estimates a short approach distance; it never reads simulator
pose, wheel velocity, route geometry, progress or a track label.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .config import RobotConfig
from .simulation import sensor_locations
from .tracks import SENSOR_COUNT


@dataclass(frozen=True)
class ResearchPIDConfig:
    """Editable gains use normalized sensor-index error in [-1, 1]."""

    kp: float = .84
    ki: float = .0007
    kd: float = .007
    base: float = .60
    minimum_command: float = .10
    derivative_filter_s: float = .04
    integral_limit: float = .40
    slowdown_gain: float = 1.0
    loss_hold_s: float = .35
    search_command: float = .40
    corner_supervision: bool = False
    approach_command: float = .35
    turn_lead_m: float = .01
    pivot_command: float = .40

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "corner_supervision" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.corner_supervision, bool):
            raise ValueError("corner_supervision must be boolean")
        if not 0 < self.minimum_command <= self.base <= 1:
            raise ValueError("Need 0 < minimum_command <= base <= 1")
        if not 0 < self.derivative_filter_s or not 0 < self.integral_limit:
            raise ValueError("Derivative filter and integral limit must be positive")
        if not all(0 < value <= 1 for value in (self.search_command, self.approach_command, self.pivot_command)):
            raise ValueError("Recovery motor commands must lie in (0,1]")

    @property
    def family(self) -> str:
        return "pid_corner_supervised" if self.corner_supervision else "pid_filtered"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class ResearchPIDController:
    """Filtered centroid PID with bounded integral and declared recovery logic."""

    CORNER_MINIMUM_ANGLE = .65  # rad of estimated rotation before accepting a central signal.
    CORNER_TIMEOUT_S = 2.5
    CORNER_COOLDOWN_S = .25

    def __init__(self, config: ResearchPIDConfig, robot: RobotConfig):
        self.config = config
        self.robot = robot
        self.sensor_x = sensor_locations(robot)[:, 0]
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        self.direction = 1.0
        self.loss_time = 0.0
        self.last_valid_action = np.zeros(2)
        self.estimated_forward_speed = 0.0
        self.mode = "tracking"
        self.mode_time = 0.0
        self.approach_remaining = 0.0
        self.estimated_turn = 0.0
        self.cooldown = 0.0
        self.debug = {"pid_error": 0.0, "pid_integral": 0.0, "pid_derivative": 0.0,
                      "pid_turn": 0.0, "pid_mode": self.mode}

    def _command(self, action, dt: float) -> np.ndarray:
        command = np.clip(np.asarray(action, dtype=float), -1, 1)
        target = float(np.mean(command))*self.robot.max_speed
        fraction = 1-math.exp(-dt/self.robot.motor_time_constant)
        self.estimated_forward_speed += fraction*(target-self.estimated_forward_speed)
        self.debug.update(pid_integral=self.integral, pid_derivative=self.derivative, pid_mode=self.mode)
        return command

    def _supervised_action(self, active: np.ndarray, error: float, dt: float):
        self.mode_time += dt
        central = bool(np.any((active >= 6) & (active <= 8)))
        if self.mode == "approach":
            self.approach_remaining -= max(0, self.estimated_forward_speed)*dt
            if self.approach_remaining > 0:
                return [self.config.approach_command]*2
            if central:
                # A main line continuing through a branch remains under the nose.
                self.mode = "tracking"
                self.cooldown = self.CORNER_COOLDOWN_S
                return None
            self.mode, self.mode_time, self.estimated_turn = "pivot", 0.0, 0.0
        if self.mode == "pivot":
            self.estimated_turn += 2*self.robot.max_speed*self.config.pivot_command/self.robot.wheel_spacing*dt
            if self.estimated_turn >= self.CORNER_MINIMUM_ANGLE and central and len(active) <= 3:
                self.mode = "tracking"
                self.cooldown = self.CORNER_COOLDOWN_S
                self.integral, self.derivative, self.previous_error = 0.0, 0.0, error
                return None
            if self.mode_time > self.CORNER_TIMEOUT_S:
                self.mode = "tracking"
                self.cooldown = self.CORNER_COOLDOWN_S
                return None
            return [self.direction*self.config.pivot_command, -self.direction*self.config.pivot_command]
        return None

    def act(self, observation: np.ndarray, dt: float) -> np.ndarray:
        readings = np.asarray(observation)
        if readings.shape != (SENSOR_COUNT,) or not np.all(np.isfinite(readings)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("PID requires 15 finite readings and a positive finite timestep")
        active = np.flatnonzero(readings > 0)
        error = float(np.mean(active)/7-1) if len(active) else self.previous_error
        self.debug.update(pid_error=error, pid_turn=0.0)
        self.cooldown = max(0, self.cooldown-dt)
        if self.mode != "tracking":
            supervised = self._supervised_action(active, error, dt)
            if supervised is not None:
                return self._command(supervised, dt)
        if not len(active):
            self.loss_time += dt
            if self.loss_time <= self.config.loss_hold_s:
                return self._command(self.last_valid_action, dt)
            # Alternate the search after one second; loss direction is ambiguous.
            sweep = self.direction*(-1 if int(self.loss_time-self.config.loss_hold_s) % 2 else 1)
            self.mode = "tracking"
            return self._command([sweep*self.config.search_command, -sweep*self.config.search_command], dt)
        self.loss_time = 0.0
        if abs(error) > .02:
            self.direction = math.copysign(1.0, error)
        if (self.config.corner_supervision and self.cooldown == 0 and len(active) >= 3
                and abs(error) >= .18 and np.any((active >= 6) & (active <= 8))):
            self.mode, self.mode_time = "approach", 0.0
            self.approach_remaining = max(0, float(np.median(self.sensor_x[active]))-self.config.turn_lead_m)
            return self._command([self.config.approach_command]*2, dt)
        raw_derivative = (error-self.previous_error)/dt
        fraction = 1-math.exp(-dt/self.config.derivative_filter_s)
        self.derivative += fraction*(raw_derivative-self.derivative)
        self.previous_error = error
        forward = max(self.config.minimum_command, self.config.base/(1+self.config.slowdown_gain*abs(error)))
        turn = self.config.kp*error+self.config.ki*self.integral+self.config.kd*self.derivative
        # Conditional integration avoids accumulating a turn that the motors cannot deliver.
        if abs(turn) < 1-forward or error*turn < 0:
            self.integral = float(np.clip(self.integral+error*dt, -self.config.integral_limit, self.config.integral_limit))
        turn = self.config.kp*error+self.config.ki*self.integral+self.config.kd*self.derivative
        self.debug["pid_turn"] = turn
        action = self._command([forward+turn, forward-turn], dt)
        self.last_valid_action = action.copy()
        return action
