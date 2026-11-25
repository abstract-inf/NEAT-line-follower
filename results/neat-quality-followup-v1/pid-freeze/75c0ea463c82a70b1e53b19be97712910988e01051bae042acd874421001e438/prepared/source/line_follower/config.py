"""Explicit, serializable experiment parameters. All physical values use SI units."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import hashlib
import math
from pathlib import Path

# Body-frame origin height is shared by geometry construction and sensor sites.
BODY_ORIGIN_HEIGHT = 0.033


@dataclass(frozen=True)
class RobotConfig:
    """Nominal robot properties; these are assumptions, not identified hardware."""

    design: str = "competition"
    design_revision: str = "integrated-pcb-1"
    total_mass: float = 0.55
    wheel_spacing: float = 0.175
    wheel_radius: float = 0.0165
    wheel_width: float = 0.012
    chassis_length: float = 0.090
    chassis_width: float = 0.130
    chassis_height: float = 0.0025
    caster_offset: float = 0.045
    max_speed: float = 0.30
    max_torque: float = 0.03
    motor_time_constant: float = 0.050
    motor_gain: float = 0.005
    wheel_friction: float = 0.75
    caster_friction: float = 0.05
    physics_hz: int = 600
    control_hz: int = 60
    sensor_layout: str = "semicircle"
    sensor_radius: float = 0.110
    sensor_offset: float = 0.090
    sensor_angle: float = 1.70
    sensor_height: float = 0.006
    sensor_max_distance: float = 0.025

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"sensor_layout", "design", "design_revision"}:
                continue
            if field.name == "caster_offset":
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value == 0:
                    raise ValueError("caster_offset must be finite and nonzero")
                continue
            minimum = 1e-15
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"{field.name} must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if not isinstance(self.physics_hz, int) or not isinstance(self.control_hz, int):
            raise ValueError("physics_hz and control_hz must be integers")
        if self.sensor_layout not in {"semicircle", "xline"}:
            raise ValueError("sensor_layout must be semicircle or xline")
        if self.design not in {"competition", "paper"}:
            raise ValueError("design must be competition or paper")
        if self.design_revision != ("integrated-pcb-1" if self.design == "competition" else "paper-1"):
            raise ValueError("Unsupported robot design_revision; old geometries require explicit migration")
        if self.design == "competition" and self.sensor_layout != "semicircle":
            raise ValueError("The competition PCB uses the curved array; use the paper preset for xline")

    @classmethod
    def paper(cls) -> RobotConfig:
        """Earlier rectangular model and the original experiment's sensor geometry."""
        return cls(design="paper", design_revision="paper-1", chassis_length=0.150,
                   chassis_height=0.025, caster_offset=-0.075, sensor_radius=0.050,
                   sensor_offset=0.050, sensor_angle=1.57)

    @property
    def sensor_arc_center_x(self) -> float:
        """Arc translation: sensor_offset is axle to middle sensor, not arc center."""
        return self.sensor_offset - self.sensor_radius

    @property
    def dt(self) -> float:
        """Controller period in seconds."""
        return 1.0 / self.control_hz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None) -> RobotConfig:
        if path is None:
            return cls()
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "design_revision" not in values:
            raise ValueError("Robot JSON must declare design_revision; migrate older development configurations explicitly")
        return cls(**values)


@dataclass(frozen=True)
class EpisodeConfig:
    """Evaluator stopping rules and fitness weights."""

    minimum_time_limit: float = 45.0
    time_multiplier: float = 3.0
    stall_seconds: float = 5.0
    progress_increment: float = 0.010
    corridor_half_width: float = 0.045
    departure_seconds: float = 1.0
    gate_radius: float = 0.035
    # Optional continuation experiment: reward time saved only after completion.
    # Zero retains the original study's fitness exactly.
    completion_speed_bonus: float = 0.0
    maximum_time_limit: float | None = None  # None preserves the original unbounded formula.
    time_reference_speed: float | None = None  # Conservative route-average speed, m/s.
    command_jitter_penalty: float = 0.0
    fitness_version: str = "progress-1.2"

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for name, value in asdict(self).items()
               if name not in {"fitness_version", "completion_speed_bonus", "command_jitter_penalty",
                               "maximum_time_limit", "time_reference_speed"}):
            raise ValueError("Episode parameters must be finite and positive")
        if not math.isfinite(self.completion_speed_bonus) or not 0 <= self.completion_speed_bonus <= 0.5:
            raise ValueError("completion_speed_bonus must be finite and between 0 and 0.5")
        if not math.isfinite(self.command_jitter_penalty) or not 0 <= self.command_jitter_penalty <= 0.5:
            raise ValueError("command_jitter_penalty must be finite and between 0 and 0.5")
        if any(v is not None and (not math.isfinite(v) or v <= 0) for v in (self.maximum_time_limit, self.time_reference_speed)):
            raise ValueError("Optional time-limit parameters must be finite and positive")
        if self.maximum_time_limit is not None and self.maximum_time_limit < self.minimum_time_limit:
            raise ValueError("Maximum time limit is below minimum")
        if self.fitness_version not in {"progress-1.2", "route-speed-1.0"}:
            raise ValueError("Unknown fitness version")

    def time_limit(self, length: float, max_speed: float) -> float:
        allowance = length/self.time_reference_speed if self.time_reference_speed else self.time_multiplier*length/max_speed
        return min(self.maximum_time_limit or float("inf"), max(self.minimum_time_limit, allowance))


def geometry_record(robot: RobotConfig) -> dict:
    """Identify both parameters and generated MJCF, so code changes are detected."""
    from .simulation import robot_xml

    values = robot.to_dict()
    return {"robot": values,
            "configuration_sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
            "mjcf_sha256": hashlib.sha256(robot_xml(robot).encode()).hexdigest()}


def freeze_geometry(path: str | Path, robot: RobotConfig) -> Path:
    """Explicit user-invoked selection; never performed by a pilot or training."""
    from .simulation import Simulation

    Simulation(robot)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Preserve the existing geometry freeze; choose a new versioned filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"format_version": 1, "selected_for_training": True,
                                      **geometry_record(robot)}, indent=2), encoding="utf-8")
    return destination


def verify_geometry(path: str | Path, robot: RobotConfig) -> dict:
    """Keep main evolution behind the user's explicit geometry selection."""
    source = Path(path)
    if not source.exists():
        raise ValueError("Select geometry with freeze-robot before starting NEAT; see README.md")
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("format_version") != 1 or record.get("selected_for_training") is not True:
        raise ValueError("Not a supported, explicitly selected geometry freeze")
    if any(record.get(key) != value for key, value in geometry_record(robot).items()):
        raise ValueError("Robot configuration or generated MJCF differs from the selected geometry freeze")
    return record
