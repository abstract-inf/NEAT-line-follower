"""Sensor-only controllers and explicit, versioned model loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import pickle
from pathlib import Path
from typing import Protocol

import neat
import numpy as np

from .tracks import SENSOR_COUNT

MODEL_FORMAT_VERSION = 1
NEAT_VERSION = "0.92"
TEMPORAL_INPUT_COUNT = SENSOR_COUNT + 6
_SENSOR_OFFSETS = tuple(range(-7, 8))


class Controller(Protocol):
    """No route, pose, progress, difficulty label, or future line is observable."""

    def reset(self) -> None: ...
    def act(self, observation: np.ndarray, dt: float) -> np.ndarray: ...


@dataclass(frozen=True)
class PIDConfig:
    """Original gains and base command; tuning must produce a new saved config."""

    kp: float = 2.0
    ki: float = 0.0001
    kd: float = 0.5
    base: float = 1.0
    sensor_layout: str = "semicircle"


class PIDController:
    """Port of the implemented positional PID, with a documented frame adapter.

    The old screen's +Y points down. Its motor mixing therefore has the opposite
    yaw sign from physical left/right wheels in a right-handed MuJoCo world.
    We preserve gains, weighted error, derivative, integral, and original
    deadband order, and correct the mixing sign at this boundary. XLINE's
    historical indexing runs in the opposite direction to the semicircle.
    """

    def __init__(self, config: PIDConfig | None = None):
        self.config = config or PIDConfig()
        if self.config.sensor_layout not in {"semicircle", "xline"}:
            raise ValueError("Unknown PID sensor layout")
        if not all(np.isfinite(v) for v in (self.config.kp, self.config.ki, self.config.kd, self.config.base)):
            raise ValueError("PID gains and base must be finite")
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def act(self, observation: np.ndarray, dt: float) -> np.ndarray:
        readings = np.asarray(observation)
        if readings.shape != (SENSOR_COUNT,) or dt <= 0:
            raise ValueError("PID needs 15 readings and a positive timestep")
        active = readings > 0
        if np.any(active):
            error = float(np.mean(np.linspace(-7, 7, SENSOR_COUNT)[active]))
            if self.config.sensor_layout == "xline":
                error = -error
        else:
            error = 7.0 if self.previous_error > 0 else -7.0
        proportional = self.config.kp * error
        self.integral += self.config.ki * error * dt
        derivative = self.config.kd * (error - self.previous_error) / dt
        self.previous_error = error
        # The source clears integral after computing P and D; preserve that order.
        if abs(error) <= 1.0:
            self.integral = 0.0
        correction = proportional + self.integral + derivative
        return np.clip([self.config.base + correction, self.config.base - correction], -1, 1)


class NEATController:
    """Load direct-wheel policies and the versioned sensor-history experiment."""

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config, swap_legacy_motors: bool = False):
        inputs = config.genome_config.num_inputs
        if inputs not in {SENSOR_COUNT, TEMPORAL_INPUT_COUNT} or config.genome_config.num_outputs != 2:
            raise ValueError("NEAT needs 15 direct or 21 sensor-history inputs and 2 outputs")
        if not config.genome_config.feed_forward:
            raise ValueError("This experiment requires a feedforward NEAT network")
        self.temporal = inputs == TEMPORAL_INPUT_COUNT
        if self.temporal and swap_legacy_motors:
            raise ValueError("Legacy motor swapping is only supported by direct-wheel models")
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.swap_legacy_motors = swap_legacy_motors
        self.reset()

    def reset(self) -> None:
        self.last_center = self.edge_memory = self.loss_age = 0.0
        self.previous_forward = self.previous_turn = 0.0

    def act(self, observation: np.ndarray, dt: float) -> np.ndarray:
        if np.asarray(observation).shape != (SENSOR_COUNT,):
            raise ValueError("NEAT needs exactly 15 sensor readings")
        if self.temporal:
            return self._temporal_action(observation.tolist(), dt)
        action = np.asarray(self.network.activate(observation.tolist()), dtype=float)
        if not np.all(np.isfinite(action)):
            raise ValueError("NEAT produced non-finite motor commands")
        # Do not rescale sigmoid outputs: preserve the original activation semantics.
        return np.clip(action[::-1] if self.swap_legacy_motors else action, -1, 1)

    def _temporal_action(self, readings: list[float], dt: float) -> np.ndarray:
        """Reflect the sensor frame to share steering experience across turn directions."""
        if dt <= 0:
            raise ValueError("Sensor history requires a positive timestep")
        hits = sum(readings)
        center = sum(offset * hit for offset, hit in zip(_SENSOR_OFFSETS, readings)) / (7 * hits) if hits else 0.0
        self.loss_age = 0.0 if hits else min(1.0, self.loss_age + dt / .50)
        self.edge_memory *= max(0.0, 1 - dt / .25)
        if abs(center) >= 4 / 7:
            self.edge_memory = center

        # The sign also reflects previous actions when the current frame changes sides.
        side = center or self.last_center or self.edge_memory or self.previous_turn
        if not side:
            side = next((right - left for left, right in zip(readings[:7], readings[:7:-1]) if left != right), 0.0)
        reflection = -1 if side < 0 else 1
        canonical = readings[::-1] if reflection < 0 else readings
        features = canonical + [center * reflection, self.last_center * reflection, self.edge_memory * reflection,
                                self.loss_age, self.previous_forward, self.previous_turn * reflection]
        drive, steering = self.network.activate(features)
        if not np.isfinite(drive) or not np.isfinite(steering):
            raise ValueError("NEAT produced non-finite motor commands")
        forward = .35 + .65 * drive
        turn = steering * reflection if side else 0.0
        left = min(1.0, max(-1.0, forward + turn))
        right = min(1.0, max(-1.0, forward - turn))
        self.previous_forward, self.previous_turn = (left + right) / 2, (left - right) / 2
        if hits:
            self.last_center = center
        return np.array((left, right))


def load_neat_config(path: str | Path) -> neat.Config:
    if importlib.metadata.version("neat-python") != NEAT_VERSION:
        raise RuntimeError("Use neat-python==0.92; checkpoint formats are version-specific")
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def save_model(path: str | Path, genome: neat.DefaultGenome, config: neat.Config, metadata: dict) -> None:
    """Save a native 0.92 model with its exact config and experiment provenance."""
    if config.genome_config.num_inputs == TEMPORAL_INPUT_COUNT:
        metadata = {**metadata, "controller_semantics": "mirrored-sensor-history-v1"}
    artifact = {"format_version": MODEL_FORMAT_VERSION, "neat_version": NEAT_VERSION,
                "genome": genome, "config": config, "metadata": metadata}
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def read_model_artifact(path: str | Path) -> dict:
    """Load a trusted project artifact; reject bare legacy pickles explicitly."""
    with Path(path).open("rb") as stream:
        artifact = pickle.load(stream)
    if not isinstance(artifact, dict) or artifact.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Not a project model. Use import-legacy with the original NEAT config")
    if artifact.get("neat_version") != importlib.metadata.version("neat-python"):
        raise ValueError("Saved model's NEAT version differs from the installed version")
    return artifact


def load_model(path: str | Path) -> tuple[NEATController, dict]:
    artifact = read_model_artifact(path)
    metadata = artifact["metadata"]
    return NEATController(artifact["genome"], artifact["config"], metadata.get("swap_legacy_motors", False)), metadata
