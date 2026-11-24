"""MuJoCo dynamics and pose-dependent reflectance sensors, without viewer imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import io
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import BODY_ORIGIN_HEIGHT, RobotConfig
from .tracks import SENSOR_COUNT, TrackMask, TrackSpec

CASTER_RADIUS = 0.006
SETTLE_SECONDS = 0.25
MINIMUM_MOUNT_WHEEL_CLEARANCE = 0.002


def sample_sensor_rays(origins: np.ndarray, rotations: np.ndarray, mask: TrackMask,
                       max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    """Intersect sensor rays with the floor; support one robot or whole batches.

    Leading dimensions are preserved. Float64 matches MuJoCo CPU observations,
    including when the source poses came from float32 CUDA state transfers.
    """
    origins = np.asarray(origins, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    directions = rotations[..., :, 2]
    downward = directions[..., 2] < -1e-8
    distances = np.divide(-origins[..., 2], directions[..., 2],
                          out=np.full(downward.shape, np.inf), where=downward)
    valid = downward & (distances >= 0) & (distances <= max_distance)
    hits = np.full(origins.shape, np.nan)
    hits[valid] = origins[valid] + distances[valid, None] * directions[valid]
    readings = np.zeros(valid.shape)
    readings[valid] = mask.sample(hits[valid, :2])
    return readings, hits


def sensor_locations(config: RobotConfig) -> np.ndarray:
    """Body-frame sensor sites (+X forward, +Y left), ordered left to right.

    The screen-coordinate original has +Y downward. Negating that coordinate
    preserves sensor indices while converting to a right-handed world frame.
    The historical XLINE 64 mm centering offset is intentionally preserved.
    """
    if config.sensor_layout == "semicircle":
        angles = np.linspace(-config.sensor_angle / 2, config.sensor_angle / 2, SENSOR_COUNT)
        xy = np.column_stack((config.sensor_arc_center_x + config.sensor_radius * np.cos(angles),
                              -config.sensor_radius * np.sin(angles)))
    else:
        index = np.arange(SENSOR_COUNT)
        offset = np.where(index < 4, index * 0.005, np.where(index < 10, 0.020, (14 - index) * 0.005))
        xy = np.column_stack((offset + config.sensor_offset, index * 0.009 - 0.064))
    return np.column_stack((xy, np.full(SENSOR_COUNT, config.sensor_height - BODY_ORIGIN_HEIGHT)))


def robot_xml(config: RobotConfig) -> str:
    """Parameterize the readable MJCF template; all masses sum to total_mass."""
    root = ET.fromstring(files("line_follower").joinpath("assets/robot.xml").read_text())
    root.find("option").set("timestep", str(1 / config.physics_hz))
    ET.SubElement(root.find("option"), "flag", autoreset="disable")
    chassis = root.find(".//body[@name='chassis']")
    body = root.find(".//geom[@name='chassis_geom']")
    body.set("size", f"{config.chassis_length / 2} {config.chassis_width / 2} {config.chassis_height / 2}")
    scale = config.total_mass / 0.55
    for geom in chassis.iter("geom"):
        if "mass" in geom.attrib:
            geom.set("mass", str(float(geom.get("mass")) * scale))
    for side, sign in (("left", 1), ("right", -1)):
        wheel = root.find(f".//body[@name='{side}_wheel']")
        wheel.set("pos", f"0 {sign * config.wheel_spacing / 2} {config.wheel_radius - BODY_ORIGIN_HEIGHT}")
        wheel.find("geom").set("size", f"{config.wheel_radius} {config.wheel_width / 2}")
    root.find(".//body[@name='caster']").set("pos", f"{config.caster_offset} 0 {CASTER_RADIUS-BODY_ORIGIN_HEIGHT}")
    for index, xyz in enumerate(sensor_locations(config)):
        ET.SubElement(chassis, "site", name=f"line_sensor_{index}", pos=" ".join(map(str, xyz)),
                      size="0.0015", rgba="0.4 0.4 0.4 1", quat="0 1 0 0")
    for pair in root.find("contact"):
        coefficient = config.caster_friction if pair.get("name") == "caster_floor" else config.wheel_friction
        pair.set("friction", f"{coefficient} {coefficient} 0.0001 0.00001 0.00001")
    for actuator in root.find("actuator"):
        actuator.set("dynprm", str(config.motor_time_constant))
        actuator.set("gainprm", str(config.motor_gain))
        actuator.set("biasprm", f"0 0 {-config.motor_gain}")
        angular_limit = config.max_speed / config.wheel_radius
        actuator.set("ctrlrange", f"{-angular_limit} {angular_limit}")
        actuator.set("forcerange", f"{-config.max_torque} {config.max_torque}")
    if config.design == "competition":
        from .robot_design import add_competition_body
        add_competition_body(root, config, sensor_locations(config))
    return ET.tostring(root, encoding="unicode")


@dataclass(frozen=True)
class RobotState:
    """SI diagnostics for evaluation; never part of the controller observation."""

    time: float
    position: np.ndarray
    yaw: float
    forward_speed: float
    yaw_rate: float
    wheel_speeds: np.ndarray
    torques: np.ndarray


class Simulation:
    """One independent physical robot. Headless stepping never creates OpenGL."""

    def __init__(self, config: RobotConfig | None = None, *, model: mujoco.MjModel | None = None):
        self.config = config or RobotConfig()
        self.model = model if model is not None else mujoco.MjModel.from_xml_string(robot_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.chassis_id = self.model.body("chassis").id
        self.wheel_dofs = [self.model.joint(f"{side}_joint").dofadr[0] for side in ("left", "right")]
        self.site_ids = [self.model.site(f"line_sensor_{i}").id for i in range(SENSOR_COUNT)]
        self.mask: TrackMask | None = None
        self.track: TrackSpec | None = None
        self.observation = np.zeros(SENSOR_COUNT)
        self.sensor_hits = np.full((SENSOR_COUNT, 3), np.nan)
        self.last_action = np.zeros(2)
        self._start_time = 0.0
        self.validate_sensor_mount()

    def validate_sensor_mount(self) -> None:
        """Reject wheel overlap instead of silently changing the requested offset."""
        if self.config.design != "competition":
            return
        if np.min(sensor_locations(self.config)[:, 0]) <= 0:
            raise ValueError("Sensor offset puts part of the semicircle behind the wheel axle")
        mujoco.mj_forward(self.model, self.data)
        gap = self.mount_wheel_clearance()
        if gap < MINIMUM_MOUNT_WHEEL_CLEARANCE:
            raise ValueError(f"Sensor mount wheel clearance is {gap*1000:.2f} mm; need at least 2 mm. Increase sensor_offset")

    def board_geom_ids(self) -> list[int]:
        """Rigid mounting surfaces, excluding visual pads and sensor packages."""
        return [index for index in range(self.model.ngeom)
                if self.model.geom(index).name == "sensor_board"
                or (self.config.design == "paper" and self.model.geom(index).name == "chassis_geom")
                or (self.model.geom(index).name or "").startswith(("sensor_arc_", "pcb_collision_"))]

    def mount_wheel_clearance(self) -> float:
        """Smallest signed separation between the mounting PCB and either tire."""
        boards = [index for index in self.board_geom_ids()
                  if self.model.geom(index).name == "sensor_board"
                  or (self.model.geom(index).name or "").startswith("sensor_arc_")]
        return min(mujoco.mj_geomDistance(self.model, self.data, board, self.model.geom(tire).id, 1.0, None)
                   for board in boards for tire in ("left_tire", "right_tire"))

    def board_ground_clearance(self) -> float:
        """Minimum actual board-to-floor separation, including pitch and roll."""
        floor = self.model.geom("floor").id
        return min(mujoco.mj_geomDistance(self.model, self.data, board, floor, 1.0, None)
                   for board in self.board_geom_ids())

    def reset(self, track: TrackSpec, mask: TrackMask | None = None) -> np.ndarray:
        """Reset all dynamics and settle under gravity before starting the clock."""
        self.track = track
        self.mask = mask or TrackMask(track)
        mujoco.mj_resetData(self.model, self.data)
        x, y, yaw = track.start
        self.data.qpos[:3] = [x, y, BODY_ORIGIN_HEIGHT + 0.0002]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.ctrl[:] = 0
        for _ in range(round(SETTLE_SECONDS * self.config.physics_hz)):
            mujoco.mj_step(self.model, self.data)
        # Settling is initial-condition preparation, not episode time or reward.
        self.data.qvel[:] = 0
        self.data.act[:] = 0
        self.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._start_time = float(self.data.time)
        self.last_action[:] = 0
        self.observation = self.read_sensors()
        return self.observation.copy()

    def step(self, action: np.ndarray | tuple[float, float] | list[float]) -> np.ndarray:
        """Hold a bounded wheel-speed target for one control period."""
        if self.mask is None:
            raise RuntimeError("reset(track) must be called before step")
        commands = np.asarray(action, dtype=float)
        if commands.shape != (2,) or not np.all(np.isfinite(commands)):
            raise ValueError("Motor action must contain two finite values")
        self.last_action = np.clip(commands, -1.0, 1.0)
        self.data.ctrl[:] = self.last_action * self.config.max_speed / self.config.wheel_radius
        mujoco.mj_step(self.model, self.data, nstep=self.config.physics_hz // self.config.control_hz)
        # mj_step leaves position-dependent derived arrays at its last substep.
        mujoco.mj_forward(self.model, self.data)
        self.observation = self.read_sensors()
        return self.observation.copy()

    def read_sensors(self) -> np.ndarray:
        """Ray/floor intersection accounts for full chassis roll, pitch, and yaw."""
        if self.mask is None:
            raise RuntimeError("No track loaded")
        origins = self.data.site_xpos[self.site_ids]
        rotations = self.data.site_xmat[self.site_ids].reshape(-1, 3, 3)
        readings, hits = sample_sensor_rays(origins, rotations, self.mask, self.config.sensor_max_distance)
        self.sensor_hits[:] = hits
        return readings

    def enable_rendering(self) -> None:
        """Allocate visual assets only on request; headless workers stay GPU-free.

        A finite UV-mapped quad draws the floor image above a collision plane.
        Explicit UVs avoid primitive texture repetition/scaling ambiguities.
        Calling this rebuilds the model and resets the currently loaded track.
        """
        if self.mask is None or self.track is None:
            raise RuntimeError("Load a track before enabling rendering")
        root = ET.fromstring(robot_xml(self.config))
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="track_texture", type="2d", file="floor.png")
        ET.SubElement(asset, "material", name="track_material", texture="track_texture", specular="0", shininess="0", reflectance="0")
        half = self.track.arena_size / 2
        vertices = f"{-half} {-half} 0 {half} {-half} 0 {half} {half} 0 {-half} {half} 0"
        ET.SubElement(asset, "mesh", name="track_quad", inertia="shell", vertex=vertices, face="0 1 2 0 2 3",
                      texcoord="0 1 1 1 1 0 0 0")
        ET.SubElement(root.find("worldbody"), "geom", name="track_visual", type="mesh", mesh="track_quad",
                      material="track_material", pos="0 0 0.00008", contype="0", conaffinity="0", group="2")
        image_bytes = io.BytesIO()
        self.mask.image.convert("RGB").save(image_bytes, format="PNG")
        with mujoco.MjVfs() as vfs:
            vfs["floor.png"] = image_bytes.getvalue()
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), vfs=vfs)
        self.data = mujoco.MjData(self.model)
        self.chassis_id = self.model.body("chassis").id
        self.wheel_dofs = [self.model.joint(f"{side}_joint").dofadr[0] for side in ("left", "right")]
        self.site_ids = [self.model.site(f"line_sensor_{i}").id for i in range(SENSOR_COUNT)]
        self.reset(self.track, self.mask)

    def update_visual_track(self, viewer=None) -> None:
        """Upload a same-sized mask after reset without changing robot dynamics."""
        if not self.model.ntex:
            raise RuntimeError("enable_rendering() must be called first")
        texture = self.model.texture("track_texture")
        if texture.width[0] != self.mask.pixels or texture.height[0] != self.mask.pixels:
            raise ValueError("A new canvas size requires rebuilding the viewer")
        # Keep PNG row order. The quad UVs map its top image row to world +Y.
        rgb = np.repeat(np.asarray(self.mask.image)[:, :, None], 3, axis=2)
        self.model.tex_data[:] = rgb.reshape(-1)
        if viewer is not None:
            viewer.update_texture(texture.id)

    @property
    def state(self) -> RobotState:
        rotation = self.data.xmat[self.chassis_id].reshape(3, 3)
        velocity = np.empty(6)
        # MuJoCo's local BODY velocity uses the inertial principal axes. Project
        # world velocity onto chassis axes explicitly, not the inertia frame.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.chassis_id, velocity, 0)
        local_linear = rotation.T @ velocity[3:]
        local_angular = rotation.T @ velocity[:3]
        return RobotState(float(self.data.time - self._start_time), self.data.xpos[self.chassis_id].copy(),
                          math.atan2(rotation[1, 0], rotation[0, 0]), float(local_linear[0]), float(local_angular[2]),
                          self.data.qvel[self.wheel_dofs].copy() * self.config.wheel_radius,
                          self.data.actuator_force.copy())
