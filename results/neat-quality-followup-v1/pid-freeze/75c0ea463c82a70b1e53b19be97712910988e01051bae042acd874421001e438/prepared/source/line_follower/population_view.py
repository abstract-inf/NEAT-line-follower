"""Read-only, throttled observation of independent evaluation worlds.

Snapshots never supply controller inputs or change the physics clock. Each CPU
process owns its files; CUDA publishes its entire active batch in one file.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
import time

import mujoco
import numpy as np

SNAPSHOT_HZ = 10.0
POPULATION_DISPLAY_HZ = 5.0  # UI updates are independent of the worker's observation/control clocks.
STALE_EXPORT_SECONDS = 5.0  # Label old observations; never animate them as fresh training.


def atomic_snapshot(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    try:
        temporary.replace(path)
    except PermissionError:
        # Windows can briefly retain a reader handle. Drop this display update;
        # scientific evaluation must never wait on rendering or lose a result.
        pass


def read_snapshot(path: Path) -> dict | None:
    """Generation cleanup and atomic Windows replacement can race an observer."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None


class SnapshotPublisher:
    """Keep rendering optional and bounded, including during very fast episodes."""

    def __init__(self, context: dict | None, channel: str):
        self.context = context
        self.channel = channel
        self.next_update = 0.0

    @property
    def due(self) -> bool:
        return self.context is not None and time.perf_counter() >= self.next_update

    def publish(self, robots: list[dict], force: bool = False) -> None:
        if self.context is None:
            return
        now = time.perf_counter()
        if force or now >= self.next_update:
            atomic_snapshot(Path(self.context["directory"]) / f"{self.channel}.json",
                            {"token": self.context["token"], "robots": robots})
            self.next_update = now + 1 / SNAPSHOT_HZ


def robot_snapshot(genome: int, track: int, episode) -> dict:
    return {"genome": genome, "track": track, "qpos": episode.simulation.data.qpos.tolist(),
            "time": episode.steps * episode.simulation.config.dt,
            "status": episode.termination or "running"}


class PopulationScene:
    """Instance the real robot meshes in MuJoCo without coupled robot contacts.

    All individuals share the selected track visually, exactly as independent
    training worlds do. Ghost overlap cannot influence sensing or fitness.
    """

    def __init__(self, simulation):
        # Display-only groups/colors must never mutate the viewer/physics model.
        # Mesh/material IDs remain identical, so the native renderer reuses assets.
        self.model = deepcopy(simulation.model)
        self.data = mujoco.MjData(self.model)
        self.source = mujoco.MjvScene(self.model, maxgeom=1000)
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        self.camera = mujoco.MjvCamera()
        self.chassis_geom = self.model.geom("chassis_geom").id
        self.visible = (self.model.geom_bodyid != 0) & (self.model.geom_rgba[:, 3] > 0)
        self.original_rgba = self.model.geom_rgba.copy()
        self.model.geom_group[self.visible] = 4
        essential = ["chassis_geom", "left_tire", "right_tire", "caster_ball"]
        essential += [f"reflectance_package_{index}" for index in range(15)] if simulation.config.design == "competition" else ["sensor_board"]
        for name in essential:
            self.model.geom(name).group = 5
        self.option.geomgroup[5] = 1
        self.perturb = mujoco.MjvPerturb()
        self.tile_size = np.full(2, simulation.track.arena_size + .1)

    def offset(self, index: int, count: int, grid: bool) -> np.ndarray:
        if not grid:
            return np.zeros(3)
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        return np.r_[((index % columns)-(columns-1)/2)*self.tile_size[0],
                     ((index // columns)-(rows-1)/2)*self.tile_size[1], 0]

    def draw(self, scene, poses: list[tuple[list, str]], grid: bool = False, selected: int = 0) -> None:
        """Append whole robots in native code instead of copying 17,000 geoms in Python.

        Distant individuals retain the PCB, wheels, caster and 15 apertures.
        Only the selected robot needs small motor mounts, solder pads and screws.
        Opaque colors avoid sorting thousands of transparent surfaces each frame.
        """
        scene.ngeom = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        for index, (pose, status) in enumerate(poses):
            translation = self.offset(index, len(poses), grid)
            self.data.qpos[:] = pose
            self.data.qpos[:3] += translation
            mujoco.mj_kinematics(self.model, self.data)
            self.option.geomgroup[4] = index == selected
            if scene.ngeom + self.model.ngeom > scene.maxgeom:
                raise ValueError("Population exceeds the MuJoCo viewer geometry capacity")
            self.model.geom_rgba[:] = self.original_rgba
            self.model.geom_rgba[self.visible, 3] = 1
            hue = (index * .61803398875) % 1
            self.model.geom_rgba[self.chassis_geom, :3] = [.25+.6*hue, .75-.5*hue, .8]
            if status != "running":
                self.model.geom_rgba[self.visible, :3] *= .6 if status == "waiting" else .8
            mujoco.mjv_addGeoms(self.model, self.data, self.option, self.perturb,
                                mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        offset = scene.ngeom
        # In grid mode use the same finite textured floor mesh for each world.
        if grid:
            floor = self.model.geom("track_visual").id
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera,
                                   mujoco.mjtCatBit.mjCAT_STATIC, self.source)
            source = next(self.source.geoms[i] for i in range(self.source.ngeom)
                          if self.source.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
                          and self.source.geoms[i].objid == floor)
            for index in range(len(poses)):
                if offset >= scene.maxgeom:
                    raise ValueError("Population floors exceed the MuJoCo viewer geometry capacity")
                geom = scene.geoms[offset]
                mujoco.mjv_initGeom(geom, source.type, source.size,
                                   source.pos+self.offset(index, len(poses), grid), source.mat.ravel(), source.rgba)
                for name in ("dataid", "matid", "texid", "texcoord", "texuniform", "emission", "specular", "shininess", "reflectance"):
                    setattr(geom, name, getattr(source, name))
                geom.texrepeat[:] = source.texrepeat
                offset += 1
        scene.ngeom = offset
