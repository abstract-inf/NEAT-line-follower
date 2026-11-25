"""Integrated PCB concept inspired by the blue competition-robot reference.

The visible board is one extruded, concave outline. Separate invisible convex
pieces give MuJoCo the same collision footprint without filling the openings
between the sensor wings and the main body. Coordinates use meters and the
axle-centered frame: +X forward, +Y left, +Z up.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .config import BODY_ORIGIN_HEIGHT, RobotConfig

PCB_RGBA = "0.025 0.26 0.48 1"
SILVER_RGBA = "0.65 0.68 0.70 1"
PCB_CENTER_HEIGHT = 0.009
SENSOR_BOARD_WIDTH = 0.016
NECK_WIDTH = 0.028
MOTOR_RADIUS = 0.006
MOTOR_HALF_LENGTH = 0.009
EFFECTIVE_BODY_HEIGHT = 0.025


def _cross(a, b, c) -> float:
    ab, ac = np.asarray(b) - a, np.asarray(c) - a
    return float(ab[0]*ac[1] - ab[1]*ac[0])


def _triangles(polygon: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clip a simple counterclockwise outline; preserve its concavities."""
    remaining = list(range(len(polygon)))
    faces = []
    while len(remaining) > 3:
        for position, middle in enumerate(remaining):
            first, last = remaining[position - 1], remaining[(position + 1) % len(remaining)]
            a, b, c = polygon[first], polygon[middle], polygon[last]
            if _cross(a, b, c) <= 1e-12:
                continue
            if any(min(_cross(a, b, polygon[i]), _cross(b, c, polygon[i]),
                       _cross(c, a, polygon[i])) >= -1e-12
                   for i in remaining if i not in (first, middle, last)):
                continue
            faces.append((first, middle, last))
            remaining.pop(position)
            break
        else:
            raise ValueError("PCB outline intersects itself; increase sensor_offset or reduce the chassis")
    faces.append(tuple(remaining))
    return faces


def _prism(asset, name: str, polygon: list[tuple[float, float]], thickness: float) -> None:
    count = len(polygon)
    vertices = [(x, y, z) for z in (-thickness / 2, thickness / 2) for x, y in polygon]
    faces = []
    for a, b, c in _triangles(polygon):
        faces.extend(((c, b, a), (a + count, b + count, c + count)))
    for first in range(count):
        second = (first + 1) % count
        faces.extend(((first, second, second + count), (first, second + count, first + count)))
    ET.SubElement(asset, "mesh", name=name,
                  vertex=" ".join(str(v) for point in vertices for v in point),
                  face=" ".join(str(v) for face in faces for v in face))


def _bezier(points: list[tuple[float, float]], count: int = 10) -> list[tuple[float, float]]:
    a, b, c, d = np.asarray(points)
    return [tuple((1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d)
            for t in np.linspace(0, 1, count)]


def board_geometry(config: RobotConfig) -> tuple[list, list, list]:
    """Return the unified perimeter, main-body slices, and curved-board slices."""
    length, width = config.chassis_length, config.chassis_width
    rear, shoulder, tip = -length / 3, length * 0.16, length * 2 / 3
    lower = _bezier([(rear, 0), (rear, -width/2), (-length*.20, -width/2), (-length*.05, -width/2)])
    lower += _bezier([lower[-1], (shoulder, -width/2),
                      (tip*.52, -NECK_WIDTH/2), (tip, -NECK_WIDTH/2)])[1:]
    inner, outer = config.sensor_radius - SENSOR_BOARD_WIDTH/2, config.sensor_radius + SENSOR_BOARD_WIDTH/2
    if inner <= NECK_WIDTH/2:
        raise ValueError("Sensor radius is too small for the PCB neck")
    join = math.asin(NECK_WIDTH / (2 * inner))
    half_angle = config.sensor_angle / 2
    if not join < half_angle < math.pi / 2:
        raise ValueError("Competition PCB sensor_angle must fit a forward-facing curved bar")
    center = config.sensor_arc_center_x

    def point(radius, angle):
        return (center + radius * math.cos(angle), radius * math.sin(angle))

    junction = point(inner, -join)
    if junction[0] - tip < 0.005:
        raise ValueError("Sensor offset leaves less than 5 mm of exposed rigid neck; increase sensor_offset")
    right_inner = [point(inner, angle) for angle in np.linspace(-join, -half_angle, 16)]
    outer_edge = [point(outer, angle) for angle in np.linspace(-half_angle, half_angle, 49)]
    left_inner = [point(inner, angle) for angle in np.linspace(half_angle, join, 16)]
    outline = lower + right_inner + outer_edge + left_inner + [(x, -y) for x, y in reversed(lower[1:])]
    body_pieces = []
    for first, second in zip(lower[:-1], lower[1:]):
        polygon = [first, second, (second[0], -second[1]), (first[0], -first[1])]
        if first[1] == 0:
            polygon.pop()
        body_pieces.append(polygon)
    angles = sorted(set(np.linspace(-half_angle, half_angle, 49).tolist() + [-join, join]))
    arc_pieces = [[point(inner, a), point(outer, a), point(outer, b), point(inner, b)]
                  for a, b in zip(angles[:-1], angles[1:])]
    return outline, body_pieces, arc_pieces


def _geom(body, name: str, kind: str, **attributes) -> None:
    """Rigid assembly parts use the body's explicit aggregate mass/inertia."""
    ET.SubElement(body, "geom", name=name, type=kind, mass="0", **attributes)


def _visual(body, name: str, kind: str, **attributes) -> None:
    _geom(body, name, kind, contype="0", conaffinity="0", **attributes)


def _add_drive_mount(body, config: RobotConfig, side: str, sign: int) -> None:
    """Connect a coaxial gearmotor, rigid U bracket, shaft, and rotating wheel."""
    axle_z = config.wheel_radius - BODY_ORIGIN_HEIGHT
    board_top = PCB_CENTER_HEIGHT + config.chassis_height/2 - BODY_ORIGIN_HEIGHT
    motor_y = config.wheel_spacing/2 - config.wheel_width/2 - 0.029
    gearbox_y = motor_y + MOTOR_HALF_LENGTH + 0.008
    if motor_y - MOTOR_HALF_LENGTH < 0 or motor_y > config.chassis_width/2:
        raise ValueError("Wheel spacing does not leave room for the motor mounts on the chassis")
    if config.wheel_radius - MOTOR_RADIUS < PCB_CENTER_HEIGHT + config.chassis_height/2:
        raise ValueError("Motor intersects the PCB; increase wheel radius or lower the board")
    _geom(body, f"{side}_motor_housing", "cylinder", size=f"{MOTOR_RADIUS} {MOTOR_HALF_LENGTH}",
          quat="0.7071067812 0.7071067812 0 0", pos=f"0 {sign*motor_y} {axle_z}", rgba="0.24 0.27 0.29 1")
    _geom(body, f"{side}_gearbox", "box", size="0.006 0.008 0.006",
          pos=f"0 {sign*gearbox_y} {axle_z}", rgba="0.76 0.73 0.61 1")
    _geom(body, f"{side}_bracket_base", "box", size="0.011 0.014 0.00075",
          pos=f"0 {sign*(motor_y+.005)} {board_top+.00075}", rgba=SILVER_RGBA)
    for edge in (-1, 1):
        _geom(body, f"{side}_bracket_wall_{edge}", "box", size="0.001 0.009 0.0045",
              pos=f"{edge*.007} {sign*motor_y} {axle_z-.001}", rgba=SILVER_RGBA)
        _visual(body, f"{side}_mount_screw_{edge}", "cylinder", size="0.0015 0.0006",
                pos=f"{edge*.009} {sign*motor_y} {board_top+.0021}", rgba="0.14 0.15 0.16 1")
    wheel = body.find(f"body[@name='{side}_wheel']")
    wheel.find("geom").set("rgba", "0.075 0.08 0.09 1")
    shaft_start = gearbox_y + .007
    shaft_end = config.wheel_spacing / 2
    _visual(wheel, f"{side}_axle_shaft", "cylinder", size=f"0.0015 {(shaft_end-shaft_start)/2}",
            pos=f"0 {-sign*(shaft_end-shaft_start)/2} 0", quat="0.7071067812 0.7071067812 0 0", rgba=SILVER_RGBA)
    _visual(wheel, f"{side}_hub", "cylinder", size="0.007 0.0006",
            quat="0.7071067812 0.7071067812 0 0", pos=f"0 {sign*(config.wheel_width/2+.0005)} 0", rgba=SILVER_RGBA)


def add_competition_body(root, config: RobotConfig, sensors: np.ndarray) -> None:
    """Build a continuous board, fixed gearmotor mounts, and supported ball caster.

    The 490 g fixed assembly is a nominal lumped robot mass, not PCB material
    density. It includes unmodeled electronics/power hardware; there is no LCD
    or battery replica. Moving the optical strip holds that approximation fixed.
    """
    body = root.find(".//body[@name='chassis']")
    for name in ("chassis_geom", "sensor_board", "heading"):
        body.remove(body.find(f"geom[@name='{name}']"))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    body_mass = config.total_mass * .49/.55
    length, width, height = config.chassis_length, config.chassis_width, EFFECTIVE_BODY_HEIGHT
    inertia = body_mass / 12 * np.array([width*width + height*height,
                                        length*length + height*height, length*length + width*width])
    com_x = length*.20
    if not 0 < com_x < config.caster_offset:
        raise ValueError("Fixed-body center of mass must lie between the rear axle and front caster")
    ET.SubElement(body, "inertial", pos=f"{com_x} 0 {0.021-BODY_ORIGIN_HEIGHT}", mass=str(body_mass),
                  diaginertia=" ".join(map(str, inertia)))
    outline, body_pieces, arc_pieces = board_geometry(config)
    board_z = PCB_CENTER_HEIGHT - BODY_ORIGIN_HEIGHT
    _prism(asset, "integrated_pcb", outline, config.chassis_height)
    _visual(body, "chassis_geom", "mesh", mesh="integrated_pcb", pos=f"0 0 {board_z}", rgba=PCB_RGBA)
    for index, polygon in enumerate(body_pieces + arc_pieces):
        name = f"pcb_collision_{index:02d}" if index < len(body_pieces) else f"sensor_arc_{index-len(body_pieces):02d}"
        _prism(asset, name, polygon, config.chassis_height)
        _geom(body, name, "mesh", mesh=name, pos=f"0 0 {board_z}", rgba="0 0 0 0", group="3")
    neck_start, neck_end = length*2/3, config.sensor_offset - SENSOR_BOARD_WIDTH/2 + .0002
    _geom(body, "sensor_board", "box", size=f"{(neck_end-neck_start)/2} {NECK_WIDTH/2} {config.chassis_height/2}",
          pos=f"{(neck_start+neck_end)/2} 0 {board_z}", rgba="0 0 0 0", group="3")

    for side, sign in (("left", 1), ("right", -1)):
        _add_drive_mount(body, config, side, sign)
    # The flanged cup is fixed to the board; only its captured ball rotates.
    _geom(body, "caster_cup", "cylinder", size="0.0085 0.005",
          pos=f"{config.caster_offset} 0 {0.011-BODY_ORIGIN_HEIGHT}", rgba="0.24 0.27 0.29 1")
    _geom(body, "caster_flange", "cylinder", size="0.011 0.0006",
          pos=f"{config.caster_offset} 0 {PCB_CENTER_HEIGHT+config.chassis_height/2+.0006-BODY_ORIGIN_HEIGHT}", rgba=SILVER_RGBA)

    for index, (x, y, z) in enumerate(sensors):
        _visual(body, f"reflectance_package_{index}", "box", size="0.0022 0.0025 0.001",
                pos=f"{x} {y} {z+.001}", rgba="0.035 0.035 0.04 1")
        for sign in (-1, 1):
            _visual(body, f"sensor_pad_{index}_{sign}", "cylinder", size="0.0012 0.00015",
                    pos=f"{x+sign*.0035} {y} {board_z+config.chassis_height/2+.00015}", rgba=SILVER_RGBA)
    for index, end_y in enumerate((-.004, .004)):
        _visual(body, f"forward_arrow_{index}", "capsule", size="0.0005",
                fromto=f"0.022 {end_y} {board_z+.0016} 0.031 0 {board_z+.0016}", rgba="0.9 0.82 0.56 1")
