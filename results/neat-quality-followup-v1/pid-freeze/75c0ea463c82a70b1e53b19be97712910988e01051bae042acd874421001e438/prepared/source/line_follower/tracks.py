"""Seeded competition-style tracks, analytic primitives, and shared floor masks.

The ordered route is evaluator-only data. Gaps and distractors change reflectance,
not the intended route. Distances are meters and signed angles are radians.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GENERATOR_VERSION = "1.0"
SENSOR_COUNT = 15
ROUTE_SAMPLE_SPACING = 0.002
DIFFICULTIES = ("beginner", "intermediate", "advanced")


class TrackGenerationError(ValueError):
    """A seed exhausted bounded geometry search; configuration errors are separate."""


@dataclass(frozen=True)
class Section:
    """An exact straight, signed circular arc, or zero-length sharp corner."""

    kind: str
    length: float = 0.0
    angle: float = 0.0
    radius: float = 0.0
    width: float = 0.020
    feature: str = "straight"

    def __post_init__(self) -> None:
        if self.kind not in {"straight", "arc", "corner"}:
            raise ValueError(f"Unknown primitive: {self.kind}")
        if not 0.005 <= self.width <= 0.060:
            raise ValueError("Line width must be between 5 and 60 mm")
        if self.kind == "straight" and self.length <= 0:
            raise ValueError("A straight must have positive length")
        if self.kind == "arc" and (self.radius <= 0 or self.angle == 0):
            raise ValueError("An arc needs a positive radius and nonzero angle")
        if self.kind == "corner" and self.angle == 0:
            raise ValueError("A corner needs a nonzero angle")
        if not all(math.isfinite(v) for v in (self.length, self.angle, self.radius, self.width)):
            raise ValueError("Primitive parameters must be finite")


@dataclass(frozen=True)
class Branch:
    """A straight distractor attached at route arclength ``at``.

    ``crossing`` extends the line to both sides. The intended route always goes
    straight through the junction; no hidden turn instruction is introduced.
    """

    at: float
    length: float = 0.14
    angle: float = math.pi / 2
    width: float = 0.020
    crossing: bool = False


@dataclass(frozen=True)
class Route:
    """Fine geometry with exact primitive arclength at each sample."""

    points: np.ndarray
    distance: np.ndarray
    widths: np.ndarray
    corners: tuple[float, ...]

    @property
    def length(self) -> float:
        return float(self.distance[-1])

    def point_at(self, distance: float) -> np.ndarray:
        return np.array([np.interp(distance, self.distance, self.points[:, axis]) for axis in (0, 1)])

    def heading_at(self, distance: float) -> float:
        index = int(np.clip(np.searchsorted(self.distance, distance), 1, len(self.distance) - 1))
        direction = self.points[index] - self.points[index - 1]
        return math.atan2(direction[1], direction[0])


@dataclass(frozen=True)
class TrackSpec:
    """Serializable route and visible challenges; no controller receives this."""

    name: str
    sections: tuple[Section, ...]
    start: tuple[float, float, float] = (-0.5, 0.0, 0.0)
    gaps: tuple[tuple[float, float], ...] = ()
    branches: tuple[Branch, ...] = ()
    arena_size: float = 4.0
    resolution: float = 0.001
    seed: int | None = None
    difficulty: str = "manual"
    split: str = "manual"
    generator_version: str = GENERATOR_VERSION
    runout: float = 0.25

    def __post_init__(self) -> None:
        if self.arena_size <= 0 or self.resolution <= 0 or not all(math.isfinite(v) for v in (*self.start, self.arena_size, self.resolution)):
            raise ValueError("Track pose, arena, and resolution must be finite and positive where applicable")

    @cached_property
    def route(self) -> Route:
        return sample_route(self.sections, self.start)

    @property
    def features(self) -> tuple[str, ...]:
        names = {section.feature for section in self.sections}
        if self.gaps:
            names.add("gap")
        names.update("crossing" if branch.crossing else "branch" for branch in self.branches)
        return tuple(sorted(names))

    @cached_property
    def geometry_hash(self) -> str:
        """Canonical identity ignores translation, rotation, seed, and reflection."""
        heading = self.start[2]
        rotation = np.array([[math.cos(heading), math.sin(heading)], [-math.sin(heading), math.cos(heading)]])
        samples = np.append(np.arange(0, self.route.length, 0.005), self.route.length)
        points = np.array([self.route.point_at(s) for s in samples])
        local = (points - self.start[:2]) @ rotation.T
        distractors = [((branch_points(self, b) - self.start[:2]) @ rotation.T, b.width) for b in self.branches]
        identities = []
        for sign in (1, -1):
            reflect = np.array([1, sign])
            identity = {"route": np.rint(local * reflect * 100000).astype(int).tolist(),
                        "widths": np.rint(np.interp(samples, self.route.distance, self.route.widths) * 100000).astype(int).tolist(),
                        "gaps": np.rint(np.asarray(self.gaps) * 100000).astype(int).tolist(), "runout": round(self.runout, 5),
                        "branches": sorted((np.rint(xy * reflect * 100000).astype(int).tolist(), round(width, 5)) for xy, width in distractors)}
            identities.append(json.dumps(identity, sort_keys=True))
        canonical = min(identities)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TrackSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values: dict) -> TrackSpec:
        """Decode JSON-compatible data without mutating the caller's record."""
        values = dict(values)
        values["sections"] = tuple(Section(**s) for s in values["sections"])
        values["branches"] = tuple(Branch(**b) for b in values.get("branches", []))
        values["gaps"] = tuple(tuple(gap) for gap in values.get("gaps", []))
        values["start"] = tuple(values.get("start", (-0.5, 0.0, 0.0)))
        return cls(**values)


def sample_route(sections: tuple[Section, ...], start: tuple[float, float, float]) -> Route:
    """Tessellate analytic primitives without rounding polyline vertices."""
    if not sections:
        raise ValueError("A route needs at least one section")
    position = np.array(start[:2], dtype=float)
    heading = start[2]
    points, distances, widths = [position.copy()], [0.0], [sections[0].width]
    corners: list[float] = []
    for section in sections:
        if section.kind == "corner":
            heading += section.angle
            corners.append(distances[-1])
            continue
        length = section.length if section.kind == "straight" else section.radius * abs(section.angle)
        local_s = np.linspace(0.0, length, max(1, math.ceil(length / ROUTE_SAMPLE_SPACING)) + 1)[1:]
        if section.kind == "straight":
            samples = position + local_s[:, None] * np.array([math.cos(heading), math.sin(heading)])
        else:
            sign = math.copysign(1.0, section.angle)
            theta = heading + sign * local_s / section.radius
            samples = position + sign * section.radius * np.column_stack(
                (np.sin(theta) - math.sin(heading), math.cos(heading) - np.cos(theta)))
            heading += section.angle
        offset = distances[-1]
        points.extend(samples)
        distances.extend(offset + local_s)
        widths.extend([section.width] * len(samples))
        position = samples[-1]
    if len(points) < 2:
        raise ValueError("A route must have nonzero length")
    return Route(np.asarray(points), np.asarray(distances), np.asarray(widths), tuple(corners))


def branch_points(track: TrackSpec, branch: Branch) -> np.ndarray:
    center = track.route.point_at(branch.at)
    heading = track.route.heading_at(branch.at) + branch.angle
    direction = branch.length * np.array([math.cos(heading), math.sin(heading)])
    return np.array([center - direction if branch.crossing else center, center + direction])


@dataclass(frozen=True)
class GeneratorConfig:
    """Geometric distributions and finite search limits, in SI units."""

    arena_size: float = 4.0
    resolution: float = 0.001
    edge_margin: float = 0.16
    minimum_clearance: float = 0.20
    minimum_radius: float = 0.12
    width_min: float = 0.012
    width_max: float = 0.028
    max_attempts: int = 800
    excluded_features: tuple[str, ...] = ()
    generator_version: str = "1.0"
    target_motifs: int | None = None
    route_length_min: float | None = None
    route_length_max: float | None = None
    gap_count: int = 1
    gap_max_length: float = 0.085
    junctions_per_kind: int = 1

    def __post_init__(self) -> None:
        if self.arena_size <= 0 or self.resolution <= 0 or self.max_attempts < 1:
            raise ValueError("Invalid arena, resolution, or attempt count")
        if self.minimum_radius < 0.10 or self.minimum_clearance < 0.19:
            raise ValueError("Default robot needs radius >= 0.10 m and clearance >= 0.19 m")
        if not 0.005 <= self.width_min <= self.width_max <= 0.060:
            raise ValueError("Invalid line-width range")
        if self.generator_version not in {"1.0", "2.0"}:
            raise ValueError("Unsupported generator version")
        if self.target_motifs is not None and (type(self.target_motifs) is not int or self.target_motifs < 2):
            raise ValueError("target_motifs must be an integer >= 2")
        if any(v is not None and (not math.isfinite(v) or v <= 0) for v in (self.route_length_min, self.route_length_max)):
            raise ValueError("Length bounds must be finite and positive")
        if self.route_length_min and self.route_length_max and self.route_length_min > self.route_length_max:
            raise ValueError("Minimum length exceeds maximum length")
        if self.gap_count < 0 or self.junctions_per_kind < 0 or not .020 <= self.gap_max_length <= .15:
            raise ValueError("Invalid gap/junction settings")


def validate_track(track: TrackSpec, config: GeneratorConfig | None = None) -> list[str]:
    """Return explicit geometry errors. Adjacent swept regions may overlap."""
    config = config or GeneratorConfig(arena_size=track.arena_size, resolution=track.resolution)
    route = track.route
    errors: list[str] = []
    bound = track.arena_size / 2 - config.edge_margin
    if np.any(np.abs(route.points) + route.widths[:, None] / 2 > bound):
        errors.append("route exceeds arena clearance")
    if any(s.kind == "arc" and s.radius < config.minimum_radius for s in track.sections):
        errors.append("arc is below minimum turning radius")
    # A coarse broad check is conservative; dense samples handle near misses.
    stride = max(1, round(0.015 / ROUTE_SAMPLE_SPACING))
    points = route.points[::stride]
    distance = route.distance[::stride]
    separation = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    nonlocal_pair = np.abs(distance[:, None] - distance[None, :]) > 0.45
    if np.any(separation[nonlocal_pair] < config.minimum_clearance):
        errors.append("unintended self-intersection or insufficient route clearance")
    if track.generator_version == "2.0":
        # The visible finish extension also matters to sensors before completion.
        heading = route.heading_at(route.length)
        runout_end = route.points[-1]+track.runout*np.array([math.cos(heading), math.sin(heading)])
        if np.any(np.abs(runout_end)+route.widths[-1]/2 > track.arena_size/2):
            errors.append("finish runout leaves the reflectance floor")
        far_route = route.points[route.distance < route.length-.45]
        if len(far_route):
            runout_points = np.linspace(route.points[-1], runout_end, 20)
            if np.min(np.linalg.norm(runout_points[:, None]-far_route[None, :], axis=2)) < config.minimum_clearance:
                errors.append("finish runout approaches an earlier route section")
    previous_end = -1.0
    for begin, end in sorted(track.gaps):
        if not 0.10 < begin < end < route.length - 0.10 or begin < previous_end:
            errors.append("invalid or overlapping gap")
        previous_end = end
    for branch in track.branches:
        if not 0.15 < branch.at < route.length - 0.15 or branch.length <= 0 or branch.width <= 0:
            errors.append("invalid junction")
            continue
        if any(begin - 0.10 < branch.at < end + 0.10 for begin, end in track.gaps):
            errors.append("junction overlaps a gap")
        if any(abs(branch.at - corner) < 0.16 for corner in route.corners):
            errors.append("junction too close to a sharp corner")
        heading_change = math.remainder(route.heading_at(branch.at + 0.10) - route.heading_at(branch.at - 0.10), 2 * math.pi)
        if abs(heading_change) > 0.03:
            errors.append("junction must lie on a straight route")
        branch_xy = branch_points(track, branch)
        if np.any(np.abs(branch_xy) > bound):
            errors.append("distractor exceeds arena clearance")
        along = np.linspace(branch_xy[0], branch_xy[-1], 30)
        far_route = route.points[np.abs(route.distance - branch.at) > 0.45]
        if len(far_route) and np.min(np.linalg.norm(along[:, None] - far_route[None, :], axis=2)) < config.minimum_clearance:
            errors.append("distractor intersects another route section")
    return list(dict.fromkeys(errors))


def _motif(name: str, rng: np.random.Generator, config: GeneratorConfig) -> list[Section]:
    width = float(rng.uniform(config.width_min, config.width_max))
    sign = int(rng.choice([-1, 1]))
    straight = lambda length: Section("straight", length=float(length), width=width, feature=name)
    corner = lambda angle: Section("corner", angle=angle, width=width, feature=name)
    arc = lambda radius, angle: Section("arc", radius=float(radius), angle=angle, width=width, feature=name)
    if name == "straight":
        return [straight(rng.uniform(0.22, 0.75))]
    if name in {"gentle_curve", "tight_curve"}:
        radius = rng.uniform(0.26, 0.42) if name == "gentle_curve" else rng.uniform(config.minimum_radius, 0.20)
        return [arc(radius, sign * rng.uniform(math.pi / 6, math.pi / 2)), straight(0.14)]
    if name == "corner":
        return [straight(0.18), corner(sign * math.pi / 2), straight(rng.uniform(0.22, 0.45))]
    if name == "s_curve":
        radius = rng.uniform(0.17, 0.27)
        return [arc(radius, sign * math.pi / 3), arc(radius, -sign * 2 * math.pi / 3), arc(radius, sign * math.pi / 3)]
    if name == "zigzag":
        sections = [corner(sign * math.pi / 4)]
        for turn in (-1, 1, -1):
            sections.extend([straight(rng.uniform(0.20, 0.30)), corner(sign * turn * math.pi / 2)])
        sections.extend([straight(0.22), corner(sign * math.pi / 4)])
        return sections
    if name in {"hairpin", "u_turn"}:
        radius = rng.uniform(max(config.minimum_radius, 0.13), 0.18 if name == "hairpin" else 0.25)
        return [straight(0.18), arc(radius, sign * math.pi), straight(0.20)]
    raise ValueError(f"Unknown feature: {name}")


def generate_track(seed: int, difficulty: str = "beginner", config: GeneratorConfig | None = None,
                   *, required_feature: str | None = None, required_feature_count: int = 1) -> TrackSpec:
    """Build a bounded motif sequence, optionally guaranteeing a training feature.

    Default generation is unchanged. Focused tracks begin with a straight run-in
    and insert the required motif next; backtracking must retain that requirement.
    The same arena, turn feasibility and clearance checks still apply.
    """
    config = config or GeneratorConfig()
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    rng = np.random.default_rng(seed)
    choices = ["straight", "gentle_curve"]
    if difficulty != "beginner":
        choices += ["corner", "s_curve", "zigzag", "u_turn"]
    if difficulty == "advanced":
        choices += ["tight_curve", "hairpin"]
    choices = [name for name in choices if name not in config.excluded_features]
    if not choices:
        raise ValueError("All available geometric features were excluded")
    if required_feature is not None and (required_feature not in choices or "straight" not in choices):
        raise ValueError("Required feature and straight run-in must be available at this difficulty")
    if type(required_feature_count) is not int or not 1 <= required_feature_count <= 3:
        raise ValueError("Required feature count must be an integer from 1 to 3")
    start = (float(rng.uniform(-0.45, 0.45)), float(rng.uniform(-0.45, 0.45)), float(rng.uniform(-math.pi, math.pi)))
    target = config.target_motifs or {"beginner": 4, "intermediate": 6, "advanced": 8}[difficulty]
    blocks: list[list[Section]] = []
    failures = 0
    for attempt in range(config.max_attempts):
        if required_feature and not blocks:
            name = "straight"
        elif required_feature and sum(b[0].feature == required_feature for b in blocks) < required_feature_count:
            name = required_feature
        else:
            name = str(rng.choice(choices))
        proposed = _motif(name, rng, config)
        sections = tuple(s for block in blocks + [proposed] for s in block)
        candidate = TrackSpec(f"{difficulty}-{seed}", sections, start, arena_size=config.arena_size,
                              resolution=config.resolution, seed=seed, difficulty=difficulty,
                              generator_version=config.generator_version)
        final_too_short = len(blocks)+1 == target and config.route_length_min and candidate.route.length < config.route_length_min
        too_long = config.route_length_max and candidate.route.length > config.route_length_max
        if validate_track(candidate, config) or final_too_short or too_long:
            failures += 1
            if failures >= 15 and blocks:
                blocks.pop()
                failures = 0
            continue
        blocks.append(proposed)
        failures = 0
        if len(blocks) == target:
            return _add_challenges(candidate, rng, config)
    raise TrackGenerationError(f"Track generation exhausted {config.max_attempts} attempts: seed={seed}, difficulty={difficulty}")


def _add_challenges(track: TrackSpec, rng: np.random.Generator, config: GeneratorConfig) -> TrackSpec:
    from dataclasses import replace

    if track.difficulty != "advanced":
        return track
    gaps: tuple[tuple[float, float], ...] = ()
    if "gap" not in config.excluded_features and config.gap_count:
        at = float(rng.uniform(0.25, track.route.length - 0.30))
        gaps = ((at, at + float(rng.uniform(0.020, config.gap_max_length))),)
    result = replace(track, gaps=gaps)
    for _ in range(max(0, config.gap_count-1) if "gap" not in config.excluded_features else 0):
        for attempt in range(40):
            at = float(rng.uniform(.25, track.route.length-.30))
            candidate = replace(result, gaps=tuple(sorted(result.gaps+((at, at+float(rng.uniform(.020, config.gap_max_length))),))))
            if not validate_track(candidate, config):
                result = candidate
                break
    for crossing in (True, False)*config.junctions_per_kind:
        if ("crossing" if crossing else "branch") in config.excluded_features:
            continue
        for _ in range(40):
            branch = Branch(float(rng.uniform(0.25, track.route.length - 0.25)),
                            length=float(rng.uniform(0.10, 0.18)), crossing=crossing)
            candidate = replace(result, branches=result.branches + (branch,))
            if any(abs(branch.at - other.at) < 0.45 for other in result.branches):
                continue
            if not validate_track(candidate, config):
                result = candidate
                break
    return result


def straight_track(length: float = 1.0) -> TrackSpec:
    """Fixed straight benchmark, centered in the arena."""
    return TrackSpec("straight", (Section("straight", length=length),), (-length / 2, 0.0, 0.0))


class TrackMask:
    """One floor image used by sensor sampling, previews, and MuJoCo texture.

    Image rows increase downward; world +Y points upward. The floor is finite,
    white outside its image, and has no raised line geometry or contact seams.
    """

    def __init__(self, track: TrackSpec):
        self.track = track
        self.pixels = math.ceil(track.arena_size / track.resolution)
        if self.pixels > 8192:
            raise ValueError("Floor texture exceeds 8192 pixels; configure an explicit resolution budget")
        self.meters_per_pixel = track.arena_size / self.pixels
        image = Image.new("L", (self.pixels, self.pixels), 255)
        draw = ImageDraw.Draw(image)
        intervals = []
        position = 0.0
        for begin, end in sorted(track.gaps):
            intervals.append((position, begin))
            position = end
        intervals.append((position, track.route.length))
        for begin, end in intervals:
            distances = np.concatenate(([begin], track.route.distance[(track.route.distance > begin) & (track.route.distance < end)], [end]))
            points = np.column_stack([np.interp(distances, track.route.distance, track.route.points[:, axis]) for axis in (0, 1)])
            widths = np.interp(distances, track.route.distance, track.route.widths)
            self._draw_stroke(draw, points, widths)
        # The chassis finish gate precedes the end of the paint, so forward
        # sensors stay over line until the chassis has actually crossed it.
        heading = track.route.heading_at(track.route.length)
        end = track.route.points[-1]
        runout_end = end + track.runout * np.array([math.cos(heading), math.sin(heading)])
        self._draw_stroke(draw, np.array([end, runout_end]), np.repeat(track.route.widths[-1], 2))
        for branch in track.branches:
            self._draw_stroke(draw, branch_points(track, branch), np.array([branch.width, branch.width]))
        self.image = image
        self.field = np.asarray(image)

    def _draw_stroke(self, draw: ImageDraw.ImageDraw, points: np.ndarray, widths: np.ndarray) -> None:
        # Miter joins retain true angular vertices, including the outside of 90° turns.
        direction = np.diff(points, axis=0)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        normal = np.column_stack((-direction[:, 1], direction[:, 0]))
        offsets = np.empty_like(points)
        offsets[0], offsets[-1] = normal[0] * widths[0] / 2, normal[-1] * widths[-1] / 2
        if len(points) > 2:
            bisector = normal[:-1] + normal[1:]
            denominator = np.sum(bisector * normal[1:], axis=1)
            offsets[1:-1] = bisector * (widths[1:-1] / 2 / np.maximum(denominator, 0.1))[:, None]
        polygon = np.concatenate((points + offsets, (points - offsets)[::-1]))
        draw.polygon([tuple(p) for p in self.world_to_pixel(polygon)], fill=0)

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return np.column_stack(((points[:, 0] + self.track.arena_size / 2) / self.meters_per_pixel,
                                (self.track.arena_size / 2 - points[:, 1]) / self.meters_per_pixel))

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Binary reflectance at world XY; black is 1, white/outside is 0."""
        pixels = np.floor(self.world_to_pixel(points)).astype(int)
        valid = np.all((pixels >= 0) & (pixels < self.pixels), axis=1)
        result = np.zeros(len(points), dtype=float)
        result[valid] = self.field[pixels[valid, 1], pixels[valid, 0]] < 128
        return result
