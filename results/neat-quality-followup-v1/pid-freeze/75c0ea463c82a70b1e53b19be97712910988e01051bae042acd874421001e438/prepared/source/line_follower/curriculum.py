"""Training-only curriculum with rolling competence and explicit plateau rules.

No validation/test observations enter this module. State is checkpointed after
each generation, so a resume cannot restart or silently advance the schedule.
"""

from dataclasses import dataclass, field, replace
import math

from .tracks import GeneratorConfig

# Ten-slot allocations: master basic tracking before exposing harder levels.
STAGE_MIXTURES = (
    ("beginner",)*10,
    ("beginner",)*5 + ("intermediate",)*5,
    ("beginner",)*2 + ("intermediate",)*6 + ("advanced",)*2,
    ("beginner",)*2 + ("intermediate",)*3 + ("advanced",)*5,
    ("beginner",) + ("intermediate",)*2 + ("advanced",)*7,
)


@dataclass(frozen=True)
class CurriculumConfig:
    """Editable advancement controls; thresholds describe training competence."""

    window: int = 5
    minimum_generations: int = 5
    maximum_generations: int = 25
    elite_fraction: float = 0.10
    beginner_threshold: float = 0.85
    intermediate_threshold: float = 0.70
    advanced_threshold: float = 0.55
    length_scale: float = 1.0
    motif_scale: float = 1.0

    def __post_init__(self):
        if not 1 <= self.window <= self.minimum_generations <= self.maximum_generations:
            raise ValueError("Need 1 <= window <= minimum_generations <= maximum_generations")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if any(not 0 <= v <= 1 for v in (self.beginner_threshold, self.intermediate_threshold, self.advanced_threshold)):
            raise ValueError("Curriculum thresholds must be in [0, 1]")
        if any(not math.isfinite(v) or v <= 0 for v in (self.length_scale, self.motif_scale)):
            raise ValueError("Length/motif scales must be finite and positive")


@dataclass
class CurriculumState:
    stage: int = 0
    age: int = 0
    history: list[dict] = field(default_factory=list)

    def observe(self, counts: dict, config: CurriculumConfig) -> dict:
        """Advance at most one stage; never demote because of a noisy generation.

        counts maps difficulty to [completed, attempted] for the top training
        fraction. A plateau transition increases exposure, not a mastery claim.
        """
        before = self.stage
        self.age += 1
        self.history.append(counts)
        self.history = self.history[-config.window:]
        rates = {}
        for difficulty in ("beginner", "intermediate", "advanced"):
            successes = sum(h.get(difficulty, [0, 0])[0] for h in self.history)
            attempted = sum(h.get(difficulty, [0, 0])[1] for h in self.history)
            rates[difficulty] = successes/attempted if attempted else 0.0
        target = "beginner" if before == 0 else "intermediate" if before <= 2 else "advanced"
        threshold = getattr(config, f"{target}_threshold")
        mastered = (len(self.history) == config.window and self.age >= config.minimum_generations
                    and rates[target] >= threshold)
        plateau = self.age >= config.maximum_generations
        reason = "competence" if mastered else "plateau_exposure" if plateau else "stay"
        record = {"stage": before, "stage_age": self.age, "next_stage": before, "reason": reason,
                  "target_difficulty": target, **{f"rolling_{d}_completion": r for d, r in rates.items()}}
        if before < len(STAGE_MIXTURES)-1 and (mastered or plateau):
            self.stage += 1
            self.age, self.history = 0, []
        elif before == len(STAGE_MIXTURES)-1:
            record["reason"] = "final_stage"
        record["next_stage"] = self.stage
        return record


def stage_batch(stage: int, count: int, generation: int) -> tuple[str, ...]:
    if not 0 <= stage < len(STAGE_MIXTURES) or count < 1:
        raise ValueError("Invalid curriculum stage or episode count")
    mixture = STAGE_MIXTURES[stage]
    offset = generation*count % len(mixture) if count % len(mixture) else 0
    return tuple(mixture[(offset+i) % len(mixture)] for i in range(count))


def stage_geometry(base: GeneratorConfig, stage: int, difficulty: str,
                   config: CurriculumConfig = CurriculumConfig()) -> GeneratorConfig:
    """Separate length bounds (m) from motif counts; keep the 4 m arena budget.

    Original saved routes span B 1.21-2.52 m, I 2.80-5.45 m, A 3.25-7.25 m.
    Later levels extend that observed envelope; feasibility rejection still wins.
    """
    if difficulty == "beginner":
        motifs, low, high, radius = 4, 1.0, 2.8, .12
    elif difficulty == "intermediate":
        motifs, low, high = ((3, 1.0, 2.8), (4, 1.7, 3.8), (6, 2.7, 5.5),
                             (8, 3.5, 6.8), (8, 3.5, 6.8))[stage]
        radius = .12
    else:
        motifs, low, high, radius = ((6, 2.8, 5.5, .16), (6, 2.8, 5.5, .16),
                                     (8, 3.5, 6.8, .14), (10, 4.5, 8.5, .12),
                                     (12, 6.0, 10.0, .12))[stage]
    return replace(base, generator_version="2.0", target_motifs=base.target_motifs or max(2, round(motifs*config.motif_scale)),
                   route_length_min=base.route_length_min or low*config.length_scale,
                   route_length_max=base.route_length_max or high*config.length_scale, minimum_radius=radius,
                   gap_count=1 if stage < 3 else 2 if stage == 3 else 3,
                   gap_max_length=(.03, .04, .05, .07, .085)[stage],
                   junctions_per_kind=1 if stage < 4 else 2)


def training_competence(evaluated: dict, fitness: dict[int, float], fraction: float) -> dict:
    """Pooled completion counts from the best training fraction, never held-outs."""
    keys = sorted(evaluated, key=lambda key: (-fitness[key], key))[:max(1, math.ceil(len(evaluated)*fraction))]
    counts = {}
    for key in keys:
        for result in evaluated[key]:
            value = counts.setdefault(result.difficulty, [0, 0])
            value[0] += int(result.completed)
            value[1] += 1
    return counts
