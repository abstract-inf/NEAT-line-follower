"""Descriptive training diagnostics; feature tags indicate association, not cause."""

from collections import Counter, defaultdict
import math

import numpy as np


def episode_summaries(generation: int, role: str, episodes: list) -> list[dict]:
    """Include failures and observed-duration errors; do not imply independence."""
    groups = defaultdict(list)
    for result in episodes:
        groups[("all", "all")].append(result)
        groups[("difficulty", result.difficulty)].append(result)
        for feature in result.features.split(";"):
            groups[("contains_feature", feature)].append(result)
        lower = math.floor(result.route_length_m/2)*2
        groups[("length_m", f"{lower}-{lower+2}")].append(result)
    rows = []
    for (category, value), results in sorted(groups.items()):
        completed = [r for r in results if r.completed]
        failures = Counter(r.termination for r in results)
        rows.append({"generation": generation, "role": role, "category": category, "value": value,
                     "episodes": len(results), "completed": len(completed),
                     "completion_rate": len(completed)/len(results),
                     "mean_progress": float(np.mean([r.progress for r in results])),
                     "observed_lateral_rmse_m": float(np.mean([r.lateral_rmse_m for r in results])),
                     "mean_forward_speed_m_s": float(np.mean([r.average_forward_speed_m_s for r in results])),
                     "mean_length_m": float(np.mean([r.route_length_m for r in results])),
                     "mean_time_limit_s": float(np.mean([r.time_limit_s for r in results])),
                     "completion_time_mean_s": float(np.mean([r.completion_time_s for r in completed])) if completed else "",
                     "mean_progress_before_failure": float(np.mean([r.progress for r in results if not r.completed]))
                         if len(completed) < len(results) else "",
                     **{f"fail_{reason}": failures[reason] for reason in ("off_route", "stalled", "timeout", "tipped", "invalid_physics", "controller_or_simulation_error")}})
    return rows
