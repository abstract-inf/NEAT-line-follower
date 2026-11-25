"""Joint speed/quality pressure using the unchanged V1 measurements."""
from neat_quality_study.fitness import LIMITS as ORIGINAL_LIMITS, score_batch as original_score

VERSION = 'joint-quality-speed-1.0'
LIMITS = {**ORIGINAL_LIMITS, 'speed_shortfall_scale_m_s': .05}


def score_batch(episodes, reference):
    result = original_score(episodes, reference)
    quality_max = max(result['violations'].values())
    shortfall = max(0., (LIMITS['straight_speed_target_m_s'] - result['minimum_straight_speed_m_s']) /
                    LIMITS['speed_shortfall_scale_m_s'])
    joint = max(quality_max, shortfall)
    complete = result['completed'] == result['courses']
    result.update(fitness_version=VERSION, original_v1_fitness=result['fitness'],
                  original_quality_violation_sum=result['quality_violation'],
                  quality_only_feasible=result['quality_feasible'], quality_max_violation=quality_max,
                  speed_shortfall=shortfall, joint_violation=joint, joint_feasible=complete and joint == 0.,
                  joint_components={**result['violations'], 'speed_shortfall': shortfall})
    if complete:
        result['fitness'] = 10.*result['courses'] + (10. + 9.*result['speed_utility'] if joint == 0. else 9./(1.+joint))
        result['tier'] = 'completed_joint_feasible' if joint == 0. else 'completed_joint_violation'
    return result
