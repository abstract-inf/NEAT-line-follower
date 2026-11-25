"""Continuous quality discount with the original measurements and speed utility."""
from neat_quality_study.fitness import LIMITS, score_batch as original_score

VERSION = 'continuous-quality-speed-1.0'


def score_batch(episodes, reference):
    result = original_score(episodes, reference)
    result['fitness_version'] = VERSION
    result['quality_discount'] = 1. / (1. + result['quality_violation'])
    result['discounted_speed_utility'] = result['speed_utility'] * result['quality_discount']
    if result['completed'] == result['courses']:
        result['fitness'] = 10. * result['courses'] + 9. * result['discounted_speed_utility']
    return result
