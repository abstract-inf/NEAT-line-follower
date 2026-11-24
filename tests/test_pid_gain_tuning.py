"""Protect the gain-only experiment boundary and deterministic search schedule."""

from dataclasses import asdict

import pytest

from line_follower.pid import ResearchPIDConfig
from run_pid_gain_tuning import GAIN_NAMES, gain_config, initial_parameters


@pytest.mark.parametrize("supervised", [False, True])
def test_search_is_reproducible_unique_and_changes_only_gains(supervised):
    baseline = ResearchPIDConfig(corner_supervision=supervised)
    first = initial_parameters(baseline)
    assert first == initial_parameters(baseline)
    assert len(first) == 160
    assert first[0] == ("baseline", baseline)
    triples = {tuple(getattr(p, n) for n in GAIN_NAMES) for _, p in first}
    assert len(triples) == len(first)
    for _, parameters in first:
        assert all(getattr(parameters, n) == getattr(baseline, n)
                   for n in asdict(baseline) if n not in GAIN_NAMES)


def test_out_of_protocol_gain_triples_are_rejected():
    with pytest.raises(ValueError):
        gain_config(ResearchPIDConfig(), [-1, .1, .1])
    with pytest.raises(ValueError):
        gain_config(ResearchPIDConfig(), [1, .1])
