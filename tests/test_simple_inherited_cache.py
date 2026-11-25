"""Allow only explicitly frozen ancestor caches at a continuation boundary."""
from copy import deepcopy

import pytest

from line_follower.experiments import _save_checkpoint
from neat_quality_followup import training
from neat_smooth_study.common import read, write_json
from test_quality_followup_training import checkpoint, prepared_run


@pytest.mark.parametrize('cache_signature', ['exact-parent', 'unknown', 'none'])
def test_continuation_accepts_only_exact_declared_ancestor(prepared_run, cache_signature):
    fixture = prepared_run
    directory = training.train(fixture.run, 'fast', workers=6, stop_after=1)
    parent = checkpoint(directory)
    original_rng = deepcopy(parent['random_state'])
    validation_calls = sum(label.startswith('validation') for label, _, _ in fixture.calls)
    ancestor = deepcopy(parent['signature'])
    fixture.plan['inherited_validation_signatures'] = {'fast': ancestor}
    write_json(fixture.run / 'prepared/protocol.json', fixture.plan)
    parent['signature'] = training.training_signature(fixture.run, fixture.plan, 'fast', training.VERSION)
    _save_checkpoint(directory / 'checkpoints/latest.pkl.gz', parent)
    if cache_signature != 'exact-parent':
        for path in (directory / 'validation-cache').glob('*.json'):
            row = read(path)
            row['signature'] = {} if cache_signature == 'unknown' else None
            write_json(path, row)
        with pytest.raises(ValueError, match='Validation cache'):
            training.train(fixture.run, 'fast', workers=8, resume=True)
    else:
        training.train(fixture.run, 'fast', workers=8, resume=True)
        final = checkpoint(directory)
        assert final['population'].generation == 2
        assert final['population'].draws[:1] == parent['population'].draws
        assert original_rng != final['random_state']
        # Validation actually reused the exact parent records; no new physics.
        assert sum(label.startswith('validation') for label, _, _ in fixture.calls) == validation_calls
