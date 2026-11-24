from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import line_follower.experiments as engine
from line_follower.tracks import Section, TrackSpec


def allocation_probe(monkeypatch, context_type):
    context = context_type.__new__(context_type)
    context.masks = OrderedDict()
    tracks = [TrackSpec(f"probe-{i}", (Section("straight", length=1+i*.1),),
                        arena_size=.016, resolution=.001) for i in range(3)]
    def allocate(track):
        if len(context.masks) >= 2:
            raise MemoryError("TrackMask allocation while old rasters are retained")
        return SimpleNamespace(image=Image.new("L", (16, 16)), field=np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(engine, "TrackMask", allocate)
    return context, tracks


def test_original_cache_can_fail_before_post_allocation_eviction(monkeypatch):
    context, tracks = allocation_probe(monkeypatch, engine.WorkerContext)
    context.mask(tracks[0])
    context.mask(tracks[1])
    with pytest.raises(MemoryError, match="old rasters"):
        context.mask(tracks[2])


def test_bounded_cache_evicts_before_allocation_and_keeps_hits(monkeypatch):
    from fast_study_memory import MemoryBoundWorker
    monkeypatch.setattr(MemoryBoundWorker, "CACHE_BYTES", 1024)
    context, tracks = allocation_probe(monkeypatch, MemoryBoundWorker)
    for track in tracks:
        context.mask(track)
    assert len(context.masks) == 2
    assert context.mask(tracks[-1]) is context.mask(tracks[-1])


def test_oversized_masks_are_not_cached(monkeypatch):
    from fast_study_memory import MemoryBoundWorker
    monkeypatch.setattr(MemoryBoundWorker, "CACHE_BYTES", 128)
    context, tracks = allocation_probe(monkeypatch, MemoryBoundWorker)
    context.mask(tracks[0])
    assert not context.masks


def study_fixture():
    from pathlib import Path
    from line_follower.controllers import read_model_artifact
    root = Path("runs/neat-fast-study")
    parent = root / "prepared/inputs/seed-05-low.pkl"
    if not parent.exists():
        pytest.skip("Local frozen study is optional")
    artifact = read_model_artifact(parent)
    tracks = [TrackSpec.load(root / f"seed-05/training/tracks/generation-0240/{i}.json") for i in range(6)]
    return artifact, tracks


def test_real_masks_and_episode_results_are_identical():
    from fast_study_memory import MemoryBoundWorker
    from line_follower.config import EpisodeConfig, RobotConfig
    artifact, tracks = study_fixture()
    args = (RobotConfig(max_speed=1.25), EpisodeConfig(), artifact["config"], .25)
    original, bounded = engine.WorkerContext(*args), MemoryBoundWorker(*args)
    for track in tracks:
        assert np.array_equal(original.mask(track).field, bounded.mask(track).field)
        assert original.evaluate((artifact["genome"], [track])) == bounded.evaluate((artifact["genome"], [track]))


def test_windows_workers_recycle_and_return_identical_results(monkeypatch):
    from fast_study_memory import MemorySafeEvaluator
    from line_follower.config import EpisodeConfig, RobotConfig
    artifact, tracks = study_fixture()
    monkeypatch.setattr(MemorySafeEvaluator, "MAX_TASKS", 1)
    monkeypatch.setattr(MemorySafeEvaluator, "WORKERS", 2)
    args = (RobotConfig(max_speed=1.25), EpisodeConfig(), artifact["config"])
    with engine.Evaluator(*args, workers=1, time_limit=.1) as serial:
        expected = serial.validate(artifact["genome"], tracks)
    with MemorySafeEvaluator(*args, workers=8, time_limit=.1) as parallel:
        assert len(parallel.pool._pool) == 2
        original_pids = {worker.pid for worker in parallel.pool._pool}
        assert parallel.validate(artifact["genome"], tracks) == expected
        assert {worker.pid for worker in parallel.pool._pool} != original_pids
