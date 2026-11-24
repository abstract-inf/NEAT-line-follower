"""Recorded, opt-in memory management for the frozen CPU study engine."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch

import line_follower.experiments as engine


class MemoryBoundWorker(engine.WorkerContext):
    CACHE_BYTES = 64 * 1024 * 1024

    def mask(self, track):
        identity = hashlib.sha256(json.dumps(asdict(track), sort_keys=True).encode()).hexdigest()
        if identity in self.masks:
            self.masks.move_to_end(identity)
            return self.masks[identity]
        pixels = math.ceil(track.arena_size / track.resolution)
        needed = 2 * pixels * pixels  # Pillow grayscale image and NumPy raster.
        retained = sum(m.field.nbytes + m.image.width*m.image.height for m in self.masks.values())
        while self.masks and (len(self.masks) >= engine.MASK_CACHE_SIZE or retained + needed > self.CACHE_BYTES):
            _, old = self.masks.popitem(last=False)
            retained -= old.field.nbytes + old.image.width*old.image.height
            del old
        mask = engine.TrackMask(track)
        if needed <= self.CACHE_BYTES:
            self.masks[identity] = mask
        return mask


def initialize_worker(robot, episode, neat_config, time_limit):
    # Importable top-level initializer for Windows spawn, including replacements.
    engine._WORKER_CONTEXT = MemoryBoundWorker(robot, episode, neat_config, time_limit)


class MemorySafeEvaluator(engine.Evaluator):
    MAX_TASKS = 128
    WORKERS = None

    def __init__(self, robot, episode, neat_config, workers=1, time_limit=None,
                 backend="cpu", gpu_batch_size=128):
        if backend != "cpu":
            raise ValueError("Memory-safe study mode requires the CPU backend")
        workers = self.WORKERS if self.WORKERS is not None else workers
        super().__init__(robot, episode, neat_config, 1, time_limit, backend, gpu_batch_size)
        self.serial = MemoryBoundWorker(robot, episode, neat_config, time_limit)
        if workers > 1:
            self.pool = mp.get_context("spawn").Pool(
                workers, initializer=initialize_worker,
                initargs=(robot, episode, neat_config, time_limit), maxtasksperchild=self.MAX_TASKS)


@contextmanager
def memory_safe_execution(seed_directory, workers=None):
    """Archive the runtime adjustment without changing frozen input hashes."""
    if workers is not None and not 1 <= workers <= 8:
        raise ValueError("Evaluation workers must be between 1 and 8")
    sources = [Path(__file__), Path(__file__).with_name("run_fast_neat_replicates.py")]
    destination = Path(seed_directory) / "memory-recovery" / str(time.time_ns())
    destination.mkdir(parents=True)
    hashes = {}
    for source in sources:
        data = source.read_bytes()
        (destination / source.name).write_bytes(data)
        hashes[source.name] = hashlib.sha256(data).hexdigest()
    record = {"change": "Pre-allocation cache eviction and periodic CPU-worker recycling",
              "cache_bytes_per_worker": MemoryBoundWorker.CACHE_BYTES,
              "tasks_per_worker_before_replacement": MemorySafeEvaluator.MAX_TASKS,
              "source_sha256": hashes, "evolution_settings_changed": False,
              "runtime_workers_override": workers,
              "note": "Execution-only adjustment; retain and verify original protocol and checkpoint fingerprints."}
    (destination / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with patch.object(engine, "Evaluator", MemorySafeEvaluator), patch.object(MemorySafeEvaluator, "WORKERS", workers):
        yield
