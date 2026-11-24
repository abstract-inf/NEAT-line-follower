"""Protocol constants, fingerprints and atomic study records."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from line_follower.experiments import write_json, append_csv
from line_follower.suites import derived_seed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'runs/neat-smooth-study-v1'
STUDY_SEED = 2026112301
PARENTS = (134, 136, 131, 107, 134)
REWARDS = {'old': 'route-speed-1.0', 'smooth': 'route-speed-smooth-1.0'}
SETTINGS = {'generations':100, 'population':100, 'episodes_per_genome':6,
            'cap_m_s':1., 'workers':2, 'mask_cache_bytes':64*1024**2,
            'worker_recycle_tasks':128, 'aggregation':'mean_cvar', 'cvar_fraction':.5}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    import csv
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def dependencies():
    return {name:importlib.metadata.version(name) for name in
            ('neat-python','mujoco','numpy','Pillow','matplotlib')}


def sources(root=ROOT):
    paths = [root/'run_neat_smooth_study.py', root/'requirements-lock.txt']
    for folder in ('line_follower','neat_smooth_study','configs'):
        paths.extend(p for p in (root/folder).rglob('*')
                     if p.is_file() and p.suffix in {'.py','.xml','.json','.txt'})
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(paths) if p.is_file()}


def verify(run):
    prepared = Path(run)/'prepared'
    if (prepared/'protocol.sha256').read_text(encoding='utf-8').strip()!=digest(prepared/'protocol.json'):
        raise ValueError('Prepared protocol changed after commitment')
    plan = read(prepared/'protocol.json')
    if plan['settings'] != SETTINGS or plan['dependencies'] != dependencies():
        raise ValueError('Study settings or dependencies changed; do not resume this protocol')
    if sources() != plan['source_files']:
        raise ValueError('Experiment source changed; preserve the frozen source to resume')
    for relative, sha in plan['prepared_files'].items():
        if digest(prepared/relative) != sha:
            raise ValueError(f'Prepared input changed: {relative}')
    if sources(prepared/'source') != plan['source_files']:
        raise ValueError('Frozen source bundle changed')
    return plan


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'session.lock').open('a+b') as stream:
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another process owns {directory}') from error
        yield


def entry_path(run, phase, entry):
    return Path(run)/'prepared'/phase/'suite'/entry['path']


def epoch_seed(phase, seed):
    return derived_seed(STUDY_SEED, phase, seed)
