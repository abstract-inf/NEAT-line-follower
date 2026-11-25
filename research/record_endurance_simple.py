"""Replay the accepted lambda0 genome once on the known Endurance Track.

The course is the unchanged historical Rhythm Changes geometry. No training,
controller selection, tuning or new final-test geometry is performed.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'runs/neat-simple-pilot-50-v1'
ENGINE = STUDY / 'prepared/source'
sys.path.insert(0, str(ENGINE))
sys.path.insert(1, str(ROOT))

import mujoco
import numpy as np
from line_follower.config import RobotConfig
from line_follower.controllers import read_model_artifact
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask, TrackSpec
from line_follower.viewer import set_transparency
from neat_smooth_study.common import dependencies, digest, read, write_json
from run_rhythm_changes_case_study import geometric_errors, replay, setup
from record_rhythm_changes import record, VLC

OUTPUT = ROOT / 'runs/endurance-simple-lambda0-20261125'
IDENTITY = '77fb2a4f7caad1dd33e3594335e1639faf2d03f339e25694b4dc3744c9205d6a'


def prepare():
    original = read(STUDY / 'prepared/protocol.json')
    for name, expected in original['source_files'].items():
        if digest(ENGINE / name) != expected:
            raise ValueError('Frozen training/evaluation source changed.')
    accepted = ROOT / 'results/neat-simple-pilot-50-v1/accepted-development'
    manifest = read(accepted / 'manifest.json')
    inputs = {'frozen/neat-lambda0-g44.pkl': accepted / 'neat-lambda0-g44-r0.pkl',
              'track.json': ROOT / 'runs/rhythm-changes-case-study-20261121/track.json'}
    if digest(inputs['frozen/neat-lambda0-g44.pkl']) != manifest['model_sha256']:
        raise ValueError('The selected development model changed.')
    metadata = read_model_artifact(inputs['frozen/neat-lambda0-g44.pkl'])['metadata']
    if (metadata['generation'] != 44 or metadata['arm'] != 'lambda-0'
            or metadata['genotype_sha256'] != manifest['genotype_sha256']
            or metadata['protocol_sha256'] != digest(STUDY / 'prepared/protocol.json')):
        raise ValueError('Unexpected selected controller provenance.')
    if TrackSpec.load(inputs['track.json']).geometry_hash != IDENTITY:
        raise ValueError('The historical long course geometry differs.')
    sources = {'recorder': Path(__file__), 'render_helper': ROOT / 'research/record_rhythm_changes.py',
               'case_helper': ROOT / 'run_rhythm_changes_case_study.py',
               'index_helper': ROOT / 'research/record_speed_comparisons.py'}
    fingerprints = {name: digest(path) for name, path in inputs.items()}
    source_hashes = {name: digest(path) for name, path in sources.items()}
    protocol = OUTPUT / 'protocol.json'
    if protocol.exists():
        plan = read(protocol)
        if (plan['inputs'] != fingerprints or plan['render_sources'] != source_hashes
                or plan['dependencies'] != dependencies()):
            raise ValueError('Saved showcase source, inputs or dependencies differ.')
        for name, expected in fingerprints.items():
            if digest(OUTPUT / name) != expected:
                raise ValueError('Frozen showcase input changed.')
        return plan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        target = OUTPUT / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name, path in sources.items():
        target = OUTPUT / 'source' / (name + '.py')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    policy = {'family': 'NEAT', 'file': 'frozen/neat-lambda0-g44.pkl',
              'seed': 1, 'generation': 44, 'native_cap_m_s': 1.,
              'display_name': 'NEAT | simple fitness lambda=0 | generation 44'}
    task = {'id': 'neat-lambda0-g44__cap-100', 'policy': 'neat-lambda0-g44',
            'cap_m_s': 1., 'time_limit_s': 180.}
    plan = {'scope': 'One requested showcase on a known development course; no tuning, evolution or controller reselection.',
            'track_display_name': 'Endurance Track', 'historical_track_name': 'Rhythm Changes',
            'track_geometry_sha256': IDENTITY, 'inputs': fingerprints, 'render_sources': source_hashes,
            'training_protocol_sha256': digest(STUDY / 'prepared/protocol.json'),
            'dependencies': dependencies(), 'robot': original['robot'], 'episode': original['episode'],
            'policies': {task['policy']: policy}, 'tasks': [task]}
    write_json(protocol, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', action='store_true', help='Open the already saved model/course in the native viewer.')
    args = parser.parse_args()
    plan = prepare()
    task = plan['tasks'][0]
    if args.view:
        # The generic viewer verifies a different historical protocol format;
        # use the public viewer with the exact frozen model and saved track.
        from line_follower.viewer import view_tracks
        from line_follower.controllers import load_model
        controller, _ = load_model(OUTPUT / plan['policies'][task['policy']]['file'])
        view_tracks([TrackSpec.load(OUTPUT / 'track.json')],
                    RobotConfig(**{**plan['robot'], 'max_speed': 1.}), controller)
        return
    if not VLC.is_file():
        raise FileNotFoundError('The existing VLC installation is needed to encode this video.')
    mask = TrackMask(TrackSpec.load(OUTPUT / 'track.json'))
    path = OUTPUT / 'episodes' / (task['id'] + '.json')
    if not path.exists():
        episode = setup(OUTPUT, plan, task, mask)
        if type(episode.controller).__name__ != 'NEATController':
            raise ValueError('Expected the frozen NEAT controller.')
        positions = []
        while not episode.done:
            episode.advance()
            positions.append(episode.simulation.state.position[:2].copy())
        result = asdict(episode.result())
        if result['error']:
            raise RuntimeError(result['error'])
        error = geometric_errors(np.asarray(positions), mask.track.route.points)
        write_json(path, {**result, 'geometry_lateral_rmse_m': float(np.sqrt(np.mean(error**2))),
                         'cap_m_s': 1., 'controller_class': type(episode.controller).__name__,
                         'controller_sha256': plan['inputs']['frozen/neat-lambda0-g44.pkl'],
                         'protocol_sha256': digest(OUTPUT / 'protocol.json')})
    expected = read(path)
    if (expected['protocol_sha256'] != digest(OUTPUT / 'protocol.json')
            or expected['controller_sha256'] != plan['inputs']['frozen/neat-lambda0-g44.pkl']):
        raise ValueError('Saved episode identity differs.')
    print({key: expected[key] for key in ('completed', 'elapsed_s', 'progress', 'average_forward_speed_m_s',
                                         'peak_forward_speed_m_s', 'geometry_lateral_rmse_m', 'command_jitter')}, flush=True)
    videos = OUTPUT / 'videos'
    videos.mkdir(exist_ok=True)
    destination = videos / 'endurance-neat-lambda0-g44.mp4'
    sidecar = destination.with_suffix('.json')
    if sidecar.exists():
        if digest(destination) != read(sidecar)['sha256']:
            raise ValueError('Saved video changed.')
    else:
        visual = Simulation(RobotConfig(**{**plan['robot'], 'max_speed': 1.}))
        visual.reset(mask.track, mask)
        visual.enable_rendering()
        set_transparency(visual, True)
        visual.model.vis.global_.offwidth = 736
        visual.model.vis.global_.offheight = 560
        print('Rendering one verified Endurance Track replay.', flush=True)
        with mujoco.Renderer(visual.model, height=560, width=736) as renderer:
            clip = record(OUTPUT, plan, task, mask, visual, renderer, destination,
                          course_label='Endurance Track')
        write_json(sidecar, {**clip, 'controller_sha256': expected['controller_sha256'],
                             'protocol_sha256': digest(OUTPUT / 'protocol.json')})
    print('Saved ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
