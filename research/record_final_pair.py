"""Exactly two verified 1 m/s replays on the known Rhythm Changes course."""
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'runs/neat-joint-study-v1/prepared/source'
sys.path.insert(0, str(ENGINE))
sys.path.insert(1, str(ROOT))

import mujoco
import numpy as np
from line_follower.config import RobotConfig
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask, TrackSpec
from line_follower.viewer import set_transparency
from neat_smooth_study.common import digest, read, write_json
from run_rhythm_changes_case_study import setup
from record_rhythm_changes import record, VLC

OUTPUT = ROOT / 'runs/rhythm-changes-current-pair-20261125'


def prepare():
    source = ROOT / 'runs/neat-pid-finalization-20261125/frozen'
    old = read(source / 'manifest.json')
    track = ROOT / 'runs/rhythm-changes-case-study-20261121/track.json'
    inputs = {'frozen/neat-prototype.pkl': source / 'neat-prototype.pkl',
              'frozen/pid-100.json': source / 'pid-100.json', 'track.json': track}
    if (OUTPUT / 'protocol.json').exists():
        plan = read(OUTPUT / 'protocol.json')
        for name, expected in plan['inputs'].items():
            if digest(OUTPUT / name) != expected or digest(inputs[name]) != expected:
                raise ValueError('Recording inputs changed.')
        if plan['recorder_sha256'] != digest(Path(__file__)):
            raise ValueError('Recorder changed; preserve this run and version any revision.')
        return plan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        destination = OUTPUT / name; destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
    for name in ('neat-prototype.pkl', 'pid-100.json'):
        if digest(source / name) != old['controllers'][name]['sha256']:
            raise ValueError('Finalization controller fingerprint mismatch.')
    if read(source / 'pid-100.json')['base'] != 1.:
        raise ValueError('PID must retain full-cruise base=1.')
    original = read(ROOT / 'runs/neat-joint-study-v1/prepared/protocol.json')
    for name, expected in original['source_files'].items():
        if digest(ENGINE / name) != expected:
            raise ValueError('Frozen simulation source changed.')
    plan = {'scope': 'Known development-course showcase; fixed models, no training or model selection. NEAT remains validation-ineligible for global error/jitter.',
        'robot': original['robot'], 'episode': original['episode'],
        'inputs': {name:digest(path) for name,path in inputs.items()},
        'recorder_sha256': digest(Path(__file__)),
        'render_helper_sha256': digest(ROOT / 'research/record_rhythm_changes.py'),
        'policies': {
            'neat-joint-g49': {'family':'NEAT','file':'frozen/neat-prototype.pkl', 'seed':1, 'generation':49, 'native_cap_m_s':1.},
            'pid-retuned': {'family':'PID','file':'frozen/pid-100.json', 'native_cap_m_s':1.}},
        'tasks': [{'id':name+'__cap-100','policy':name,'cap_m_s':1.,'time_limit_s':180.}
                  for name in ('neat-joint-g49','pid-retuned')]}
    write_json(OUTPUT / 'protocol.json', plan)
    return plan


def main():
    if not VLC.is_file(): raise FileNotFoundError('Existing VLC installation is required.')
    plan = prepare()
    mask = TrackMask(TrackSpec.load(OUTPUT / 'track.json'))
    if mask.track.geometry_hash != '77fb2a4f7caad1dd33e3594335e1639faf2d03f339e25694b4dc3744c9205d6a':
        raise ValueError('This is not the selected Rhythm Changes course.')
    videos = OUTPUT / 'videos'; videos.mkdir(exist_ok=True)
    episodes = OUTPUT / 'episodes'; episodes.mkdir(exist_ok=True)
    for task in plan['tasks']:
        path = episodes / (task['id']+'.json')
        if path.exists(): continue
        episode = setup(OUTPUT,plan,task,mask)
        expected_class = 'NEATController' if plan['policies'][task['policy']]['family']=='NEAT' else 'ResearchPIDController'
        if type(episode.controller).__name__ != expected_class:
            raise ValueError('Controller identity mismatch.')
        while not episode.done: episode.advance()
        result = asdict(episode.result())
        if result['error']: raise RuntimeError(result['error'])
        write_json(path, {**result,'cap_m_s':1.,'controller_class':expected_class,
                          'controller_sha256':digest(OUTPUT/plan['policies'][task['policy']]['file']),
                          'protocol_sha256':digest(OUTPUT/'protocol.json')})
        print(task['id'], result['termination'], result['elapsed_s'], flush=True)
    visual = Simulation(RobotConfig(**{**plan['robot'],'max_speed':1.}))
    visual.reset(mask.track,mask); visual.enable_rendering(); set_transparency(visual,True)
    visual.model.vis.global_.offwidth=736; visual.model.vis.global_.offheight=560
    clips=[]
    with mujoco.Renderer(visual.model,height=560,width=736) as renderer:
        for task in plan['tasks']:
            target=videos/(task['id']+'.mp4'); sidecar=target.with_suffix('.json')
            if sidecar.exists():
                clip=read(sidecar)
                if digest(target)!=clip['sha256']:raise ValueError('Saved video changed.')
            else:
                print('Rendering '+task['id'],flush=True);started=time.perf_counter()
                clip=record(OUTPUT,plan,task,mask,visual,renderer,target)
                clip.update(controller_sha256=digest(OUTPUT/plan['policies'][task['policy']]['file']),
                            cap_m_s=1.,protocol_sha256=digest(OUTPUT/'protocol.json'))
                write_json(sidecar,clip)
                preview=videos/'preview.png'
                if preview.exists():preview.rename(videos/(task['policy']+'-preview.png'))
                print(f"Saved {target.name}; wall {time.perf_counter()-started:.1f}s",flush=True)
            clips.append(clip)
    assert len(clips)==2
    write_json(videos/'manifest.json',{'scope':plan['scope'],'fps':15,'playback':'1x','clips':clips})
    (videos/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Current NEAT and PID</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:25px auto}video{width:100%}</style>'
        '<h1>Rhythm Changes — current NEAT and tuned PID</h1><p>Same 1.00 m/s cap, 1x playback, 15 fps. Known development course; frozen models.</p>'+
        ''.join(f'<h2>{c["episode"]}</h2><video controls preload="none" src="{c["file"]}"></video>' for c in clips),encoding='utf-8')
    print('Exactly two verified MP4 videos are ready.',flush=True)


if __name__=='__main__':main()
