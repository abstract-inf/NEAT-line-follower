"""Disjoint speed/recovery banks and deterministic shared generation batches."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from line_follower.fast_study import centered_course
from line_follower.suites import corner_track, derived_seed
from line_follower.tracks import GeneratorConfig, Section, TrackGenerationError, generate_track, validate_track
from .common import read, write_json, STUDY_SEED


def protected_geometry(project):
    protected=set()
    for folder in ('runs','benchmarks'):
        for path in (Path(project)/folder).rglob('suite.json'):
            data=read(path)
            protected.update(data.get('protected_geometry_hashes',[]))
            protected.update(r['geometry_hash'] for split in data.get('splits',{}).values() for r in split)
        for path in (Path(project)/folder).rglob('batch.json'):
            protected.update(read(path).get('hashes',[]))
        for path in (Path(project)/folder).rglob('training-manifest.json'):
            protected.update(r['geometry_hash'] for r in read(path).get('tracks',[]))
    return protected


def speed_course(seed, slot, split):
    rng=np.random.default_rng(seed)
    length=float(rng.uniform(5.05,8.))
    sign=1 if slot%2 else -1
    sections=[]
    if slot>=10:
        sections.append(Section('straight',length=float(rng.uniform(.8,1.5))))
        kind=(slot-10)//2
        if kind in (0,1):
            sections.append(Section('corner',angle=sign*(math.pi/4 if kind==0 else math.pi/2),feature='recovery-turn'))
        elif kind in (2,3):
            angle=sign*(math.pi/3 if kind==2 else math.pi/2)
            radius=float(rng.uniform(.35,.8))
            sections.extend((Section('arc',radius=radius,angle=angle,feature='recovery-curve'),
                             Section('arc',radius=radius,angle=-angle,feature='recovery-curve')))
        else:
            for direction in (sign,-sign,-sign,sign):
                sections.extend((Section('corner',angle=direction*math.pi/4,feature='recovery-chicane'),
                                 Section('straight',length=float(rng.uniform(.4,.65)))))
    sections.append(Section('straight',length=length,feature='recovery-straight' if slot>=10 else 'speed-straight'))
    return centered_course(sections,f'smooth-speed-{seed}',seed,split)


def make_course(seed, group, index, split):
    if group=='speed':return speed_course(seed,index,split)
    if group=='corner':return replace(corner_track(seed,3,index%2,'advanced',GeneratorConfig()),split=split)
    difficulty=('beginner','intermediate','advanced')[index%3]
    return replace(generate_track(seed,difficulty),split=split)


def make_suite(directory, namespace, counts, used):
    directory=Path(directory)
    manifest={'suite_version':1,'root_seed':STUDY_SEED,'namespace':namespace,'splits':{}}
    for split,groups in counts.items():
        entries=[]
        for group,count in groups.items():
            for index in range(count):
                # Divide the speed group evenly between isolated and recovery courses.
                slot=index if group!='speed' else (index%2)*10+(index//2)%10
                for attempt in range(100):
                    seed=derived_seed(STUDY_SEED,namespace,split,group,index,attempt)
                    try:track=make_course(seed,group,slot,split)
                    except TrackGenerationError:continue
                    if not validate_track(track) and track.geometry_hash not in used:break
                else:raise ValueError(f'Unable to generate {namespace}/{split}/{group}/{index}')
                used.add(track.geometry_hash)
                relative=f'{split}/{group}-{index:03d}.json'
                track.save(directory/relative)
                entries.append({'path':relative,'geometry_hash':track.geometry_hash,'group':group,
                                'kind':'post_turn' if group=='speed' and slot>=10 else 'isolated' if group=='speed' else group,
                                'difficulty':track.difficulty,'seed':seed})
        manifest['splits'][split]=entries
    write_json(directory/'suite.json',manifest)
    return manifest


def generation_batch(seed,generation,bank,forbidden):
    rng=np.random.default_rng(derived_seed(seed,'bank-cycle',generation//10))
    isolated=rng.permutation(10)[generation%10]
    recovery=10+rng.permutation(10)[generation%10]
    tracks=[bank[isolated],bank[recovery]]
    used=set(forbidden)|{t.geometry_hash for t in bank}
    for slot in range(4):
        for attempt in range(100):
            track_seed=derived_seed(seed,'smooth-batch',generation,slot,attempt)
            try:
                track=(generate_track(track_seed,'intermediate' if slot==0 else 'advanced') if slot<2
                       else corner_track(track_seed,3,(generation+slot)%2,'advanced',GeneratorConfig()))
            except TrackGenerationError:continue
            if track.geometry_hash not in used and not validate_track(track):break
        else:raise ValueError('Fresh generation track search exhausted')
        tracks.append(replace(track,split='training'));used.add(track.geometry_hash)
    return tracks
