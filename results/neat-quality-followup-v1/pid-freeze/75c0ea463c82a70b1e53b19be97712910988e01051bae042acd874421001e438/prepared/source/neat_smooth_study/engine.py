"""Unchanged MuJoCo/NEAT evaluation with versioned scoring and atomic generations."""
from collections import OrderedDict
from dataclasses import asdict, replace
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np

from line_follower.config import RobotConfig, EpisodeConfig
from line_follower.controllers import NEATController, load_model, read_model_artifact, save_model
from line_follower.episodes import Episode, TELEMETRY_FIELDS
from line_follower.experiments import warm_start_population, _save_checkpoint, _restore_logs
from line_follower.fitness import aggregate_fitness, fitness_components
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.pid_experiments import geometric_tracking_error
from line_follower.tracks import TrackMask, TrackSpec
from .common import SETTINGS, REWARDS, read, digest, write_json, append_csv, lock, entry_path, epoch_seed, verify
from .courses import generation_batch
from .metrics import straight_windows, window_metrics, added_costs, quality, select


class Context:
    def __init__(self,run,parent=None):
        self.run=Path(run)
        self.plan=read(self.run/'prepared/protocol.json')
        self.neat_config=read_model_artifact(parent)['config'] if parent else None
        self.simulations={}
        self.masks=OrderedDict()

    def mask(self,track):
        key=digest_track(track)
        if key not in self.masks:
            estimate=2*int(np.ceil(track.arena_size/track.resolution))**2
            while self.masks and sum(m.field.nbytes+m.pixels*m.pixels for m in self.masks.values())+estimate>SETTINGS['mask_cache_bytes']:
                _,old=self.masks.popitem(last=False)
                for sim in self.simulations.values():
                    if sim.mask is old:sim.mask=None
            mask=TrackMask(track)
            if estimate<=SETTINGS['mask_cache_bytes']:self.masks[key]=mask
            else:return mask
        self.masks.move_to_end(key)
        return self.masks[key]

    def evaluate(self,task):
        from line_follower.simulation import Simulation
        cap=task.get('cap',1.)
        if cap not in self.simulations:
            self.simulations[cap]=Simulation(RobotConfig(**{**self.plan['robot'],'max_speed':cap}))
        sim=self.simulations[cap]
        if task.get('genome') is not None:
            controller=NEATController(task['genome'],self.neat_config)
        elif task.get('model'):
            controller=load_model(task['model'])[0]
        else:
            controller=ResearchPIDController(ResearchPIDConfig(**task['parameters']),sim.config)
        records=[]
        for slot,path in enumerate(task['paths']):
            track=TrackSpec.load(path)
            mask=self.mask(track)
            episode=Episode(sim,track,controller,EpisodeConfig(**self.plan['episode']),mask)
            windows=straight_windows(track)
            samples=[[] for _ in windows]
            observed_ends=[False]*len(windows)
            positions=[]
            trace=None
            if task.get('telemetry'):
                import csv
                target=Path(task['telemetry'])/f'{slot:03d}.csv.gz'
                target.parent.mkdir(parents=True,exist_ok=True)
                trace=gzip.open(target,'wt',encoding='utf-8',newline='')
                writer=csv.DictWriter(trace,fieldnames=TELEMETRY_FIELDS);writer.writeheader()
            try:
                while not episode.done:
                    row=episode.advance(telemetry=bool(windows) or trace is not None)
                    state=sim.state
                    if task.get('detailed'):positions.append(state.position[:2].copy())
                    if trace is not None and row:writer.writerow(row)
                    for i,(lower,upper) in enumerate(windows):
                        if lower<=episode.route.distance<upper:
                            samples[i].append((state.forward_speed,state.yaw_rate,float((sim.last_action[0]-sim.last_action[1])/2)))
                        if episode.route.distance>=upper:observed_ends[i]=True
            finally:
                if trace is not None:trace.close()
            record=asdict(episode.result())
            if record['error']:raise RuntimeError(record['error'])
            if positions:
                record['geometry_lateral_rmse_m']=geometric_tracking_error(np.asarray(positions),track.route.points)[0]
            measured=[window_metrics(s,end) for s,end in zip(samples,observed_ends)]
            base=fitness_components(record['progress'],record['completed'],record['elapsed_s'],record['time_limit_s'],
                record['legacy_return'],record['control_steps'],.5,record['command_jitter'],.2,record['route_length_m'],'route-speed-1.0')
            extra=added_costs(measured,record['lateral_rmse_m'])
            branch=task.get('branch','old')
            record.update(base_score=record['score'],straight_windows=measured,
                          components={**base,**{k:(v if branch=='smooth' else 0.) for k,v in extra.items()}},
                          cap_m_s=cap,group=task.get('groups',{}).get(track.geometry_hash,'training'))
            if branch=='smooth':record['score']+=sum(extra.values())
            records.append(record)
        return records


def digest_track(track):
    import hashlib
    return hashlib.sha256(json.dumps(asdict(track),sort_keys=True).encode()).hexdigest()


_CONTEXT=None


def initialize(run,parent):
    global _CONTEXT
    _CONTEXT=Context(run,parent)


def worker(task):
    return _CONTEXT.evaluate(task)


def evaluate_tasks(run,tasks,workers=2,parent=None):
    if not tasks:return []
    if workers==1:
        context=Context(run,parent)
        return [context.evaluate(t) for t in tasks]
    with mp.get_context('spawn').Pool(workers,initialize,(str(run),str(parent) if parent else None),
                                      maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
        return collect(pool,tasks,'evaluation')


def collect(pool,tasks,label):
    output=[]
    iterator=pool.imap(worker,tasks,chunksize=1)
    started=last=time.perf_counter()
    total=sum(len(t['paths']) for t in tasks)
    def status():
        print(f'{label}: episodes {sum(len(rows) for rows in output)}/{total}; wall {time.perf_counter()-started:.1f}s',flush=True)
    while len(output)<len(tasks):
        try:output.append(iterator.next(timeout=20))
        except mp.TimeoutError:
            status()
            continue
        if time.perf_counter()-last>20 or len(output)==len(tasks):
            status();last=time.perf_counter()
    return output


def validation_task(run,phase,split,**policy):
    entries=read(Path(run)/'prepared'/phase/'suite/suite.json')['splits'][split]
    return {**policy,'paths':[str(entry_path(run,phase,e)) for e in entries],
            'groups':{e['geometry_hash']:e['group'] for e in entries},'detailed':True,
            'cap':.5 if split=='retention' else policy.get('cap',1.)}


def references(run,phase,seed,workers):
    path=Path(run)/phase/'references'/f'seed-{seed:02d}.json'
    parent=Path(run)/'prepared/inputs'/f'seed-{seed:02d}.pkl'
    if path.exists():
        saved=read(path)
        if saved['parent_sha256']!=digest(parent) or saved['protocol_sha256']!=digest(Path(run)/'prepared/protocol.json'):
            raise ValueError('Stale quality reference')
        return saved
    tasks=[validation_task(run,phase,split,model=str(parent)) for split in ('validation','retention')]
    primary,low=evaluate_tasks(run,tasks,workers)
    record={'parent_sha256':digest(parent),'protocol_sha256':digest(Path(run)/'prepared/protocol.json'),
            'primary':primary,'retention':low}
    write_json(path,record)
    return record


def train(run,phase,seed,branch,workers=2,resume=False,stop_after=None):
    """One library population.run per generation; generation N+1 commits atomically."""
    run=Path(run)
    plan=verify(run)
    directory=run/phase/f'seed-{seed:02d}'/branch
    parent=run/'prepared/inputs'/f'seed-{seed:02d}.pkl'
    signature={'protocol_sha256':digest(run/'prepared/protocol.json'),'phase':phase,'seed':seed,'branch':branch,
               'parent_sha256':digest(parent),'settings':SETTINGS,'evolution_seed':epoch_seed(phase,seed)}
    with lock(directory):
        reference=references(run,phase,seed,workers)
        signature['reference_sha256']=digest(run/phase/'references'/f'seed-{seed:02d}.json')
        checkpoint=directory/'checkpoints/latest.pkl.gz'
        if resume:
            with gzip.open(checkpoint,'rb') as stream:saved=pickle.load(stream)
            if saved['signature']!=signature:raise ValueError('Resume inputs changed')
            population=saved['population'];random.setstate(saved['random_state'])
            seen=saved['seen_hashes'];total=saved['total_training_episodes']
            _restore_logs(directory,population.generation)
            for path in (directory/'validation').glob('generation-*.json'):
                if int(path.stem.split('-')[-1])>=population.generation:path.unlink()
            if saved['best_model_bytes'] is None:(directory/'best.pkl').unlink(missing_ok=True)
            else:(directory/'best.pkl').write_bytes(saved['best_model_bytes'])
            (directory/'complete.json').unlink(missing_ok=True)
        else:
            if any(p.name!='session.lock' for p in directory.iterdir()):raise FileExistsError('Use --resume for an interrupted stage')
            random.seed(signature['evolution_seed'])
            population=warm_start_population(parent,SETTINGS['population'])
            seen=set();total=0
            write_json(directory/'run.json',{**signature,'runtime_workers':workers,'source':'prepared/source',
                'reward_version':REWARDS[branch],'claim_scope':'Warm-started controller comparison; actual episode counts are logged'})
        entries=read(run/'prepared'/phase/'suite/suite.json')['splits']['training']
        from datetime import datetime,timezone
        runtime={'started_utc':datetime.now(timezone.utc).isoformat(),'workers':workers,'resume':resume,
                 'next_generation':population.generation}
        append_csv(directory/'runtime.csv',[runtime],list(runtime))
        bank_entries=sorted(entries,key=lambda e:(e['kind']!='isolated',e['path']))
        bank=[TrackSpec.load(entry_path(run,phase,e)) for e in bank_entries]
        forbidden=set(plan['protected_geometry'])
        for namespace in ('pilot','confirmation','pid'):
            manifest=read(run/'prepared'/namespace/'suite/suite.json')
            forbidden.update(e['geometry_hash'] for es in manifest['splits'].values() for e in es)
        # A speed bank may be shared by the two branches, not by fresh mixed tracks.
        (directory/'checkpoints').mkdir(exist_ok=True)
        (directory/'validation').mkdir(exist_ok=True)
        generations_this_call=0
        with mp.get_context('spawn').Pool(workers,initialize,(str(run),str(parent)),
                                          maxtasksperchild=SETTINGS['worker_recycle_tasks']) as pool:
            while population.generation<SETTINGS['generations']:
                generation=population.generation;started=time.perf_counter()
                tracks=generation_batch(signature['evolution_seed'],generation,bank,forbidden|seen)
                paths=[]
                for i,track in enumerate(tracks):
                    path=directory/'tracks'/f'generation-{generation:04d}'/f'{i}.json';track.save(path);paths.append(str(path))
                hashes=[t.geometry_hash for t in tracks];seen.update(hashes)
                write_json(directory/'tracks'/f'generation-{generation:04d}'/'batch.json',{'hashes':hashes,'paths':paths})
                stats={}
                def evaluate_generation(genomes,config):
                    tasks=[{'genome':g,'paths':paths,'branch':branch} for _,g in genomes]
                    evaluated=collect(pool,tasks,f'generation {generation} training')
                    for (key,genome),episodes in zip(genomes,evaluated):
                        genome.fitness=aggregate_fitness([r['score'] for r in episodes],'mean_cvar',.5)
                    champion=max((g for _,g in genomes),key=lambda g:g.fitness)
                    held=collect(pool,[validation_task(run,phase,s,genome=champion,branch=branch)
                                       for s in ('validation','retention')],f'generation {generation} validation')
                    gate=quality(held[0],held[1],reference['primary'],reference['retention'])
                    append_csv(directory/'training-episodes.csv',[{'generation':generation,'genome_id':g.key,
                        **{k:v for k,v in r.items() if k not in ('components','straight_windows')}}
                        for (_,g),episodes in zip(genomes,evaluated) for r in episodes],
                        ['generation','genome_id']+[k for k in evaluated[0][0] if k not in ('components','straight_windows')])
                    for role,records in (('training',[r for es in evaluated for r in es]),('validation',held[0])):
                        parts={k:float(np.mean([r['components'][k] for r in records])) for k in records[0]['components']}
                        append_csv(directory/'fitness-components.csv',[{'generation':generation,'role':role,**parts}],['generation','role',*parts])
                    active={n[0] for n in NEATController(champion,config).network.node_evals}
                    outputs=config.genome_config.output_keys
                    gate.update(generation=generation,branch=branch,drive_evaluated=outputs[0] in active,steering_evaluated=outputs[1] in active)
                    quality_row={**gate,'reasons':';'.join(gate['reasons'])}
                    append_csv(directory/'quality.csv',[quality_row],list(quality_row))
                    write_json(directory/'validation'/f'generation-{generation:04d}.json',{'quality':gate,'primary':held[0],'retention':held[1]})
                    stats.update(champion=champion,gate=gate,episodes=len(genomes)*6,
                        training_mean=mean_fitness(genomes),training_best=champion.fitness,
                        validation_completion_rate=np.mean([r['completed'] for r in held[0]]),
                        validation_score=float(np.mean([r['score'] for r in held[0]])))
                species_sizes={k:len(s.members) for k,s in population.species.species.items()}
                population.run(evaluate_generation,1)
                champion_path=directory/'champions'/f'generation-{generation:04d}.pkl'
                champion_path.parent.mkdir(parents=True,exist_ok=True)
                save_model(champion_path,stats['champion'],population.config,{'generation':generation,'parent':str(parent),
                    'parent_sha256':digest(parent),'evolution_seed':signature['evolution_seed'],'fitness_version':REWARDS[branch],
                    'robot':{**plan['robot'],'max_speed':1.},'episode':plan['episode'],'condition':f'smooth-study-{branch}'})
                candidates=[read(p)['quality'] for p in sorted((directory/'validation').glob('generation-*.json'))]
                winner=select(candidates)
                if winner:
                    shutil.copy2(directory/'champions'/f"generation-{winner['generation']:04d}.pkl",directory/'best.pkl')
                    write_json(directory/'selection.json',winner)
                total+=stats['episodes']
                row={'generation':generation,'training_mean':stats['training_mean'],'training_best':stats['training_best'],
                     'validation_completion_rate':float(stats['validation_completion_rate']),'validation_score':stats['validation_score'],
                     'training_episodes':stats['episodes'],'total_training_episodes':total,'validation_episodes':80,
                     'wall_seconds':time.perf_counter()-started,'eligible':stats['gate']['eligible']}
                append_csv(directory/'generations.csv',[row],list(row))
                append_csv(directory/'evolution.csv',[{'generation':generation,'species_sizes':json.dumps(species_sizes),
                    'champion_id':stats['champion'].key,'champion_nodes':stats['champion'].size()[0],
                    'champion_enabled_connections':stats['champion'].size()[1]}],
                    ['generation','species_sizes','champion_id','champion_nodes','champion_enabled_connections'])
                _save_checkpoint(checkpoint,{'checkpoint_version':2,'neat_version':'0.92','signature':signature,'population':population,
                    'random_state':random.getstate(),'seen_hashes':seen,'total_training_episodes':total,
                    'best_model_bytes':(directory/'best.pkl').read_bytes() if (directory/'best.pkl').exists() else None})
                print(f"generation={generation} completed; eligible={stats['gate']['eligible']}; reasons={stats['gate']['reasons']}",flush=True)
                generations_this_call+=1
                if stop_after and generations_this_call>=stop_after:break
        if population.generation==SETTINGS['generations']:
            write_json(directory/'complete.json',{'generations':population.generation,'signature':signature,
                'training_episodes':total,'qualified':(directory/'best.pkl').exists(),
                'best_sha256':digest(directory/'best.pkl') if (directory/'best.pkl').exists() else None})
        return directory


def mean_fitness(genomes):
    return float(np.mean([g.fitness for _,g in genomes]))
