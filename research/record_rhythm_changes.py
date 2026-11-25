"""Record verified MuJoCo replays of every frozen Rhythm Changes episode."""
import argparse
import html
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from line_follower.config import RobotConfig
from line_follower.simulation import Simulation
from line_follower.tracks import TrackSpec, TrackMask
from line_follower.viewer import camera_for, add_overlays, set_transparency
from run_rhythm_changes_case_study import OUTPUT, verify, read, setup
from run_neat_fast_study import digest
from record_speed_comparisons import fix_video_index

FPS = 15
VLC = Path('C:/Program Files/VideoLAN/VLC/vlc.exe')
SHORTLIST = ('neat-s5-low-050__cap-050','pid-followup-150__cap-050',
             'neat-s3-fast-080__cap-100','pid-low-050__cap-100',
             'neat-s1-fast-150__cap-150','pid-followup-150__cap-150')


def catalog(destination,manifest,plan):
    warning = ('Review 23 Nov 2026: historical fastest-lap shortlist, requiring correction. '
               'All selected PID clips use base 0.75, so their centered straight targets are only 75% of the cap. '
               'The existing full-speed 1 m/s PID was omitted. NEAT also shows post-turn steering oscillation. '
               'These clips are not the final full-speed, smoothness-qualified comparison.')
    lines=['# Six selected Rhythm Changes videos','','One NEAT and one PID at each main evaluation cap: 0.50, 1.00 and 1.50 m/s. Selected retrospectively as the fastest successful run on this course; where none finished, greatest progress. These are course-specific showcases, not new validation winners. At 1.50 m/s no PID completed.','','Real MuJoCo replays, 1x playback, 15 fps, with approximately two seconds holding the final state. Each replay matches its saved termination, time, progress, mean/peak speed and jitter. Assessment results are unchanged.','','Open [the six-video gallery](index.html), or open any MP4 directly. A stalled result means no authenticated route progress, not stopped motors. Earlier surplus recordings are preserved but excluded from this shortlist.','','Resume or recreate missing shortlist clips: `& .\\.venv\\Scripts\\python.exe research/record_rhythm_changes.py`','','## Clips','']
    lines[2:2] = ['**'+warning+'**','']
    lines += [f'- [{key}]({manifest["clips"][key]["file"]}) — {manifest["clips"][key]["termination"]}' for key in SHORTLIST if key in manifest['clips']]
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=[]
    for task in plan['tasks']:
        if task['id'] not in SHORTLIST:continue
        clip=manifest['clips'].get(task['id'])
        if not clip:continue
        p=plan['policies'][task['policy']]
        heading=(f"NEAT seed {p['seed']} · g{p['generation']}" if p['family']=='NEAT' else task['policy'])
        detail=f"Cap {task['cap_m_s']:.2f} m/s · trained/tuned {p['native_cap_m_s']:.2f} · {clip['termination']}"
        if p['family']=='PID':
            base=p['parameters']['base']
            detail+=f" · base {base:.2f} · centered straight target {base*task['cap_m_s']:.3f} m/s"
        cards.append(f'<article data-key="{html.escape(task["id"])}"><h2>{html.escape(heading)}</h2><p>{html.escape(detail)}</p><video controls preload="none" src="{clip["file"]}"></video><p><a href="{clip["file"]}">{task["id"]}.mp4</a></p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Rhythm Changes video archive</title><style>body{font:16px system-ui;background:#f2f5f7;color:#183343;max-width:1500px;margin:30px auto;padding:20px}h1{margin-bottom:8px}input{padding:12px;width:min(90%,600px);font:inherit}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:22px;margin-top:25px}article{background:white;border-radius:10px;padding:16px}h2{font-size:18px}video{width:100%;background:#152937;aspect-ratio:16/9}a{overflow-wrap:anywhere;color:#007f86}</style>'''
    page+=f'<h1>Rhythm Changes · six-video shortlist ({len(cards)} ready)</h1><p>One NEAT and one filtered PID at 0.50, 1.00 and 1.50 m/s. Fastest successful saved run on this course at each cap; at 1.50 m/s, show the PID with greatest progress because none finished. This retrospective showcase does not replace validation selection.</p><p>Real MuJoCo render, live measured speed, full-course position. Playback is 1× at 15 fps; final state holds for about two seconds. “Stalled” means no ordered-route progress.</p><input id="filter" placeholder="Filter: cap-050, cap-100, cap-150, neat, pid" aria-label="Filter clips"><main>'+''.join(cards)+'</main><script>document.getElementById("filter").oninput=e=>document.querySelectorAll("article").forEach(a=>a.hidden=!a.dataset.key.includes(e.target.value.toLowerCase().trim()));</script></html>'
    page=page.replace('<main>','<aside style="padding:16px;background:#fff0cb;border:1px solid #bd8b25;margin:16px 0">'+html.escape(warning)+'</aside><main>')
    (destination/'index.html').write_text(page,encoding='utf-8')
    if all(key in manifest['clips'] for key in SHORTLIST):
        selected=destination/'selected-six'
        selected.mkdir(exist_ok=True)
        for key in SHORTLIST:
            file=manifest['clips'][key]['file']
            shutil.copy2(destination/file,selected/file)
        for file in ('index.html','README.md'):shutil.copy2(destination/file,selected/file)
        shortlist_manifest={'protocol_sha256':manifest['protocol_sha256'],
            'selection':'Fastest completed saved run per family and evaluation cap; greatest progress if none completed. Retrospective single-course showcase.',
            'evaluation_caps_m_s':[.5,1.,1.5],
            'clips':{key:manifest['clips'][key] for key in SHORTLIST}}
        (selected/'manifest.json').write_text(json.dumps(shortlist_manifest,indent=2),encoding='utf-8')


def capture(output, plan, task, mask):
    ep = setup(output, plan, task, mask)
    frames = []
    def frame():
        s = ep.simulation.state
        frames.append((ep.simulation.data.qpos.copy(), ep.simulation.sensor_hits.copy(),
                       ep.simulation.observation.copy(), s.position.copy(), s.time,
                       s.forward_speed, ep.route.fraction, ep.simulation.last_action.copy()))
    frame()
    while not ep.done:
        ep.advance()
        if ep.steps % 4 == 0 or ep.done: frame()
    actual = asdict(ep.result())
    expected = read(output/'episodes'/(task['id']+'.json'))
    for key in ('termination','completed','control_steps'):
        if actual[key] != expected[key]: raise ValueError(f'Replay differs: {task["id"]} {key}')
    for key in ('elapsed_s','progress','average_forward_speed_m_s','peak_forward_speed_m_s','command_jitter'):
        if abs(actual[key]-expected[key]) > 1e-8: raise ValueError(f'Replay differs: {task["id"]} {key}')
    return frames, expected


def record(output, plan, task, mask, visual, renderer, destination, *, course_label='Rhythm Changes'):
    frames, expected = capture(output, plan, task, mask)
    policy = plan['policies'][task['policy']]
    fonts = {n:ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',n) for n in (18,22,28)}
    backdrop = Image.new('RGB',(1280,720),'#f2f5f7')
    draw = ImageDraw.Draw(backdrop)
    title = policy.get('display_name') or (f"NEAT seed {policy['seed']} | generation {policy['generation']} | trained cap {policy['native_cap_m_s']:.2f} m/s"
             if policy['family']=='NEAT' else f"Filtered PID + remembered-side recovery | {task['policy']}")
    draw.text((20,12),title,font=fonts[28],fill='#183343')
    draw.text((20,52),f"{course_label} | evaluation wheel-speed cap {task['cap_m_s']:.2f} m/s | {task['id']}",font=fonts[18],fill='#445b68')
    # Fixed overview uses the exact frozen route; this is evaluator-only graphics.
    xy = mask.track.route.points
    lo,hi=xy.min(axis=0),xy.max(axis=0)
    scale=min(440/(hi[0]-lo[0]),245/(hi[1]-lo[1]))
    def project(point): return (int(790+(point[0]-lo[0])*scale),int(350-(point[1]-lo[1])*scale))
    draw.line([project(p) for p in xy[::3]],fill='#80939e',width=3)
    for p,color in ((xy[0],'#23835b'),(xy[-1],'#b34132')):
        x,y=project(p);draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
    draw.text((790,88),'Full-course position',font=fonts[22],fill='#183343')
    draw.text((20,674),'1x real time | 15 fps | MuJoCo replay checked against the saved episode | sensor dots: blue = line contact',font=fonts[18],fill='#445b68')
    camera=camera_for(visual,'track');camera.distance=1.35
    source=destination.with_suffix('.mjpg')
    frame_count=int(np.ceil(expected['elapsed_s']*FPS))+1
    with source.open('wb') as stream:
        for tick in range(frame_count+FPS*2):
            index=min(tick,len(frames)-1)
            qpos,hits,sensors,position,elapsed,speed,progress,action=frames[index]
            visual.data.qpos[:]=qpos
            mujoco.mj_forward(visual.model,visual.data)
            visual.sensor_hits[:]=hits;visual.observation[:]=sensors
            camera.lookat[:]=[position[0],position[1],0]
            renderer.update_scene(visual.data,camera)
            add_overlays(renderer.scene,visual,landmarks=False)
            canvas=backdrop.copy();canvas.paste(Image.fromarray(renderer.render()),(20,94))
            d=ImageDraw.Draw(canvas);x,y=project(position)
            d.ellipse((x-5,y-5,x+5,y+5),fill='#007f86' if policy['family']=='NEAT' else '#c45a28')
            lines=[f"Elapsed: {elapsed:.2f} s",f"Route progress: {progress:.1%}",f"Chassis speed: {speed:.3f} m/s",
                   f"Wheel commands: L {action[0]:+.2f}   R {action[1]:+.2f}",
                   f"Status: {expected['termination'] if index==len(frames)-1 else 'running'}",
                   f"Saved lap time: {expected['completion_time_s']:.2f} s" if expected['completed'] else f"No lap time: {expected['termination']}"]
            for i,line in enumerate(lines):d.text((790,388+i*36),line,font=fonts[22],fill='#183343')
            if not expected['completed'] and index==len(frames)-1:
                d.text((790,615),'Stall = no ordered-route progress' if expected['termination']=='stalled' else 'Off route: corridor + no-contact rule',font=fonts[18],fill='#a43f31')
            canvas.save(stream,'JPEG',quality=84,subsampling=0)
            if tick==min(3*FPS,frame_count-1) and not (destination.parent/'preview.png').exists():canvas.save(destination.parent/'preview.png')
    temporary=destination.with_suffix('.encoding.mp4')
    chain=f'#transcode{{vcodec=h264,venc=x264{{preset=veryfast,crf=21}},fps={FPS}}}:standard{{access=file,mux=mp4,dst="{temporary}"}}'
    try:
        subprocess.run([str(VLC),'-I','dummy','--no-one-instance','--ignore-config','--no-media-library',
            '--demux=mjpeg',f'--mjpeg-fps={FPS}','--no-audio','--no-sout-transcode-hurry-up',str(source),'--sout',chain,'vlc://quit'],
            check=True,timeout=300,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        count=fix_video_index(temporary)
        if abs(count-(frame_count+2*FPS))>3:raise ValueError(f'Unexpected encoded frame count: {count}')
        temporary.replace(destination)
    finally: source.unlink(missing_ok=True)
    return {'episode':task['id'],'file':destination.name,'sha256':digest(destination),'frames':count,
            'fps':FPS,'duration_s':count/FPS,'termination':expected['termination'],
            'replay_verified':True,'saved_episode_sha256':digest(output/'episodes'/(task['id']+'.json'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',nargs='*',help='Override the default six-video shortlist with explicit episode IDs')
    parser.add_argument('--shard',nargs=2,type=int,metavar=('INDEX','COUNT'),help='Record an independent partition; merge manifests after all partitions finish')
    args=parser.parse_args()
    plan=verify(OUTPUT);destination=OUTPUT/'videos';destination.mkdir(exist_ok=True)
    manifest_path=destination/(f'manifest-shard-{args.shard[0]}.json' if args.shard else 'manifest.json')
    manifest=read(manifest_path) if manifest_path.exists() else read(destination/'manifest.json') if (destination/'manifest.json').exists() else {'protocol_sha256':digest(OUTPUT/'protocol.json'),'clips':{}}
    if not args.shard:
        for partition in destination.glob('manifest-shard-*.json'):
            extra=read(partition)
            assert extra['protocol_sha256']==manifest['protocol_sha256']
            for key,value in extra['clips'].items():
                if key in manifest['clips']:assert value==manifest['clips'][key]
                manifest['clips'][key]=value
        temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2),encoding='utf-8');temp.replace(manifest_path)
    assert manifest['protocol_sha256']==digest(OUTPUT/'protocol.json')
    tasks=[t for t in plan['tasks'] if t['id'] in (args.episodes or SHORTLIST)]
    if args.episodes and set(args.episodes)!={t['id'] for t in tasks}:raise ValueError('Unknown episode id')
    if args.shard:
        index,count=args.shard
        if not 0<=index<count:raise ValueError('Invalid partition')
        tasks=tasks[index::count]
    pending=[]
    for task in tasks:
        old=manifest['clips'].get(task['id'])
        if old:
            assert digest(destination/old['file'])==old['sha256']
        else:pending.append(task)
    if not pending:
        catalog(destination,manifest,plan)
        print(f"Verified {len(tasks)} requested videos; six-video shortlist saved separately when complete")
        return
    mask=TrackMask(TrackSpec.load(OUTPUT/'track.json'))
    visual=Simulation(RobotConfig(**{**plan['robot'],'max_speed':1.}))
    visual.reset(mask.track,mask);visual.enable_rendering();set_transparency(visual,True)
    visual.model.vis.global_.offwidth=736;visual.model.vis.global_.offheight=560
    with mujoco.Renderer(visual.model,height=560,width=736) as renderer:
        for task in pending:
            start=time.perf_counter();print(f'Recording {task["id"]}',flush=True)
            record_row=record(OUTPUT,plan,task,mask,visual,renderer,destination/(task['id']+'.mp4'))
            manifest['clips'][task['id']]=record_row
            manifest['recorder_sha256']=digest(Path(__file__))
            temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2),encoding='utf-8');temp.replace(manifest_path)
            ready=sum(key in manifest['clips'] for key in SHORTLIST)
            print(f'Shortlist ready {ready}/6 | {time.perf_counter()-start:.1f}s wall',flush=True)
    catalog(destination,manifest,plan)


if __name__=='__main__':main()
