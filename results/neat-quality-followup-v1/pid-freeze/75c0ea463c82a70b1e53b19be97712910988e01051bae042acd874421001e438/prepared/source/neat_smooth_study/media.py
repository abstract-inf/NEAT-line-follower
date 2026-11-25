"""Only explicitly requested, validated controller pairs are rendered."""
from dataclasses import asdict
from pathlib import Path
import struct
import subprocess

import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from line_follower.config import RobotConfig,EpisodeConfig
from line_follower.controllers import load_model
from line_follower.episodes import Episode
from line_follower.pid import ResearchPIDConfig,ResearchPIDController
from line_follower.simulation import Simulation
from line_follower.tracks import TrackSpec,TrackMask
from line_follower.viewer import camera_for,add_overlays,set_transparency
from .common import read,write_json,digest
from .engine import evaluate_tasks


def boxes(data):
    offset=0
    while offset<len(data):
        size,kind=struct.unpack_from('>I4s',data,offset)
        if size<8 or offset+size>len(data):raise ValueError('Unsupported MP4 layout')
        yield kind,data[offset+8:offset+size]
        offset+=size


def index_mp4(path):
    """Repair VLC's H.264 sync-sample table so playback can seek from frame one."""
    data=path.read_bytes();table=dict(boxes(data))[b'moov']
    for kind in (b'trak',b'mdia',b'minf',b'stbl'):table=dict(boxes(table))[kind]
    tables=dict(boxes(table));sizes=tables[b'stsz'];count=struct.unpack_from('>I',sizes,8)[0]
    if struct.unpack_from('>I',sizes,4)[0]!=0:raise ValueError('Unexpected fixed-size video samples')
    lengths=struct.unpack_from(f'>{count}I',sizes,12);offsets=tables[b'stco']
    if struct.unpack_from('>I',offsets,4)[0]!=1:raise ValueError('Unexpected video chunk layout')
    offset=struct.unpack_from('>I',offsets,8)[0];sync=[]
    for frame,length in enumerate(lengths,1):
        end=offset+length
        while offset<end:
            size=struct.unpack_from('>I',data,offset)[0]
            if size<=0 or offset+4+size>end:raise ValueError('Invalid H.264 sample')
            if data[offset+4]&31==5 and (not sync or sync[-1]!=frame):sync.append(frame)
            offset+=4+size
        if offset!=end:raise ValueError('Invalid video index')
    if not sync or sync[0]!=1:raise ValueError('Missing initial H.264 keyframe')
    payload=struct.pack(f'>{len(sync)+2}I',0,len(sync),*sync)
    def box(kind,value):return struct.pack('>I4s',len(value)+8,kind)+value
    def rebuild(content,shift):
        output=[]
        for kind,value in boxes(content):
            if kind in (b'moov',b'trak',b'mdia',b'minf',b'stbl'):
                value=rebuild(value,shift)
                if kind==b'stbl' and b'stss' not in dict(boxes(value)):value+=box(b'stss',payload)
            elif kind==b'stss':value=payload
            elif kind==b'stco':
                n=struct.unpack_from('>I',value,4)[0];pointers=struct.unpack_from(f'>{n}I',value,8)
                value=value[:8]+struct.pack(f'>{n}I',*(p+shift for p in pointers))
            output.append(box(kind,value))
        return b''.join(output)
    if next(k for k,_ in boxes(data) if k in (b'moov',b'mdat'))!=b'moov':raise ValueError('Unexpected MP4 order')
    shift=len(rebuild(data,0))-len(data)
    temporary=path.with_suffix('.indexed.mp4');temporary.write_bytes(rebuild(data,shift));temporary.replace(path)
    return count


def record_pair(run,manifest,names,cap,directory):
    vlc=Path('C:/Program Files/VideoLAN/VLC/vlc.exe')
    if not vlc.is_file():raise ValueError('Recording requires the existing VLC installation; no software is installed automatically')
    track_path=run/'prepared/inputs/showcase-track.json'
    track=TrackSpec.load(track_path);mask=TrackMask(track)
    plan=read(run/'prepared/protocol.json');robot=RobotConfig(**{**plan['robot'],'max_speed':cap})
    visual=Simulation(robot);visual.reset(track,mask);visual.enable_rendering();set_transparency(visual,True)
    visual.model.vis.global_.offwidth=736;visual.model.vis.global_.offheight=560
    camera=camera_for(visual,'track');camera.distance=1.35
    fonts={n:ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',n) for n in (18,22,26)}
    freeze_sha=digest(run/'assessment/frozen/manifest.json')
    videos=[]
    with mujoco.Renderer(visual.model,height=560,width=736) as renderer:
        for name in names:
            source=run/'assessment/frozen'/manifest['policies'][name]['file']
            target=directory/(name+'.mp4');metadata_path=target.with_suffix('.json')
            if metadata_path.exists():
                record=read(metadata_path)
                if record['controller_sha256']!=digest(source) or record['sha256']!=digest(target) or record['freeze_sha256']!=freeze_sha:
                    raise ValueError('An existing showcase was altered')
                videos.append(record);continue
            policy={'parameters':read(source)} if source.suffix=='.json' else {'model':str(source)}
            expected=evaluate_tasks(run,[{**policy,'paths':[str(track_path)],'cap':cap,'detailed':True}],workers=1)[0][0]
            sim=Simulation(robot)
            controller=(ResearchPIDController(ResearchPIDConfig.load(source),robot) if source.suffix=='.json' else load_model(source)[0])
            episode=Episode(sim,track,controller,EpisodeConfig(**plan['episode']),mask)
            frames=[]
            def capture():
                s=sim.state
                frames.append((sim.data.qpos.copy(),sim.sensor_hits.copy(),sim.observation.copy(),s.position.copy(),
                    s.time,s.forward_speed,episode.route.fraction,sim.last_action.copy()))
            capture()
            while not episode.done:
                episode.advance()
                if episode.steps%4==0 or episode.done:capture()
            actual=asdict(episode.result())
            if any(actual[k]!=expected[k] for k in ('completed','termination','control_steps')) or any(
                abs(actual[k]-expected[k])>1e-8 for k in ('elapsed_s','progress','average_forward_speed_m_s','command_jitter')):
                raise ValueError('Rendered replay differs from the headless episode')
            raw=target.with_suffix('.mjpg')
            with raw.open('wb') as stream:
                for i in range(len(frames)+30):
                    qpos,hits,sensors,position,elapsed,speed,progress,action=frames[min(i,len(frames)-1)]
                    visual.data.qpos[:]=qpos;mujoco.mj_forward(visual.model,visual.data)
                    visual.sensor_hits[:]=hits;visual.observation[:]=sensors;camera.lookat[:]=[*position[:2],0]
                    renderer.update_scene(visual.data,camera);add_overlays(renderer.scene,visual,landmarks=False)
                    canvas=Image.new('RGB',(1280,720),'#f2f5f7');draw=ImageDraw.Draw(canvas)
                    draw.text((20,12),f'{name} | evaluation cap {cap:.2f} m/s',font=fonts[26],fill='#183343')
                    draw.text((20,50),'Validation-selected frozen controller | Rhythm Changes exploratory replay',font=fonts[18],fill='#445b68')
                    canvas.paste(Image.fromarray(renderer.render()),(20,94))
                    lines=[f'Elapsed {elapsed:.2f} s',f'Progress {progress:.1%}',f'Chassis speed {speed:.3f} m/s',
                           f'L {action[0]:+.2f}   R {action[1]:+.2f}',f"Final result: {expected['termination']}"]
                    for j,line in enumerate(lines):draw.text((785,160+j*45),line,font=fonts[22],fill='#183343')
                    if source.suffix=='.json':draw.text((785,430),'PID base 1.00',font=fonts[22],fill='#183343')
                    draw.text((20,674),'1x playback | 15 fps | live measured chassis speed | stall means no ordered-route progress',font=fonts[18],fill='#445b68')
                    canvas.save(stream,'JPEG',quality=84,subsampling=0)
            temporary=target.with_suffix('.encoding.mp4')
            chain=f'#transcode{{vcodec=h264,venc=x264{{preset=veryfast,crf=21}},fps=15}}:standard{{access=file,mux=mp4,dst="{temporary}"}}'
            try:
                subprocess.run([str(vlc),'-I','dummy','--no-one-instance','--ignore-config','--no-media-library','--demux=mjpeg',
                    '--mjpeg-fps=15','--no-audio','--no-sout-transcode-hurry-up',str(raw),'--sout',chain,'vlc://quit'],
                    check=True,timeout=300,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                count=index_mp4(temporary)
                if abs(count-(len(frames)+30))>3:raise ValueError('Unexpected video frame count')
                temporary.replace(target)
            finally:raw.unlink(missing_ok=True)
            record={'policy':name,'file':target.name,'sha256':digest(target),'controller_sha256':digest(source),
                    'freeze_sha256':freeze_sha,'fps':15,'frames':count,'cap_m_s':cap,'replay_verified':True,'result':expected}
            write_json(metadata_path,record);videos.append(record)
    write_json(directory/'manifest.json',{'selection':'Declared validation ranking, never this course lap time','videos':videos})
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Validated controller pair</title>'+
        '<h1>Validation-selected controller pair</h1><p>Known exploratory course; 1x playback, 15 fps, two-second final hold.</p>'+
        ''.join(f'<h2>{r["policy"]} · {cap:.2f} m/s cap</h2><video style="width:90%;max-width:1280px" controls preload="none" src="{r["file"]}"></video>' for r in videos),encoding='utf-8')
