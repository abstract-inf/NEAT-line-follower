"""Audit the six-video selection and straight-line motion from frozen evidence."""
import argparse
from dataclasses import replace
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from line_follower.config import RobotConfig
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.tracks import TrackSpec
from run_rhythm_changes_case_study import OUTPUT, verify, read, geometric_errors
from run_neat_fast_study import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-full-speed-pids', action='store_true')
    args = parser.parse_args()
    plan = verify(OUTPUT)
    manifest = read(OUTPUT/'videos/selected-six/manifest.json')
    out = OUTPUT/'video-review-20261123'
    out.mkdir(exist_ok=True)
    track = TrackSpec.load(OUTPUT/'track.json')
    names = list(manifest['clips']) + ['pid-fullspeed-100__cap-100']
    checks, metrics, traces = [], [], {}
    for name in names:
        result = read(OUTPUT/'episodes'/(name+'.json'))
        policy = plan['policies'][result['policy']]
        if policy['family'] == 'PID':
            robot = RobotConfig(**{**plan['robot'], 'max_speed': result['cap_m_s']})
            config = ResearchPIDConfig.load(OUTPUT/policy['file'])
            controller = ResearchPIDController(config, robot)
            sensors = np.zeros(15); sensors[7] = 1
            for _ in range(120): command = controller.act(sensors, robot.dt)
            target = float(np.mean(command))*robot.max_speed
            checks.append({'episode': name, 'base':config.base, 'centered_commands':command.tolist(),
                           'nominal_straight_target_m_s':target, 'evaluation_cap_m_s':robot.max_speed,
                           'full_cap_target_feasible':abs(target-robot.max_speed)<1e-12,
                           'in_shortlist':name in manifest['clips']})
        with gzip.open(OUTPUT/result['telemetry'], 'rt', encoding='utf-8') as f:
            records = list(csv.DictReader(f))
        data = {k:np.array([float(r[k]) for r in records]) for k in
                ('time_s','progress','x_m','y_m','action_L','action_R','yaw_rate_rad_s','forward_speed_m_s')}
        traces[name] = data
        for block in plan['course']['blocks']:
            if block['name'] not in ('Opening straight','Row 2 back straight','Row 3 back straight','Finish straight'):continue
            # Exclude 0.8 m at both ends to avoid intentional braking/turning.
            use = (data['progress']*track.route.length > block['start_m']+.8) & (data['progress']*track.route.length < block['end_m']-.8)
            if not np.any(use): continue
            t = data['time_s'][use]
            turn = (data['action_L'][use]-data['action_R'][use])/2
            nonzero = turn[np.abs(turn)>.02]
            error = geometric_errors(np.column_stack((data['x_m'][use],data['y_m'][use])),track.route.points)
            metrics.append({'episode':name,'section':block['name'],'from_s':float(t[0]),'to_s':float(t[-1]),
                'samples':int(use.sum()),'mean_speed_m_s':float(data['forward_speed_m_s'][use].mean()),
                'turn_min':float(turn.min()),'turn_max':float(turn.max()),
                'turn_rms':float(np.sqrt(np.mean(turn**2))),
                'yaw_rms_rad_s':float(np.sqrt(np.mean(data['yaw_rate_rad_s'][use]**2))),
                'steering_sign_reversals_above_002':int(np.sum(nonzero[1:]*nonzero[:-1]<0)),
                'geometric_rmse_mm':float(np.sqrt(np.mean(error**2))*1000)})
    evidence={'protocol_sha256':digest(OUTPUT/'protocol.json'),'video_manifest_sha256':digest(OUTPUT/'videos/selected-six/manifest.json'),
              'centered_pid_checks':checks,'straight_metrics':metrics,
              'definitions':'60 Hz telemetry; 0.8 m straight-end exclusions. Reversals count sign changes after removing normalized turn commands of magnitude <=0.02. Yaw RMS measures physical chassis angular velocity; geometric RMSE is nearest-polyline error.'}
    (out/'audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    with (out/'straight-metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    neat=[n for n in names if n.startswith('neat')]
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    for row,name in enumerate(neat):
        m=next(m for m in metrics if m['episode']==name and m['section']=='Row 2 back straight')
        d=traces[name];use=(d['time_s']>=m['from_s'])&(d['time_s']<=m['to_s']);t=d['time_s'][use]-m['from_s']
        axes[row,0].plot(t,d['action_L'][use],label='Left wheel',lw=1)
        axes[row,0].plot(t,d['action_R'][use],label='Right wheel',lw=1)
        axes[row,0].set(title=name,ylabel='Normalized wheel command')
        axes[row,1].plot(t,d['yaw_rate_rad_s'][use],color='#007f86',lw=1)
        axes[row,1].set(title=f"{m['steering_sign_reversals_above_002']} steering reversals; {m['geometric_rmse_mm']:.2f} mm tracking RMSE",ylabel='Chassis yaw rate (rad/s)')
    for ax in axes.flat: ax.grid(alpha=.2);ax.set_xlabel('Seconds within the central back straight')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recorded NEAT steering after the chicane · full 60 Hz data, no smoothing',fontsize=14)
    for suffix in ('png','svg'):fig.savefig(out/f'straight-command-and-yaw-oscillation.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(checks,indent=2))
    print('Saved exact straight-motion metrics and raw-command/yaw plots.')
    if args.require_full_speed_pids:
        failed=[r['episode'] for r in checks if r['in_shortlist'] and not r['full_cap_target_feasible']]
        if failed:
            print('FAIL: selected PID configurations cannot command the full centered straight target:',', '.join(failed))
            raise SystemExit(1)


if __name__=='__main__':main()
