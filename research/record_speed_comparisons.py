"""Record synchronized top-down MuJoCo replays of the frozen speed comparison."""

import argparse
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_speed_comparison import OUT, ROOT, RUN, controller_for, digest, read_csv, read_json
from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.episodes import run_episode as evaluate_episode
from line_follower.experiments import append_csv, write_json
from line_follower.fitness import FITNESS_VERSION
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask, TrackSpec
from line_follower.viewer import _line, add_overlays, camera_for

TRACKS = (0, 1, 31, 32, 17)
CAPS = (1.50, .50)
POLICIES = {
    "speed-trained": ("NEAT g98", "#147867"),
    "pid-filtered": ("Filtered PID + recovery", "#a2670d"),
    "pid-plain": ("Plain PID", "#a84e39"),
}
FPS = 30
PANEL = 480
SIZE = (1536, 720)
PAPER, INK, MUTED = "#f1f3ee", "#172d2d", "#4d6260"
DEST = ROOT / "research/artifacts/neat-speed-videos"


@dataclass
class Frame:
    qpos: np.ndarray
    xyz: np.ndarray
    hits: np.ndarray
    sensors: np.ndarray
    time: float
    speed: float
    progress: float
    termination: str


def capture(episode):
    sim = episode.simulation
    state = sim.state
    return Frame(sim.data.qpos.copy(), state.position, sim.sensor_hits.copy(),
                 sim.observation.copy(), state.time, state.forward_speed,
                 episode.route.fraction, episode.termination)


def run_episode(sim, mask, policy, model, settings, episode_config, expected, telemetry_path):
    controller = controller_for(policy, model, json.dumps(settings, sort_keys=True), sim.config)
    frames = []
    stride = sim.config.control_hz // FPS
    assert stride * FPS == sim.config.control_hz

    def observe(episode, final):
        if episode.steps % stride == 0 or final:
            frames.append(capture(episode))

    result = evaluate_episode(sim, mask.track, controller, episode_config, mask,
                              telemetry_path=telemetry_path, observer=observe)
    if result.error or result.track_hash != expected["track_hash"]:
        raise ValueError(f"Replay failed: {policy} on {result.track_name} at {sim.config.max_speed}")
    return frames, result


def fixed_camera(sim, runs):
    # Include all six trajectories and keep one framing for both speeds.
    positions = np.concatenate([sim.track.route.points, np.array(
        [f.xyz[:2] for frames in runs.values() for f in frames])])
    low, high = positions.min(axis=0), positions.max(axis=0)
    span = max(1.15, float(max(high-low)) + .40)
    center = (low+high)/2
    if span <= sim.track.arena_size:
        limit = (sim.track.arena_size-span)/2
        center = np.clip(center, -limit, limit)
    camera = camera_for(sim, "track")
    camera.lookat[:2] = center
    camera.distance = span/(2*math.tan(math.radians(sim.model.vis.global_.fovy/2)))
    return camera


def scene_frame(renderer, sim, camera, frames, index, color):
    frame = frames[index]
    sim.data.qpos[:] = frame.qpos
    mujoco.mj_forward(sim.model, sim.data)
    sim.sensor_hits[:] = frame.hits
    sim.observation[:] = frame.sensors
    renderer.update_scene(sim.data, camera)
    add_overlays(renderer.scene, sim)
    tail = frames[max(0, index-22):index+1:2]
    rgba = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)] + [.85]
    for a, b in zip(tail, tail[1:]):
        _line(renderer.scene, np.r_[a.xyz[:2], .002], np.r_[b.xyz[:2], .002], rgba, .0028)
    return Image.fromarray(renderer.render())


def load_fonts():
    fonts = Path("C:/Windows/Fonts")
    return {size: ImageFont.truetype(str(fonts / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)
            for size, bold in ((16, False), (17, True), (19, False), (22, True), (28, True), (32, True))}


def dashboard(track, slot, cap, fonts):
    canvas = Image.new("RGB", SIZE, PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 14), f"MUJOCO  /  TRACK {slot:03d}", font=fonts[17], fill="#147867")
    draw.text((32, 35), f"{track.difficulty.title()} course", font=fonts[32], fill=INK)
    features = "  ·  ".join(f.replace("_", " ") for f in track.features)
    draw.text((32, 77), features, font=fonts[16], fill=MUTED)
    draw.text((1504, 16), f"{cap:.2f} m/s wheel speed cap", anchor="rt", font=fonts[28], fill=INK)
    draw.text((1504, 54), "TOP VIEW  /  1× SIMULATION TIME", anchor="rt", font=fonts[17], fill=MUTED)
    for i, (name, color) in enumerate(POLICIES.values()):
        x = 32 + i*496
        draw.rounded_rectangle((x, 109, x+PANEL, 674), radius=8, fill="#ffffff", outline="#d4dfd8")
        draw.rounded_rectangle((x+14, 122, x+24, 142), radius=4, fill=color)
        draw.text((x+35, 116), name, font=fonts[22], fill=INK)
    draw.text((32, 688), "Blue sensor dots mark detections. Each panel freezes when its episode ends.",
              font=fonts[16], fill=MUTED)
    return canvas


def draw_frame(backdrop, panels, frames, time_s, fonts):
    canvas = backdrop.copy()
    draw = ImageDraw.Draw(canvas)
    draw.text((1504, 78), f"SIM CLOCK   {time_s:05.2f} s", anchor="rt", font=fonts[17], fill=INK)
    for i, ((_, color), panel, frame) in enumerate(zip(POLICIES.values(), panels, frames)):
        x = 32+i*496
        canvas.paste(panel, (x, 151))
        status = frame.termination.replace("_", " ").upper() or "RUNNING"
        status_color = "#147867" if status == "COMPLETED" else "#a84e39" if frame.termination else MUTED
        draw.text((x+14, 638), status, font=fonts[17], fill=status_color)
        draw.text((x+PANEL-14, 637), f"{frame.progress:5.1%}  ·  {frame.time:05.2f} s  ·  {frame.speed:.2f} m/s",
                  anchor="rt", font=fonts[19], fill=INK)
        draw.rectangle((x, 666, x+PANEL-1, 670), fill="#e1e8e3")
        if frame.progress > 0:
            draw.rectangle((x, 666, x+round((PANEL-1)*frame.progress), 670), fill=color)
    return canvas


def mp4_boxes(data):
    offset = 0
    while offset < len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        if size < 8 or offset+size > len(data):
            raise ValueError("Unsupported MP4 box layout")
        yield kind, data[offset+8:offset+size]
        offset += size


def fix_video_index(path):
    """Index the actual H.264 IDR frames, including frame one omitted by VLC."""
    data = path.read_bytes()
    movie = dict(mp4_boxes(data))[b"moov"]
    table = movie
    for kind in (b"trak", b"mdia", b"minf", b"stbl"):
        table = dict(mp4_boxes(table))[kind]
    tables = dict(mp4_boxes(table))
    sizes = tables[b"stsz"]
    count = struct.unpack_from(">I", sizes, 8)[0]
    assert struct.unpack_from(">I", sizes, 4)[0] == 0
    lengths = struct.unpack_from(f">{count}I", sizes, 12)
    offsets = tables[b"stco"]
    assert struct.unpack_from(">I", offsets, 4)[0] == 1
    offset = struct.unpack_from(">I", offsets, 8)[0]
    sync = []
    for index, length in enumerate(lengths, 1):
        end = offset+length
        while offset < end:
            nal_length = struct.unpack_from(">I", data, offset)[0]
            assert nal_length > 0 and offset+4+nal_length <= end
            if data[offset+4] & 31 == 5 and (not sync or sync[-1] != index):
                sync.append(index)
            offset += 4+nal_length
        assert offset == end
    assert sync and sync[0] == 1
    payload = struct.pack(f">{len(sync)+2}I", 0, len(sync), *sync)

    def box(kind, content):
        return struct.pack(">I4s", len(content)+8, kind)+content

    def rebuild(content, shift):
        parts = []
        for kind, value in mp4_boxes(content):
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                value = rebuild(value, shift)
                if kind == b"stbl" and b"stss" not in dict(mp4_boxes(value)):
                    value += box(b"stss", payload)
            elif kind == b"stss":
                value = payload
            elif kind == b"stco":
                n = struct.unpack_from(">I", value, 4)[0]
                pointers = struct.unpack_from(f">{n}I", value, 8)
                value = value[:8]+struct.pack(f">{n}I", *(p+shift for p in pointers))
            parts.append(box(kind, value))
        return b"".join(parts)

    # The MP4 movie header precedes the media, so adjust chunk pointers by its growth.
    assert next(kind for kind, _ in mp4_boxes(data) if kind in (b"moov", b"mdat")) == b"moov"
    delta = len(rebuild(data, 0))-len(data)
    fixed = rebuild(data, delta)
    temporary = path.with_suffix(".indexed.mp4")
    temporary.write_bytes(fixed)
    temporary.replace(path)
    return count


def encode(source, destination, vlc):
    temporary = destination.with_suffix(".encoding.mp4")
    temporary.unlink(missing_ok=True)
    chain = ('#transcode{vcodec=h264,venc=x264{preset=fast,crf=18},fps=30}:'
             f'standard{{access=file,mux=mp4,dst="{temporary}"}}')
    subprocess.run([str(vlc), "-I", "dummy", "--no-one-instance", "--ignore-config",
                    "--no-media-library", "--demux=mjpeg", f"--mjpeg-fps={FPS}",
                    "--no-audio", "--no-sout-transcode-hurry-up", str(source),
                    "--sout", chain, "vlc://quit"], check=True, timeout=120,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if not temporary.exists() or temporary.stat().st_size < 4096:
        raise RuntimeError(f"VLC did not encode {destination.name}")
    frames = fix_video_index(temporary)
    temporary.replace(destination)
    return frames


def record_clip(destination, vlc, scratch, visual, camera, runs, cap, slot, fonts):
    backdrop = dashboard(visual.track, slot, cap, fonts)
    final_index = max(len(frames)-1 for frames in runs)
    source = scratch / f"{destination.stem}-{cap:.2f}.mjpg"
    cached = [None]*len(runs)
    previous_indices = [-1]*len(runs)
    with source.open("wb") as stream, mujoco.Renderer(visual.model, height=PANEL, width=PANEL) as renderer:
        for tick in range(final_index + 1):
            current = []
            for i, (frames, (_, color)) in enumerate(zip(runs, POLICIES.values())):
                index = min(tick, len(frames)-1)
                if index != previous_indices[i]:
                    cached[i] = scene_frame(renderer, visual, camera, frames, index, color)
                    previous_indices[i] = index
                current.append(frames[index])
            canvas = draw_frame(backdrop, cached, current, min(tick/FPS, max(f[-1].time for f in runs)), fonts)
            canvas.save(stream, "JPEG", quality=91, subsampling=0)
            if slot == 32 and cap == 1.5 and tick == round(FPS*3):
                canvas.save(destination.parents[1] / "preview.png")
        # Extra copies account for VLC consuming frames at the MJPEG stream boundary.
        final = BytesIO()
        canvas.save(final, "JPEG", quality=91, subsampling=0)
        stream.write(final.getvalue()*(2*FPS+2))
    try:
        count = encode(source, destination, vlc)
    finally:
        source.unlink(missing_ok=True)
    if abs(count-(final_index+1+2*FPS)) > 2:
        raise ValueError(f"Unexpected frame count for {destination.name}: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlc", type=Path, default=shutil.which("vlc") or Path("C:/Program Files/VideoLAN/VLC/vlc.exe"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--slots", type=int, nargs="+", default=TRACKS)
    args = parser.parse_args()
    if not args.vlc.is_file():
        parser.error("VLC was not found. Pass --vlc with the path to vlc.exe.")
    output = args.output.resolve()
    suite = RUN / "suite/suite.json"
    assessment = RUN / "assessment/test-episodes.csv"
    supplemental = OUT / "supplemental-episodes.csv"
    freeze = read_json(OUT / "comparison-freeze.json")
    settings = read_json(RUN / "protocol.json")
    model = RUN / "assessment/frozen-best.pkl"
    assert digest(model) == freeze["phase_checkpoints"]["speed-g98"]["sha256"]
    assert digest(suite) == freeze["suite_sha256"] and digest(assessment) == freeze["original_assessment_sha256"]
    assert digest(supplemental) == read_json(OUT / "supplemental-status.json")["sha256"]
    entries = read_json(suite)["splits"]["test"]
    if len(set(args.slots)) != len(args.slots) or not all(0 <= slot < len(entries) for slot in args.slots):
        parser.error("Choose unique test track slots from 0 to 89.")
    expected = {(int(r["track_slot"]), float(r["evaluation_speed_limit_m_s"]), r["policy_id"]): r
                for r in read_csv(assessment)+read_csv(supplemental) if r["policy_id"] in POLICIES}
    robot = RobotConfig(**settings["base_robot"])
    # A 100 mm band accommodates the 90 mm forward sensor reach at corners.
    # Two seconds allow the filtered PID's 0.738 s hold and a full search sweep.
    episode_config = replace(EpisodeConfig(**settings["episode"]),
                             corridor_half_width=.10, departure_seconds=2.0)
    write_json(output / "evaluation-protocol.json", {
        "evaluator_version": FITNESS_VERSION, "episode": asdict(episode_config),
        "slots": args.slots, "speed_caps_m_s": CAPS, "neat_generation": 98,
        "neat_sha256": digest(model), "pid_settings": freeze["pid_settings"],
        "departure_requires_no_sensor_contact": True, "retrospective": True,
        "recorder_sha256": digest(Path(__file__))})
    simulations = {cap: Simulation(replace(robot, max_speed=cap)) for cap in CAPS}
    folders = {cap: output / f"{i:02d}-{cap:.2f}-mps" for i, cap in enumerate(CAPS, 1)}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()
    videos, rows = [], []
    with tempfile.TemporaryDirectory(prefix="recording-", dir=output) as scratch:
        for slot in args.slots:
            entry = entries[slot]
            track = TrackSpec.load(suite.parent / entry["path"])
            assert track.geometry_hash == entry["geometry_hash"]
            mask = TrackMask(track)
            runs = {}
            for cap in CAPS:
                for policy in POLICIES:
                    source_model = model if policy == "speed-trained" else None
                    params = freeze["pid_settings"].get(policy, {})
                    key = slot, cap, policy
                    frames, result = run_episode(simulations[cap], mask, policy, source_model,
                                                 params, episode_config, expected[key],
                                                 output / "telemetry" / f"{slot:03d}-{cap:.2f}-{policy}.csv")
                    runs[cap, policy] = frames
                    rows.append({"slot": slot, "difficulty": track.difficulty, "cap_m_s": cap,
                                 "controller": POLICIES[policy][0], "termination": result.termination,
                                 "elapsed_s": result.elapsed_s, "progress": result.progress,
                                 "lateral_rmse_mm": 1000*result.lateral_rmse_m, "track_hash": track.geometry_hash,
                                 "previous_termination": expected[key]["termination"],
                                 "previous_elapsed_s": expected[key]["elapsed_s"],
                                 "previous_progress": expected[key]["progress"]})
            visual = Simulation(robot)
            visual.reset(track, mask)
            visual.enable_rendering()
            camera = fixed_camera(visual, runs)
            for cap in CAPS:
                path = folders[cap] / f"track-{slot:03d}-{track.difficulty}.mp4"
                print(f"Recording {path.relative_to(output)}", flush=True)
                count = record_clip(path, args.vlc, Path(scratch), visual, camera,
                                    [runs[cap, p] for p in POLICIES], cap, slot, fonts)
                videos.append({"path": path.relative_to(output).as_posix(), "slot": slot,
                               "wheel_speed_cap_m_s": cap, "frames": count, "sha256": digest(path)})
    summary = output / "episode-summary.csv"
    summary.unlink(missing_ok=True)
    append_csv(summary, rows, list(rows[0]))
    write_json(output / "manifest.json", {"fps": FPS, "size": SIZE, "neat_generation": 98,
                                         "evaluator_version": FITNESS_VERSION,
                                         "corridor_half_width_m": episode_config.corridor_half_width,
                                         "departure_seconds": episode_config.departure_seconds,
                                         "departure_requires_no_sensor_contact": True,
                                         "neat_sha256": digest(model), "suite_sha256": digest(suite),
                                         "pid_settings": freeze["pid_settings"],
                                         "recorder_sha256": digest(Path(__file__)), "videos": videos})
    print(f"Saved {len(videos)} videos to {output}", flush=True)


if __name__ == "__main__":
    main()
