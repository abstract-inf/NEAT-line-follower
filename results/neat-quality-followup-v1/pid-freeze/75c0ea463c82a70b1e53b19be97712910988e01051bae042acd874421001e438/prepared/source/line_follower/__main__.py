"""Command-line entry point. No training or GUI starts merely by importing it."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
from importlib.resources import files
import json
import multiprocessing
from pathlib import Path
import pickle

from .config import EpisodeConfig, RobotConfig, freeze_geometry, verify_geometry
from .controllers import PIDConfig, load_model, load_neat_config, save_model
from .experiments import TrainConfig, evaluate, seen_tracks, train, write_json
from .suites import create_suite, load_split
from .tracks import DIFFICULTIES, GeneratorConfig, TrackMask, TrackSpec, generate_track, straight_track, validate_track

DEFAULT_SUITE = "benchmarks/suite/suite.json"


def _robot(args, metadata: dict | None = None) -> RobotConfig:
    values = dict(metadata.get("robot", {})) if metadata else {}
    if getattr(args, "preset", None):
        values = asdict(RobotConfig.paper() if args.preset == "paper" else RobotConfig())
    if getattr(args, "robot_config", None):
        values = json.loads(Path(args.robot_config).read_text(encoding="utf-8"))
    if values and "design_revision" not in values:
        raise ValueError("Robot configuration predates geometry revisions. Select --preset competition or an explicit versioned JSON and re-evaluate")
    if getattr(args, "max_speed", None) is not None:
        values["max_speed"] = args.max_speed
    if getattr(args, "sensor_offset", None) is not None:
        values["sensor_offset"] = args.sensor_offset
    return RobotConfig(**values)


def _episode(args) -> EpisodeConfig:
    path = getattr(args, "episode_config", None)
    if path:
        return EpisodeConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
    return (EpisodeConfig(maximum_time_limit=120, time_reference_speed=.10)
            if getattr(args, "strategy", None) == "adaptive" else EpisodeConfig())


def _visual_track(args) -> TrackSpec:
    if args.track:
        return TrackSpec.load(args.track)
    return generate_track(args.seed, args.difficulty) if args.seed is not None else straight_track()


def _training_settings(args) -> TrainConfig:
    if args.resume:
        with gzip.open(args.resume, "rb") as stream:
            saved = pickle.load(stream)
        return saved["settings"]
    return TrainConfig(args.strategy, args.seed, args.generations, args.population, args.workers,
                       args.time_limit, tuple(args.exclude_feature), args.backend,
                       args.gpu_batch_size if args.gpu_batch_size is not None else args.population, args.episodes_per_genome,
                       track_profile=args.track_profile, aggregation=args.aggregation, cvar_fraction=args.cvar_fraction)


def _add_physics_arguments(parser) -> None:
    parser.add_argument("--preset", choices=("competition", "paper"), help="New wide-arc PCB robot or original sensor geometry")
    parser.add_argument("--robot-config", help="JSON RobotConfig; all physical quantities use SI")
    parser.add_argument("--max-speed", type=float, help="Wheel-rim target ceiling in m/s, not compulsory forward speed")
    parser.add_argument("--sensor-offset", type=float, help="Axle to center sensor in meters; translates the whole array")


def _add_visual_arguments(parser) -> None:
    parser.add_argument("--track", help="Saved TrackSpec JSON")
    parser.add_argument("--seed", type=int, help="Generate a track; omitted means the fixed straight benchmark")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="beginner")
    _add_physics_arguments(parser)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="MuJoCo sensor-only line-following research platform")
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate and validate a reproducible track")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--difficulty", choices=DIFFICULTIES, default="beginner")
    generate.add_argument("--output", required=True)
    generate.add_argument("--preview", help="Optional floor mask PNG")
    generate.add_argument("--generator-config", help="JSON GeneratorConfig, including feature exclusions")
    suite = commands.add_parser("prepare-suite", help="Freeze training benchmark, validation, and test splits")
    suite.add_argument("--output", default="benchmarks/suite")
    suite.add_argument("--seed", type=int, default=20261027)
    suite.add_argument("--validation-count", type=int, default=30)
    suite.add_argument("--test-count", type=int, default=90)
    suite.add_argument("--generator-config")
    suite.add_argument("--protect-suite", help="Exclude every validation/test geometry from this existing suite")
    render = commands.add_parser("render", help="Save an actual MuJoCo render")
    _add_visual_arguments(render)
    render.add_argument("--output", default="runs/previews/robot.png")
    render.add_argument("--camera", choices=("robot", "top", "front", "track"), default="robot")
    render.add_argument("--transparent", action="store_true", help="Reveal the 15 sensor sites and caster")
    studio = commands.add_parser("export-studio", help="Export the robot and track for MuJoCo Live in a browser")
    _add_visual_arguments(studio)
    studio.add_argument("--output", default="runs/previews/line-follower.mjb")
    view = commands.add_parser("view", help="Open the native MuJoCo viewer")
    _add_visual_arguments(view)
    view.add_argument("--model", help="Saved project NEAT model; C switches between model and PID")
    view.add_argument("--pid-config", help="JSON PID gains/base, e.g. a geometry-pilot selection")
    view.add_argument("--count", type=int, default=6, help="Number of seeded tracks available with N/P")
    view.add_argument("--auto-close", type=float, help="Close after this many real seconds for GUI smoke checks")
    observer = commands.add_parser("preview-training", help="Attach a viewer to an existing run without starting training")
    observer.add_argument("--run", required=True)
    observer.add_argument("--auto-close", type=float)
    observer.add_argument("--replay-speed", type=int, default=10, help="Old champion-only runs: control steps per frame; live snapshots are unpaced")
    validate = commands.add_parser("validate", help="Check geometry, optionally physical traversability")
    _add_visual_arguments(validate)
    validate.add_argument("--physics", action="store_true", help="Use privileged slow reference driver; not a baseline result")
    pilot = commands.add_parser("geometry-pilot", help="Tune a shared PID and compare sensor offsets; never runs NEAT")
    pilot.add_argument("--config", help="JSON PilotConfig; omitted uses documented defaults")
    pilot.add_argument("--output", required=True)
    pilot_report = commands.add_parser("geometry-report", help="Regenerate pilot tables/figures from saved CSVs")
    pilot_report.add_argument("--run", required=True)
    freeze = commands.add_parser("freeze-robot", help="Explicitly select and fingerprint geometry before starting evolution")
    _add_physics_arguments(freeze)
    freeze.add_argument("--output", default="configs/geometry-freeze.json")
    gpu = commands.add_parser("gpu-benchmark", help="Optional CUDA physics/parity probe; never trains NEAT")
    gpu.add_argument("--worlds", type=int, default=32)
    gpu.add_argument("--control-steps", type=int, default=60)
    gpu.add_argument("--workers", type=int, default=4)
    gpu.add_argument("--output", default="runs/gpu-benchmark.json")
    training = commands.add_parser("train", help="Evolve one policy across a declared track condition")
    training.add_argument("--strategy", choices=("single", "multi", "curriculum", "adaptive"), default="curriculum")
    training.add_argument("--track-profile", choices=("standard", "zigzag", "progressive"), default="standard")
    training.add_argument("--aggregation", choices=("mean_min", "mean_cvar"), default="mean_min")
    training.add_argument("--cvar-fraction", type=float, default=.30)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--suite", default=DEFAULT_SUITE)
    training.add_argument("--output", required=True)
    training.add_argument("--generations", type=int, default=500)
    training.add_argument("--population", type=int, default=200)
    training.add_argument("--episodes-per-genome", type=int, default=4, help="Track attempts per genome each generation; default four-track mixture")
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="GPU uses experimental batched CUDA physics; NEAT inference remains CPU")
    training.add_argument("--gpu-batch-size", type=int, help="Defaults to the population size; smaller values evaluate subsets concurrently")
    presentation = training.add_mutually_exclusive_group()
    presentation.add_argument("--visualize", action="store_true", help="Show champions in a separate MuJoCo window; physics workers remain unpaced")
    presentation.add_argument("--headless", dest="visualize", action="store_false", help="No rendering (default)")
    training.set_defaults(visualize=False)
    training.add_argument("--geometry-lock", default="configs/geometry-freeze.json")
    training.add_argument("--neat-config")
    training.add_argument("--episode-config")
    training.add_argument("--time-limit", type=float, help="Explicit episode override, useful for integration checks")
    training.add_argument("--exclude-feature", action="append", default=[])
    training.add_argument("--resume", help="Trusted latest.pkl.gz from this run; retain its saved settings")
    training.add_argument("--stop-after", type=int, help="Stop after N additional generations, retaining the original curriculum horizon")
    _add_physics_arguments(training)
    evaluation = commands.add_parser("evaluate", help="Evaluate PID or a saved NEAT policy on the same tracks")
    source = evaluation.add_mutually_exclusive_group()
    source.add_argument("--suite", default=None)
    source.add_argument("--seen-run", help="Sample encountered training tracks for the memorization diagnostic")
    source.add_argument("--track", help="One manually specified benchmark")
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--final-test", action="store_true", help="Explicitly authorize final test evaluation")
    evaluation.add_argument("--model")
    evaluation.add_argument("--pid-config", help="JSON gains and base; provenance is saved in evaluation.json")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--repeats", type=int, default=1)
    evaluation.add_argument("--telemetry", action="store_true")
    evaluation.add_argument("--episode-config")
    evaluation.add_argument("--time-limit", type=float)
    _add_physics_arguments(evaluation)
    analysis = commands.add_parser("analyze", help="Make tables, paired bootstrap estimates, and figures from CSV data")
    analysis.add_argument("inputs", nargs="+", help="Episode CSV paths or glob expressions")
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--bootstrap-samples", type=int, default=2000)
    analysis.add_argument("--seed", type=int, default=0)
    legacy = commands.add_parser("import-legacy", help="Import a trusted historical genome with an explicit motor-frame conversion")
    legacy.add_argument("--genome", required=True)
    legacy.add_argument("--neat-config", required=True)
    legacy.add_argument("--output", required=True)
    _add_physics_arguments(legacy)
    study = commands.add_parser("study", help="Run the three NEAT conditions across independent evolution seeds; never opens test files")
    study.add_argument("--suite", default=DEFAULT_SUITE)
    study.add_argument("--output", required=True)
    study.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    study.add_argument("--generations", type=int, default=500)
    study.add_argument("--population", type=int, default=200)
    study.add_argument("--episodes-per-genome", type=int, default=4)
    study.add_argument("--workers", type=int, default=1)
    study.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    study.add_argument("--gpu-batch-size", type=int, default=128)
    study.add_argument("--geometry-lock", default="configs/geometry-freeze.json")
    _add_physics_arguments(study)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "generate":
        config = GeneratorConfig(**json.loads(Path(args.generator_config).read_text())) if args.generator_config else None
        track = generate_track(args.seed, args.difficulty, config)
        track.save(args.output)
        if args.preview:
            Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
            TrackMask(track).image.save(args.preview)
        print(json.dumps({"path": args.output, "length_m": track.route.length, "features": track.features, "geometry_hash": track.geometry_hash}, indent=2))
    elif args.command == "prepare-suite":
        config = GeneratorConfig(**json.loads(Path(args.generator_config).read_text())) if args.generator_config else None
        protected = set()
        if args.protect_suite:
            from .suites import read_suite
            parent_manifest = read_suite(args.protect_suite)
            protected = {e["geometry_hash"] for split in ("validation", "test") for e in parent_manifest["splits"][split]}
        print(create_suite(args.output, args.seed, args.validation_count, args.test_count, config, protected))
    elif args.command == "render":
        from .viewer import render_track
        print(render_track(_visual_track(args), args.output, _robot(args), args.camera, args.transparent))
    elif args.command == "export-studio":
        from .viewer import export_studio_model
        print(export_studio_model(_visual_track(args), args.output, _robot(args)))
    elif args.command == "view":
        from .viewer import view_tracks
        controller, metadata = load_model(args.model) if args.model else (None, {})
        tracks = [generate_track(args.seed + i, args.difficulty) for i in range(args.count)] if args.seed is not None and not args.track else [_visual_track(args)]
        pid = PIDConfig(**json.loads(Path(args.pid_config).read_text())) if args.pid_config else None
        view_tracks(tracks, _robot(args, metadata), controller, auto_close=args.auto_close, pid_config=pid)
    elif args.command == "preview-training":
        from .viewer import _training_preview
        _training_preview(args.run, multiprocessing.get_context("spawn").Event(), auto_close=args.auto_close,
                          replay_speed=args.replay_speed)
    elif args.command == "validate":
        track = _visual_track(args)
        errors = validate_track(track)
        if errors:
            raise ValueError("; ".join(errors))
        if args.physics:
            from .validation import validate_physics
            result = validate_physics(track, _robot(args))
            print(json.dumps(asdict(result), indent=2))
            if not result.completed:
                raise SystemExit(1)
        else:
            print(f"Geometry valid: {track.name}, features={track.features}")
    elif args.command == "geometry-pilot":
        from .geometry_pilot import PilotConfig, run_pilot
        config = PilotConfig(**json.loads(Path(args.config).read_text())) if args.config else PilotConfig()
        print(run_pilot(args.output, config))
    elif args.command == "freeze-robot":
        print(freeze_geometry(args.output, _robot(args)))
    elif args.command == "geometry-report":
        from .geometry_pilot import report_pilot
        print(report_pilot(args.run))
    elif args.command == "gpu-benchmark":
        from .gpu_probe import benchmark_gpu
        print(benchmark_gpu(args.output, args.worlds, args.control_steps, args.workers))
    elif args.command == "train":
        robot, episode = _robot(args), _episode(args)
        if args.resume:
            with gzip.open(args.resume, "rb") as stream:
                saved = pickle.load(stream)
            if all(value is None for value in (args.robot_config, args.max_speed, args.preset, args.sensor_offset)):
                robot = saved["robot"]
            if args.episode_config is None:
                episode = saved["episode"]
        verify_geometry(args.geometry_lock, robot)
        from .viewer import training_preview
        with training_preview(args.output, args.visualize, robot):
            print(train(args.output, args.suite, _training_settings(args), robot, episode,
                        args.neat_config, args.resume, args.stop_after, args.geometry_lock, live_preview=args.visualize))
    elif args.command == "evaluate":
        if args.split == "test" and not args.final_test:
            raise ValueError("Test tracks are reserved: use --final-test only after freezing the experimental protocol")
        metadata = load_model(args.model)[1] if args.model else {}
        robot = _robot(args, metadata)
        pid = PIDConfig(**json.loads(Path(args.pid_config).read_text())) if args.pid_config else None
        if args.seen_run:
            tracks = seen_tracks(args.seen_run)
        elif args.track:
            tracks = [TrackSpec.load(args.track)]
        else:
            tracks = load_split(args.suite or DEFAULT_SUITE, args.split)
        episode = _episode(args) if args.episode_config or not metadata.get("episode") else EpisodeConfig(**metadata["episode"])
        print(evaluate(args.output, tracks, robot, episode, args.model, pid, args.repeats, args.telemetry,
                       args.time_limit, "seen" if args.seen_run else None))
    elif args.command == "analyze":
        from .analysis import analyze
        print(analyze(args.inputs, args.output, args.bootstrap_samples, args.seed))
    elif args.command == "import-legacy":
        config = load_neat_config(args.neat_config)
        with Path(args.genome).open("rb") as stream:
            genome = pickle.load(stream)
        robot = _robot(args)
        save_model(args.output, genome, config, {"condition": "neat_legacy", "evolution_seed": "legacy",
                   "robot": asdict(robot), "swap_legacy_motors": True,
                   "source": str(Path(args.genome).resolve()), "historical_compatibility": "Imported under 0.92; physical transfer is not implied"})
        print(args.output)
    elif args.command == "study":
        verify_geometry(args.geometry_lock, _robot(args))
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Study output must be empty")
        write_json(output / "protocol.json", {"conditions": ["single", "multi", "curriculum", "pid_original"],
                   "evolution_seeds": args.seeds, "generations": args.generations, "population": args.population,
                   "suite": str(Path(args.suite).resolve()), "robot": asdict(_robot(args)),
                   "analysis": "Paired held-out tracks; report per-evolution-run variability and paired run/track bootstrap intervals",
                   "test_results_used": False, "confirmatory_analysis_requires_prespecification": True})
        for seed in args.seeds:
            for strategy in ("single", "multi", "curriculum"):
                train(output / f"{strategy}-seed-{seed}", args.suite,
                      TrainConfig(strategy, seed, args.generations, args.population, args.workers,
                                  backend=args.backend, gpu_batch_size=args.gpu_batch_size,
                                  episodes_per_genome=args.episodes_per_genome), _robot(args),
                      geometry_lock=args.geometry_lock)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
