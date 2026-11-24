from copy import deepcopy
from dataclasses import asdict

import pytest

import compare_neat_pid_100 as comparison
from line_follower.episodes import EpisodeResult


def episode(policy, index):
    row = {name: 0 for name in EpisodeResult.__dataclass_fields__}
    row.update(policy=policy, group=("mixed", "speed", "corner")[index % 3],
               track_hash=str(index), difficulty="advanced", completed=True,
               completion_time_s=2., geometry_lateral_rmse_m=.01, straight_speed_m_s=None,
               average_forward_speed_m_s=.5, peak_forward_speed_m_s=.7,
               steady_yaw_rms_rad_s=.1, progress=1., command_jitter=.001)
    return row


def entries():
    return [{"geometry_hash": str(index), "group": ("mixed", "speed", "corner")[index % 3],
             "dedicated_straight": index % 3 == 1} for index in range(6)]


def test_identical_policies_have_zero_paired_differences():
    rows = [episode(policy, index) for policy in comparison.POLICIES for index in range(6)]
    result = comparison.summarize(rows, entries())
    paired = result["paired"]
    assert paired["shared_finishes"] == 6
    assert paired["mcnemar_exact_two_sided_p"] == 1
    assert paired["completion_difference_paired_bootstrap_95"] == [0, 0]
    assert paired["relative_time_change_paired_bootstrap_95_pct"] == [0, 0]


def test_time_statistics_exclude_failures_and_keep_completion_denominators():
    rows = [episode(policy, index) for policy in comparison.POLICIES for index in range(6)]
    for row in rows:
        if row["policy"] == "neat":
            row["completion_time_s"] = 1.
            if int(row["track_hash"]) < 3:
                row.update(completed=False, termination="stalled", completion_time_s=None)
    result = comparison.summarize(rows, entries())
    paired = result["paired"]
    assert paired["shared_finishes"] == 3 and paired["pid_only"] == 3
    assert paired["mcnemar_exact_two_sided_p"] == .25
    assert paired["relative_time_change_pct"] == -50
    assert paired["time_difference_paired_bootstrap_95_s"] == [-1, -1]
    assert result["policies"]["neat"]["all"]["completed"] == 3
    bad = deepcopy(rows)
    bad.pop()
    with pytest.raises(ValueError, match="identical track sets"):
        comparison.summarize(bad, entries())


def test_freeze_preserves_requested_artifacts_and_rejects_tampering(tmp_path):
    if not comparison.NEAT.exists() or not comparison.PID.exists():
        pytest.skip("Local research artifacts are optional")
    manifest = comparison.freeze(tmp_path)
    assert manifest["robot"]["max_speed"] == 1.
    assert manifest["policies"]["neat"]["generation"] == 134
    for policy, source in (("neat", comparison.NEAT), ("pid", comparison.PID)):
        assert (tmp_path/"frozen"/manifest["policies"][policy]["file"]).read_bytes() == source.read_bytes()
    path = tmp_path/"frozen"/manifest["policies"]["pid"]["file"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="frozen controller changed"):
        comparison.verify_frozen(tmp_path)


def test_comparison_worker_matches_the_shared_episode_loop_for_both_policies(tmp_path):
    import json
    from line_follower.controllers import read_model_artifact, load_model
    from line_follower.episodes import run_episode
    from line_follower.config import EpisodeConfig, RobotConfig
    from line_follower.pid import ResearchPIDConfig, ResearchPIDController
    from line_follower.simulation import Simulation
    from run_pid_smoothness_tuning import curve
    if not comparison.NEAT.exists() or not comparison.PID.exists():
        pytest.skip("Local research artifacts are optional")
    track = curve()
    path = tmp_path / "curve.json"
    track.save(path)
    metadata = read_model_artifact(comparison.NEAT)["metadata"]
    robot, episode_config = RobotConfig(**metadata["robot"]), EpisodeConfig(**metadata["episode"])
    for policy, model in (("neat", comparison.NEAT), ("pid", comparison.PID)):
        actual = comparison.worker({"policy": policy, "track": str(path), "config": str(model),
                                   "robot_json": json.dumps(asdict(robot)), "episode": asdict(episode_config),
                                   "entry": {"windows": [(1.4, 2.7)], "group": "speed", "replay": False}})
        controller = load_model(model)[0] if policy == "neat" else ResearchPIDController(ResearchPIDConfig.load(model), robot)
        expected = asdict(run_episode(Simulation(robot), track, controller, episode_config, detailed=True))
        assert {field: actual[field] for field in expected} == expected
