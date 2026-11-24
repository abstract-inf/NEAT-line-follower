import pytest

import run_pid_fullspeed_100 as study


def row(speed):
    return {"candidate": "policy", "features": "speed-straight", "completed": True,
            "straight_speed_m_s": speed, "progress": 1., "completion_time_s": 5.,
            "time_limit_s": 60., "command_jitter": 0., "steady_yaw_rms_rad_s": 0.,
            "geometry_lateral_rmse_m": 0.}


def test_search_candidates_allow_full_cruise_and_preserve_recovery():
    policies, _ = study.candidates()
    baseline = study.read_json(study.SLOW_PID)
    for name, parameters in policies.items():
        assert parameters["base"] == (.75 if name == "previous-base-075" else 1.)
        for field in baseline:
            if field not in {*study.FIELDS, "base"}:
                assert parameters[field] == baseline[field]


def test_slow_cruise_cannot_pass_the_speed_gate_or_be_promoted_as_fullspeed():
    slow = study.metrics([row(.743)])
    fast = study.metrics([{**row(.991), "candidate": "full"}])
    slow.update(candidate="slow", completed=100)
    assert not slow["speed_feasible"] and fast["speed_feasible"]
    assert study.promote({"slow": slow, "full": fast}) == ["full"]
    missing = study.metrics([{**row(.991), "straight_speed_m_s": None}])
    assert not missing["speed_feasible"]


def test_actual_pid_can_sustain_near_one_metre_per_second_at_the_shared_cap(tmp_path):
    if not study.SLOW_PID.exists():
        pytest.skip("Local saved PID is optional")
    study.probe(tmp_path)
    old, full = study.read_json(tmp_path/"feasibility.json")
    assert old["straight_speed_m_s"] < .8
    assert full["straight_speed_m_s"] >= .95 and full["completed"]
    assert full["mean_action_L"] == full["mean_action_R"] == 1.
