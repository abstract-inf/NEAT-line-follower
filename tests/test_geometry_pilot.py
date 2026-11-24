"""Pilot isolation and offset controls; no evolutionary training is invoked."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from line_follower.config import RobotConfig
from line_follower.geometry_pilot import _rank, _trial, pilot_tracks
from line_follower.controllers import PIDConfig
from line_follower.robot_design import board_geometry, _triangles, _cross
from line_follower.simulation import Simulation


def test_pilot_track_sets_are_valid_disjoint_and_cover_requested_features():
    tuning, evaluation = pilot_tracks("tuning"), pilot_tracks("evaluation")
    assert not {t.geometry_hash for t in tuning} & {t.geometry_hash for t in evaluation}
    assert {t.difficulty for t in evaluation} == {"straight", "sharp", "zigzag", "curve"}
    assert all(len(t.route.corners) > 0 for t in evaluation if t.difficulty == "sharp")


def test_longer_offset_changes_neck_and_array_without_changing_main_body():
    near, far = RobotConfig(), replace(RobotConfig(), sensor_offset=.140)
    assert board_geometry(near)[1] == board_geometry(far)[1]
    a, b = Simulation(near), Simulation(far)
    for name in ("left_wheel", "right_wheel", "caster"):
        assert np.array_equal(a.model.body(name).pos, b.model.body(name).pos)
    for name in ("body_mass", "body_inertia", "body_ipos", "actuator_gainprm", "actuator_forcerange"):
        assert np.array_equal(getattr(a.model, name), getattr(b.model, name))


def test_unified_visual_mesh_keeps_concave_openings():
    outline = board_geometry(RobotConfig())[0]
    polygon_area = sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(outline, outline[1:]+outline[:1]))/2
    triangles_area = sum(_cross(outline[a], outline[b], outline[c])/2 for a,b,c in _triangles(outline))
    assert triangles_area == pytest.approx(polygon_area)
    # A hull would bridge the openings and substantially increase this area.
    assert polygon_area < .014


def test_completing_track_outranks_low_early_failure_error():
    completed = {"completed": True, "progress": 1, "lateral_rmse_m": .03}
    failed = {"completed": False, "progress": .4, "lateral_rmse_m": .0001}
    assert _rank([completed]) > _rank([failed])


def test_pilot_exports_si_metrics_for_a_real_physical_episode():
    track = pilot_tracks("tuning")[0]
    result = _trial({"robot_json": json.dumps(asdict(RobotConfig())), "track_json": json.dumps(asdict(track)),
                     "pid": asdict(PIDConfig(kp=.12, ki=.0001, kd=.001, base=.6)),
                     "stage": "check", "candidate": 0, "speed": .18})
    assert result["completed"]
    assert result["minimum_pcb_clearance_m"] > .002
    assert result["maximum_tilt_rad"] < .02
    assert result["line_loss_events"] == 0
