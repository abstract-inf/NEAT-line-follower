"""Physical behavior gates, using contacts rather than inspecting actuator setup."""

import math
import numpy as np
import pytest

from line_follower.config import RobotConfig
from line_follower.simulation import Simulation
from line_follower.tracks import TrackMask, straight_track


@pytest.fixture(scope="module")
def floor():
    track = straight_track(2)
    return track, TrackMask(track)


def test_mass_sensors_and_stationary_stability(floor):
    sim = Simulation()
    readings = sim.reset(*floor)
    start = sim.state.position.copy()
    for _ in range(60):
        sim.step([0, 0])
    assert np.sum(sim.model.body_mass) == pytest.approx(0.55)
    assert np.linalg.norm(sim.state.position - start) < 0.001
    assert readings[7] == 1 and readings.shape == (15,)
    assert np.all(np.isin(readings, [0, 1]))
    assert np.allclose(sim.sensor_hits[:, 2], 0)


def test_speed_acceleration_braking_and_reverse(floor):
    sim = Simulation()
    sim.reset(*floor)
    start_x = sim.state.position[0]
    sim.step([1, 1])
    assert 0 < sim.state.forward_speed < 0.15
    assert np.max(np.abs(sim.state.torques)) <= 0.030001
    for _ in range(59):
        sim.step([1, 1])
    assert sim.state.position[0] - start_x > 0.20
    assert sim.state.forward_speed == pytest.approx(0.30, abs=0.015)
    for _ in range(60):
        sim.step([0, 0])
    assert abs(sim.state.forward_speed) < 0.005
    for _ in range(60):
        sim.step([-1, -1])
    assert sim.state.forward_speed < -0.28


@pytest.mark.parametrize("action,sign", [([-1, 1], 1), ([1, -1], -1)])
def test_opposite_wheels_turn_in_both_directions(floor, action, sign):
    sim = Simulation()
    sim.reset(*floor)
    position = sim.state.position.copy()
    for _ in range(30):
        sim.step(action)
    assert sim.state.yaw * sign > 0.7
    assert sim.state.yaw_rate * sign > 2.0
    assert np.linalg.norm(sim.state.position[:2] - position[:2]) < 0.02


def test_sensor_pose_and_upside_down_rejection(floor):
    sim = Simulation()
    sim.reset(*floor)
    sim.data.qpos[1] = 0.15
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    assert not np.any(sim.read_sensors())
    sim.data.qpos[3:7] = [0, 1, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)
    assert not np.any(sim.read_sensors())


def test_reset_erases_motion_and_controller_commands(floor):
    sim = Simulation()
    initial = sim.reset(*floor)
    qpos = sim.data.qpos.copy()
    for _ in range(30):
        sim.step([0.5, -1])
    assert np.array_equal(sim.reset(*floor), initial)
    assert np.allclose(sim.data.qpos, qpos)
    assert np.all(sim.data.qvel == 0)
    assert np.all(sim.data.act == 0)
    with pytest.raises(ValueError):
        sim.step([math.nan, 0])
