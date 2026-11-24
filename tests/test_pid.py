"""PID contracts and physical integration without NEAT evolution."""

from dataclasses import replace
import numpy as np
import pytest

from line_follower.config import RobotConfig
from line_follower.episodes import run_episode
from line_follower.pid import ResearchPIDConfig, ResearchPIDController
from line_follower.pid_experiments import ranking
from line_follower.simulation import Simulation
from line_follower.tracks import Section, TrackSpec, straight_track


def test_pid_sign_reset_and_bounded_recovery():
    config = ResearchPIDConfig()
    controller = ResearchPIDController(config, RobotConfig())
    left = np.eye(15)[2]
    action = controller.act(left, 1/60)
    assert action[0] < action[1]
    for _ in range(300):
        action = controller.act(np.zeros(15), 1/60)
        assert np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 1
    controller.reset()
    expected = ResearchPIDController(config, RobotConfig()).act(left, 1/60)
    assert np.array_equal(controller.act(left, 1/60), expected)


def test_pid_mirrored_readings_produce_mirrored_motor_commands():
    robot = RobotConfig()
    left, right = (ResearchPIDController(ResearchPIDConfig(), robot) for _ in range(2))
    for index in (7, 6, 4, 2, 1):
        a = left.act(np.eye(15)[index], robot.dt)
        b = right.act(np.eye(15)[14-index], robot.dt)
        assert np.allclose(a, b[::-1])


def test_supervisor_advances_by_commands_not_route_information():
    controller = ResearchPIDController(ResearchPIDConfig(corner_supervision=True), RobotConfig())
    observation = np.zeros(15)
    observation[3:8] = 1
    controller.act(observation, 1/60)
    assert controller.mode == "approach"
    for _ in range(100):
        controller.act(np.zeros(15), 1/60)
        if controller.mode == "pivot":
            break
    assert controller.mode == "pivot"
    assert controller.act(np.zeros(15), 1/60)[0] < 0


def test_pid_integral_stays_bounded_and_observations_are_validated():
    controller = ResearchPIDController(ResearchPIDConfig(ki=.2), RobotConfig())
    for _ in range(600):
        controller.act(np.eye(15)[0], 1/60)
    assert abs(controller.integral) <= controller.config.integral_limit
    with pytest.raises(ValueError):
        controller.act(np.zeros(14), 1/60)
    with pytest.raises(ValueError):
        controller.act(np.zeros(15), 0)
    with pytest.raises(ValueError):
        replace(ResearchPIDConfig(), base=-1)


def test_pid_straight_physics_uses_real_readings_and_moves_forward():
    robot = RobotConfig(max_speed=.35)
    simulation = Simulation(robot)
    observation = simulation.reset(straight_track(2))
    controller = ResearchPIDController(ResearchPIDConfig(), robot)
    start = simulation.state.position.copy()
    for _ in range(180):
        observation = simulation.step(controller.act(observation, robot.dt))
    assert simulation.state.position[0]-start[0] > .4
    assert abs(simulation.state.position[1]-start[1]) < .025
    assert np.max(np.abs(simulation.state.torques)) <= robot.max_torque+1e-6


def test_candidate_ranking_prioritizes_completion_over_speed_and_error():
    slow = [{"completed": True, "progress": 1, "lateral_rmse_m": .02, "completion_time_s": 40}]
    fast_failure = [{"completed": False, "progress": .99, "lateral_rmse_m": .001, "completion_time_s": None}]
    assert ranking(slow) > ranking(fast_failure)


@pytest.mark.parametrize("sign", [-1, 1])
def test_supervised_pid_completes_mirrored_diagnostic_zigzags(sign):
    robot = RobotConfig(max_speed=.35)
    sections = (Section("straight", length=.6), Section("corner", angle=sign*np.pi/2),
                Section("straight", length=.25), Section("corner", angle=-sign*np.pi/2),
                Section("straight", length=.25), Section("corner", angle=sign*np.pi/2),
                Section("straight", length=.55))
    track = TrackSpec("pid-physical-zigzag", sections)
    controller = ResearchPIDController(ResearchPIDConfig(corner_supervision=True), robot)
    result = run_episode(Simulation(robot), track, controller)
    assert result.completed, result
    assert result.progress == 1 and result.elapsed_s < result.time_limit_s
    assert result.error == ""
