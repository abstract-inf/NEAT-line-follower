"""Checks against stationary reward, route shortcuts, and hidden failure omission."""

from dataclasses import replace
import csv
import math

import numpy as np
import pytest

from line_follower.config import EpisodeConfig, RobotConfig
from line_follower.controllers import PIDController
from line_follower.episodes import Episode, run_episode
from line_follower.fitness import LegacyScorer, RouteProgress, aggregate_fitness, progress_fitness
from line_follower.simulation import RobotState, Simulation
from line_follower.tracks import Branch, Section, TrackSpec, straight_track


class ConstantController:
    def __init__(self, action=(0.0, 0.0)):
        self.action = np.asarray(action)
    def reset(self):
        pass
    def act(self, observation, dt):
        return self.action.copy()


def test_stationary_centering_loses_to_meaningful_progress():
    assert progress_fitness(0, False, 5, 45, 1e9, 300) < 0
    assert progress_fitness(0.5, False, 15, 45, -1e9, 900) > progress_fitness(0.1, False, 45, 45, 1e9, 2700)
    assert progress_fitness(1, True, 20, 45, 0, 1200) > progress_fitness(1, True, 40, 45, 0, 2400)
    assert progress_fitness(1, True, 45, 45, -1e9, 2700) > progress_fitness(0.999, False, 1, 45, 1e9, 60)
    assert aggregate_fitness([1, 1, 1, -1]) == pytest.approx(-0.25)


def test_speed_and_jitter_fitness_rewards_fast_smooth_finishes():
    def score(time, completed=True, jitter=0):
        return progress_fitness(1 if completed else .95, completed, time, 45, 0, 60, .40,
                                command_jitter=jitter, command_jitter_penalty=.20)

    assert score(4) > score(8)
    assert score(4) > score(4, jitter=1)
    assert score(8) > score(4, jitter=1)
    assert score(45, jitter=1) > score(4, completed=False)
    assert score(4, completed=False) == progress_fitness(.95, False, 4, 45, 0, 60)


def test_episode_measures_command_changes_without_penalizing_steady_turns():
    sim = Simulation()
    track = straight_track(.4)

    def measure(actions, penalty):
        episode = Episode(sim, track, ConstantController(), EpisodeConfig(command_jitter_penalty=penalty))
        for action in actions:
            episode.controller.action[:] = action
            episode.advance()
        episode.termination = "timeout"
        return episode.result()

    steady = [(1, -1)] * 3
    alternating = [(1, -1), (-1, 1), (1, -1)]
    assert measure(steady, .20).command_jitter == 0
    jittery = measure(alternating, .20)
    assert jittery.command_jitter == 1
    assert measure(alternating, 0).score - jittery.score == pytest.approx(.20)


def test_legacy_golden_values_preserve_arithmetic():
    state = RobotState(0, np.zeros(3), 0, 0.3, 0, np.array([0.3, 0.3]), np.zeros(2))
    sensors = np.zeros(15); sensors[6:9] = 1
    legacy = LegacyScorer()
    assert legacy.update(sensors, state, RobotConfig()) == pytest.approx(590.8923048454133)
    assert LegacyScorer().update(sensors, state, RobotConfig(), "green") == pytest.approx(10495)
    assert LegacyScorer().update(sensors, state, RobotConfig(), "red") == -500


def test_progress_does_not_repeat_or_jump_to_future_sections():
    track = TrackSpec("hairpin", (Section("straight", length=0.8), Section("arc", radius=0.14, angle=math.pi),
                                  Section("straight", length=0.8)), start=(-0.5, 0, 0))
    progress = RouteProgress(track, EpisodeConfig())
    for distance in np.arange(0, 0.31, 0.005):
        progress.update(track.route.point_at(distance))
    earned = progress.fraction
    progress.update(track.route.point_at(track.route.length - 0.10))
    assert progress.fraction == earned and not progress.completed
    for distance in np.arange(0.30, 0, -0.005):
        progress.update(track.route.point_at(distance))
    assert progress.fraction == earned


def test_branch_cannot_earn_progress_or_completion():
    track = replace(straight_track(), branches=(Branch(0.4, length=0.2),))
    progress = RouteProgress(track, EpisodeConfig())
    for distance in np.arange(0, 0.4, 0.005):
        progress.update(track.route.point_at(distance))
    earned = progress.fraction
    for offset in np.arange(0.01, 0.2, 0.005):
        progress.update(track.route.point_at(0.4) + [0, offset])
    assert progress.fraction <= earned + 0.01
    assert not progress.completed


@pytest.mark.parametrize("radius", [.1, .14])
def test_corner_cut_within_corridor_keeps_advancing_gates(radius):
    track = TrackSpec("rounded-corner", (Section("straight", length=.18),
                      Section("corner", angle=math.pi/2), Section("straight", length=.7)),
                      start=(0, 0, 0))
    progress = RouteProgress(track, EpisodeConfig())
    center_x = .18-radius
    approach = np.column_stack((np.linspace(0, center_x, 30), np.zeros(30)))
    angle = np.linspace(-math.pi/2, 0, 100)
    turn = np.column_stack((center_x+radius*np.cos(angle), radius+radius*np.sin(angle)))
    departure = np.column_stack((np.full(150, .18), np.linspace(radius, .7, 150)))
    for position in np.concatenate((approach, turn, departure)):
        progress.update(position)
    assert progress.completed, "A valid corner cut left the gate cursor stuck"
    assert progress.fraction == 1


def test_visible_line_resets_departure_and_genuine_loss_can_terminate():
    sim = Simulation()
    episode = Episode(sim, straight_track(), ConstantController())
    # The progress corridor can be missed temporarily during a turn.
    episode.route.inside_corridor = False
    episode.departure_time = episode.config.departure_seconds-sim.config.dt
    episode.observation[7] = 1
    episode._check_termination(0)
    assert episode.departure_time == 0 and not episode.done
    episode.observation[:] = 0
    for _ in range(round(episode.config.departure_seconds/sim.config.dt)):
        episode._check_termination(0)
    assert episode.termination == "off_route"


def test_episode_completion_failure_and_trace_are_explicit(tmp_path):
    sim = Simulation()
    trace = tmp_path / "trace.csv"
    result = run_episode(sim, straight_track(0.4), PIDController(), telemetry_path=trace)
    assert result.completed and result.progress == 1 and result.completion_time_s is not None
    assert result.lateral_rmse_m < 0.001
    with trace.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result.control_steps
    assert float(rows[1]["observation_time_s"]) == pytest.approx(float(rows[0]["time_s"]))
    assert [rows[0][f"next_sensor{i}"] for i in range(15)] == [rows[1][f"sensor{i}"] for i in range(15)]
    stalled = run_episode(sim, straight_track(), ConstantController(), EpisodeConfig(stall_seconds=0.2))
    assert not stalled.completed and stalled.termination == "stalled"
    assert stalled.completion_time_s is None and stalled.score < 0


def test_gaps_do_not_immediately_terminate_episode():
    track = replace(straight_track(0.6), gaps=((0.2, 0.25),))
    result = run_episode(Simulation(), track, ConstantController((0.5, 0.5)))
    assert result.completed and result.line_loss_events >= 1


def test_technical_failures_are_explicit():
    result = run_episode(Simulation(), straight_track(), ConstantController((np.nan, 0)))
    assert result.error and not result.completed
    assert result.termination == "controller_or_simulation_error"
    assert result.score == -1
