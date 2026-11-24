from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import pytest

import compare_neat_pid_100 as comparison
import replay_neat_pid_100 as replay
from line_follower.controllers import NEATController
from line_follower.episodes import Episode
from line_follower.pid import ResearchPIDController
from line_follower.simulation import Simulation


@pytest.mark.parametrize("output", [comparison.OUTPUT, comparison.ROOT/"runs/neat134-vs-pid100-fullspeed-20261119"])
def test_replay_switches_frozen_policies_and_different_canvases_without_changing_sensor_masks(monkeypatch, output):
    if not (output/"protocol.json").exists():
        pytest.skip("Local frozen comparison is optional")
    tracks, robot, config, controllers = comparison.replay_setup(output)
    assert len({(t.arena_size, t.resolution) for t in tracks}) > 1
    assert isinstance(controllers[1], ResearchPIDController)
    expected = comparison.read_json(output/"frozen/pid-filtered-100.json")
    assert controllers[1].config.derivative_filter_s == expected["derivative_filter_s"]
    assert controllers[1].config.base == expected["base"]
    calls, masks, frames = [], [], []
    actions = [ord("C"), *([ord("N")]*9), ord("C"), ord("N")]

    def episode(simulation, track, controller, *args, **kwargs):
        value = Episode(simulation, track, controller, *args, **kwargs)
        assert simulation.mask.track == track
        calls.append((track.name, controller))
        return value

    original_upload = replay.upload_display

    def upload(simulation, track, *args):
        physical_mask = simulation.mask
        original_upload(simulation, track, *args)
        assert simulation.mask is physical_mask
        assert simulation.mask.track == track
        masks.append((track.arena_size, track.resolution))

    @contextmanager
    def fake_window(model, data, key_callback):
        def sync():
            index = len(frames)
            frames.append(index)
            if index < len(actions):
                key_callback(actions[index])
        yield SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: len(frames) <= len(actions), lock=nullcontext,
                              update_texture=lambda identity: None, sync=sync)

    monkeypatch.setattr(replay, "replay_setup", lambda output: (tracks, robot, config, controllers))
    monkeypatch.setattr(replay, "Episode", episode)
    monkeypatch.setattr(replay, "upload_display", upload)
    monkeypatch.setattr(replay.time, "sleep", lambda duration: None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    replay.view(output)
    assert {name for name, _ in calls} == {t.name for t in tracks}
    assert calls[0][1] is controllers[0] and calls[1][1] is controllers[1]
    assert calls[-1][1] is controllers[0]
    assert len(set(masks)) > 1


@pytest.mark.parametrize("output", [comparison.OUTPUT, comparison.ROOT/"runs/neat134-vs-pid100-fullspeed-20261119"])
def test_rendered_replay_keeps_the_same_episode_dynamics(output):
    if not (output/"protocol.json").exists():
        pytest.skip("Local frozen comparison is optional")
    tracks, robot, config, controllers = comparison.replay_setup(output)
    # One original mixed raster and one adaptive speed raster exercise both modes.
    for track in (tracks[0], tracks[4]):
        headless = Simulation(robot)
        rendered = Simulation(robot)
        from dataclasses import replace
        rendered.reset(replace(track, arena_size=10, resolution=.002))
        rendered.enable_rendering()
        reference = Episode(headless, track, controllers[1], config)
        controller = ResearchPIDController(controllers[1].config, robot)
        visual = Episode(rendered, track, controller, config)
        for _ in range(40):
            reference.advance()
            visual.advance()
        import numpy as np
        assert np.allclose(headless.data.qpos, rendered.data.qpos, atol=1e-12)
        assert np.array_equal(headless.observation, rendered.observation)


def test_controller_named_in_log_is_the_controller_actually_commanding_motors(monkeypatch, capsys):
    output = comparison.ROOT/"runs/neat134-vs-pid100-fullspeed-20261119"
    if not (output/"protocol.json").exists():
        pytest.skip("Local frozen comparison is optional")
    events, observed = [], []
    original_neat, original_pid = NEATController.act, ResearchPIDController.act

    def neat_act(self, observation, dt):
        events.append("neat")
        return original_neat(self, observation, dt)

    def pid_act(self, observation, dt):
        events.append("pid")
        return original_pid(self, observation, dt)

    actions = [ord("C"), ord("N"), ord("C"), ord("P"), ord("C")]

    @contextmanager
    def fake_window(model, data, key_callback):
        def sync():
            lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Track ")]
            assert len(lines) == 1
            expected = "pid" if "Frozen tuned filtered PID" in lines[0] else "neat"
            actual_class = "ResearchPIDController" if expected == "pid" else "NEATController"
            assert f"controller={actual_class}" in lines[0]
            filename = "pid-filtered-100.json" if expected == "pid" else "neat-generation-0134.pkl"
            assert f"file={filename}" in lines[0]
            assert events == [expected], f"Label/controller mismatch: {lines[0]}, actual={events}"
            observed.append(expected)
            events.clear()
            index = len(observed)-1
            if index < len(actions):
                key_callback(actions[index])
        yield SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: len(observed) <= len(actions), lock=nullcontext,
                              update_texture=lambda identity: None, sync=sync)

    monkeypatch.setattr(NEATController, "act", neat_act)
    monkeypatch.setattr(ResearchPIDController, "act", pid_act)
    monkeypatch.setattr(replay.time, "sleep", lambda duration: None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    replay.view(output)
    assert observed == ["neat", "pid", "pid", "neat", "neat", "pid"]


def test_replay_rejects_reversed_policy_roles_before_showing_a_label(monkeypatch):
    output = comparison.ROOT/"runs/neat134-vs-pid100-fullspeed-20261119"
    if not (output/"protocol.json").exists():
        pytest.skip("Local frozen comparison is optional")
    tracks, robot, config, controllers = comparison.replay_setup(output)
    monkeypatch.setattr(replay, "replay_setup", lambda run: (tracks, robot, config, controllers[::-1]))
    with pytest.raises(ValueError, match="controller roles are inconsistent"):
        replay.view(output)
