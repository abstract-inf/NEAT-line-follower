"""A training window must not depend on completion of the first generation."""

from contextlib import contextmanager, nullcontext
from threading import Event, Thread
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from line_follower.viewer import _training_preview
from line_follower.config import RobotConfig, EpisodeConfig
from line_follower.experiments import write_json
from line_follower.tracks import straight_track
from dataclasses import asdict


def test_window_opens_before_any_champion_exists(tmp_path, monkeypatch):
    opened, stop = Event(), Event()

    @contextmanager
    def fake_window(model, data, **kwargs):
        opened.set()
        yield SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: not stop.is_set(), lock=nullcontext,
                              sync=lambda: None, set_texts=lambda texts: None)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    thread = Thread(target=_training_preview, args=(str(tmp_path), stop), daemon=True)
    thread.start()
    try:
        assert opened.wait(2), "No window opened before a generation champion was saved"
    finally:
        stop.set()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_open_waiting_window_loads_policy_when_it_arrives(tmp_path, monkeypatch):
    opened, moving, stop = Event(), Event(), Event()

    @contextmanager
    def fake_window(model, data, **kwargs):
        opened.set()
        def sync():
            if np.any(data.ctrl):
                moving.set()
        yield SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: not stop.is_set(), lock=nullcontext,
                              sync=sync, set_texts=lambda texts: None, update_texture=lambda i: None)

    class FixedController:
        def reset(self):
            pass
        def act(self, observation, dt):
            return [.5,.5]

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    monkeypatch.setattr("line_follower.viewer.load_model", lambda path:
                        (FixedController(), {"robot": asdict(RobotConfig()), "episode": asdict(EpisodeConfig())}))
    thread = Thread(target=_training_preview, args=(str(tmp_path), stop), daemon=True)
    thread.start()
    try:
        assert opened.wait(3)
        assert not moving.is_set()
        straight_track().save(tmp_path / "track.json")
        write_json(tmp_path / "preview.json", {"generation": 0, "model": "fixture-only.pkl", "tracks": ["track.json"]})
        assert moving.wait(3), "Existing preview did not load the new policy"
    finally:
        stop.set()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_preview_preserves_mouse_camera_edits_between_frames(tmp_path, monkeypatch):
    stop = Event()
    observed = []
    user_position = np.array([.9, -.7, .2])

    @contextmanager
    def fake_window(model, data, **kwargs):
        camera = mujoco.MjvCamera()
        def sync():
            if observed:
                observed.append((camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation))
                stop.set()
            else:
                # Model MuJoCo's mouse handler changing the shared free camera.
                camera.lookat[:] = user_position
                camera.distance, camera.azimuth, camera.elevation = 2.3, 18, -42
                observed.append(None)
        yield SimpleNamespace(cam=camera, user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: not stop.is_set(), lock=nullcontext,
                              sync=sync, set_texts=lambda texts: None)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    _training_preview(str(tmp_path), stop)
    position, distance, azimuth, elevation = observed[1]
    assert np.array_equal(position, user_position), "Preview snapped the mouse-controlled camera back"
    assert (distance, azimuth, elevation) == (2.3, 18, -42)


@pytest.mark.parametrize("key", ["F", "C"])
def test_camera_shortcuts_reframe_once_then_allow_mouse_control(tmp_path, monkeypatch, key):
    stop = Event()
    frames = []
    user_position = np.array([.9, -.7, .2])

    @contextmanager
    def fake_window(model, data, **kwargs):
        camera = mujoco.MjvCamera()
        def sync():
            frames.append(camera.lookat.copy())
            if len(frames) == 1:
                camera.lookat[:] = user_position
                kwargs["key_callback"](ord(key))
            elif len(frames) == 2:
                assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = user_position
            else:
                stop.set()
        yield SimpleNamespace(cam=camera, user_scn=mujoco.MjvScene(model, maxgeom=1000),
                              is_running=lambda: not stop.is_set(), lock=nullcontext,
                              sync=sync, set_texts=lambda texts: None)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    _training_preview(str(tmp_path), stop)
    assert not np.array_equal(frames[1], user_position), "Shortcut did not reframe the camera"
    assert np.array_equal(frames[2], user_position), "Shortcut enabled unwanted continuous following"
