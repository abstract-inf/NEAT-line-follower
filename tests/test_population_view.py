"""Population observation must not replace live evaluations with a replay."""

from threading import Event
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from line_follower.experiments import write_json
from line_follower.simulation import Simulation
from line_follower.tracks import straight_track
from line_follower.viewer import _training_preview
from line_follower.population_view import PopulationScene


def test_preview_draws_entire_live_population_and_follows_without_resetting_orbit(tmp_path, monkeypatch):
    track = straight_track()
    track.save(tmp_path / "track.json")
    simulation = Simulation()
    simulation.reset(track)
    pose = simulation.data.qpos.tolist()
    write_json(tmp_path / "live" / "population.json", {
        "token": "fixture", "generation": 0, "backend": "gpu", "genomes": list(range(200)),
        "tracks": [str(tmp_path / "track.json")], "initial_qpos": pose})
    write_json(tmp_path / "live" / "gpu.json", {"token": "fixture", "robots": [
        {"genome": i, "track": 0, "qpos": pose, "time": 0, "status": "running"} for i in range(200)]})
    stop, frames, titles, robot_counts = Event(), [], [], []

    @contextmanager
    def fake_window(model, data, **kwargs):
        camera = mujoco.MjvCamera()
        scene = mujoco.MjvScene(model, maxgeom=100000)
        def sync():
            robot_counts.append(sum(geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == model.geom("chassis_geom").id
                                    for geom in scene.geoms[:scene.ngeom]))
            frames.append((scene.ngeom, camera.lookat.copy()))
            if len(frames) == 1:
                kwargs["key_callback"](ord("F"))
            elif len(frames) == 2:
                camera.azimuth, camera.elevation, camera.distance = 18, -42, 2.3
                # A new physical pose must translate the camera, not its orbit.
                data.qpos[0] += .1
                mujoco.mj_forward(model, data)
            else:
                assert (camera.azimuth, camera.elevation, camera.distance) == (18, -42, 2.3)
                assert camera.lookat[0] > frames[-2][1][0] + .09
                stop.set()
        yield SimpleNamespace(cam=camera, user_scn=scene, is_running=lambda: not stop.is_set(),
                              opt=mujoco.MjvOption(), lock=nullcontext, sync=sync, set_texts=titles.append, update_texture=lambda i: None)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    _training_preview(str(tmp_path), stop)
    assert robot_counts == [200]*len(frames), "Every genome must remain visible after simplifying display meshes"
    assert frames[0][0] < 5000, "Population rendering regressed to excessive cosmetic geometry"
    assert any("200" in str(title) and "live" in str(title).lower() for title in titles)


def test_grid_keeps_real_textures_and_places_200_independent_robot_instances():
    simulation = Simulation()
    simulation.reset(straight_track())
    simulation.enable_rendering()
    population = PopulationScene(simulation)
    scene = mujoco.MjvScene(simulation.model, maxgeom=100000)
    population.draw(scene, [(simulation.data.qpos.copy(), "running")]*200, grid=True)
    # Each tiled arena retains the renderer's actual texture IDs and mesh transform.
    floor = scene.geoms[scene.ngeom-1]
    assert np.any(floor.texid >= 0)
    assert np.linalg.norm(population.offset(0, 200, True)-population.offset(199, 200, True)) > 40
    assert sum(geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == simulation.model.geom("chassis_geom").id
               for geom in scene.geoms[:scene.ngeom]) == 200
    assert scene.ngeom < 5000
    assert all(not geom.transparent for geom in scene.geoms[:scene.ngeom])
    # Display tint and detail selection never mutate the simulation model.
    assert np.array_equal(simulation.model.geom_rgba, population.original_rgba)


def test_batched_sensor_rays_match_individual_pose_dependent_readings():
    from line_follower.simulation import sample_sensor_rays

    simulation = Simulation()
    simulation.reset(straight_track())
    origins, rotations, expected = [], [], []
    for lateral in np.linspace(-.08, .08, 31):
        simulation.data.qpos[1] = lateral
        mujoco.mj_forward(simulation.model, simulation.data)
        origins.append(simulation.data.site_xpos[simulation.site_ids].copy())
        rotations.append(simulation.data.site_xmat[simulation.site_ids].reshape(15, 3, 3).copy())
        expected.append(simulation.read_sensors())
    actual, _ = sample_sensor_rays(np.array(origins), np.array(rotations), simulation.mask,
                                  simulation.config.sensor_max_distance)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("stale", [False, True])
def test_observer_follows_active_track_after_previous_episodes_finish(tmp_path, monkeypatch, stale):
    """A live evaluation can advance batches while the old track remains static."""
    import time

    track = straight_track()
    for index in range(2):
        track.save(tmp_path / f"track-{index}.json")
    simulation = Simulation()
    simulation.reset(track)
    pose = simulation.data.qpos.tolist()
    write_json(tmp_path / "live/population.json", {
        "token": "fixture", "generation": 0, "backend": "gpu", "genomes": [10, 20],
        "tracks": [str(tmp_path / f"track-{index}.json") for index in range(2)], "initial_qpos": pose})
    write_json(tmp_path / "live/gpu-batch-0.json", {"token": "fixture", "robots": [
        {"genome": key, "track": 0, "qpos": pose, "time": 1, "status": "timeout"} for key in [10, 20]]})
    moved = pose.copy()
    moved[0] += .25
    write_json(tmp_path / "live/gpu-batch-2.json", {"token": "fixture", "robots": [
        {"genome": 10, "track": 1, "qpos": pose, "time": .1, "status": "completed"},
        {"genome": 20, "track": 1, "qpos": moved, "time": .2, "status": "running"}]})
    if stale:
        import os
        for path in (tmp_path / "live").glob("gpu-*.json"):
            os.utime(path, (time.time()-60, time.time()-60))
    stop, titles = Event(), []

    @contextmanager
    def fake_window(model, data, **kwargs):
        started = time.perf_counter()
        def sync():
            if time.perf_counter()-started > .3:
                stop.set()
        yield SimpleNamespace(cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(model, maxgeom=10000),
                              opt=mujoco.MjvOption(), is_running=lambda: not stop.is_set(), lock=nullcontext,
                              sync=sync, set_texts=titles.append, update_texture=lambda i: None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_window)
    _training_preview(str(tmp_path), stop)
    assert any("Track 2/2" in str(title) and "Genome 20" in str(title) for title in titles), (
        "Observer remained on completed track/robot while another batch was running")
    assert any("No fresh exports" in str(title) for title in titles) == stale
