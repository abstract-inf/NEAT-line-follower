"""Compact mount geometry, actual contact clearance, and optical registration."""

from dataclasses import replace
import math
import mujoco
import numpy as np
import pytest

from line_follower.config import RobotConfig
from line_follower.simulation import Simulation, sensor_locations
from line_follower.tracks import straight_track


def test_integrated_offset_preserves_all_sensor_spacing_and_order():
    config = RobotConfig()
    compact = sensor_locations(config)
    farther = sensor_locations(replace(config, sensor_offset=0.120))
    assert compact.shape == (15, 3)
    assert compact[7, 0] == pytest.approx(0.090)
    assert np.all(compact[:, 0] > 0)
    assert np.all(np.diff(compact[:, 1]) < 0)
    assert np.allclose(np.linalg.norm(np.diff(compact, axis=0), axis=1),
                       np.linalg.norm(np.diff(farther, axis=0), axis=1))
    assert np.allclose(farther - compact, [0.030, 0, 0])
    assert sensor_locations(RobotConfig.paper())[7, 0] == pytest.approx(0.050)


def test_mount_dimensions_rejected_instead_of_overlapping_body():
    with pytest.raises(ValueError, match="exposed rigid neck"):
        Simulation(replace(RobotConfig(), sensor_offset=0.050))
    with pytest.raises(ValueError, match="exposed rigid neck"):
        Simulation(replace(RobotConfig(), sensor_offset=0.070))


def test_visual_sensor_packages_and_sites_are_registered():
    sim = Simulation()
    for index in range(15):
        site = sim.model.site(f"line_sensor_{index}")
        package = sim.model.geom(f"reflectance_package_{index}")
        assert np.allclose(site.pos[:2], package.pos[:2])
        assert package.pos[2] - package.size[2] == pytest.approx(site.pos[2])


def test_configurable_offset_does_not_change_other_physics():
    near = Simulation()
    far = Simulation(replace(RobotConfig(), sensor_offset=0.120))
    assert np.array_equal(near.model.body_mass, far.model.body_mass)
    assert np.array_equal(near.model.body_inertia, far.model.body_inertia)
    assert np.array_equal(near.model.actuator_forcerange, far.model.actuator_forcerange)


def test_ground_clearance_under_forward_reverse_turning_and_braking():
    sim = Simulation()
    sim.reset(straight_track(2))
    minimum = sim.board_ground_clearance()
    for action in ([1, 1], [0, 0], [-1, -1], [1, -1], [-1, 1], [0, 0]):
        for _ in range(60):
            sim.step(action)
            minimum = min(minimum, sim.board_ground_clearance())
            assert np.all(np.isfinite(sim.sensor_hits))
    assert minimum > 0.002
    assert sim.mount_wheel_clearance() >= 0.002


def test_every_sensor_samples_its_own_pose_transformed_floor_position():
    sim = Simulation()
    sim.reset(straight_track())
    sim.data.qpos[:2] = [.12, .21]
    yaw = .7
    sim.data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    mujoco.mj_forward(sim.model, sim.data)
    sim.read_sensors()
    pixels = np.floor(sim.mask.world_to_pixel(sim.sensor_hits[:, :2])).astype(int)
    sim.mask.field = np.full_like(sim.mask.field, 255)
    for index, (x,y) in enumerate(pixels):
        sim.mask.field[y,x] = 0
        readings = sim.read_sensors()
        assert readings[index] == 1
        assert np.sum(readings) == 1
        sim.mask.field[y,x] = 255


def test_motor_brackets_shafts_and_static_support_are_physically_connected():
    sim = Simulation()
    sim.reset(straight_track())
    for side in ("left", "right"):
        shaft, gearbox = [sim.model.geom(f"{side}_{name}").id for name in ("axle_shaft", "gearbox")]
        assert mujoco.mj_geomDistance(sim.model, sim.data, shaft, gearbox, 1, None) <= 0
        base = sim.model.geom(f"{side}_bracket_base").id
        assert min(mujoco.mj_geomDistance(sim.model, sim.data, base, board, 1, None)
                   for board in sim.board_geom_ids()) <= .000001
    contacts = set()
    normal_sum = 0.0
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        names = {sim.model.geom(i).name for i in contact.geom}
        contacts.update(names-{"floor"})
        force = np.empty(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, force)
        assert force[0] > 0
        normal_sum += force[0]
    assert contacts == {"left_tire", "right_tire", "caster_ball"}
    assert normal_sum == pytest.approx(sim.config.total_mass*9.81, rel=.02)
    com = sim.data.subtree_com[sim.chassis_id]
    root = sim.data.xpos[sim.chassis_id]
    assert 0 < com[0]-root[0] < sim.config.caster_offset


def test_visual_texture_uvs_and_sensor_field_have_same_world_orientation():
    track = replace(straight_track(), start=(-.5, .45, 0))
    sim = Simulation()
    sim.reset(track)
    sim.enable_rendering()
    mesh = sim.model.mesh("track_quad")
    geom = sim.model.geom("track_visual")
    vertices = sim.model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0]+mesh.vertnum[0]]
    world = vertices @ sim.data.geom_xmat[geom.id].reshape(3,3).T + sim.data.geom_xpos[geom.id]
    for index in range(mesh.faceadr[0], mesh.faceadr[0]+mesh.facenum[0]):
        points = world[sim.model.mesh_face[index]]
        uv = sim.model.mesh_texcoord[sim.model.mesh_texcoordadr[mesh.id]+sim.model.mesh_facetexcoord[index]]
        expected = np.column_stack(((points[:,0]+track.arena_size/2)/track.arena_size,
                                    (track.arena_size/2-points[:,1])/track.arena_size))
        assert np.allclose(uv, expected)
    texture = sim.model.tex_data.reshape(sim.mask.pixels,sim.mask.pixels,3)
    assert np.array_equal(texture[:,:,0], sim.mask.field)
    positive, negative = np.floor(sim.mask.world_to_pixel(np.array([[0,.45],[0,-.45]]))).astype(int)
    assert texture[positive[1],positive[0],0] == 0
    assert texture[negative[1],negative[0],0] == 255
