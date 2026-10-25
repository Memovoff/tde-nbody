import numpy as np
import pytest
import taichi as ti

from tde.config import SimConfig
from tde.core import Simulation, capture, compute_acc
from tde.initial_conditions import make_tde, orbit_velocity, sample_plummer_star


def test_force_matches_numpy_reference():
    n = 96
    rng = np.random.default_rng(123)
    pos_np = rng.normal(0.0, 2.0, (n, 3)).astype(np.float32)
    mass_np = rng.uniform(0.5, 1.5, n).astype(np.float32)
    eps2 = 0.01
    bh_mass = 100.0
    bh_eps2 = 0.02

    pos = ti.Vector.field(3, ti.f32, shape=n)
    mass = ti.field(ti.f32, shape=n)
    acc = ti.Vector.field(3, ti.f32, shape=n)
    pos.from_numpy(pos_np)
    mass.from_numpy(mass_np)

    compute_acc(pos, mass, acc, n, 1.0, eps2, bh_mass, bh_eps2)
    acc_np = acc.to_numpy()

    diff = pos_np[None, :, :] - pos_np[:, None, :]
    r2 = (diff**2).sum(axis=-1) + eps2
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    acc_ref = (
        mass_np[None, :, None] * diff * inv_r3[:, :, None]
    ).sum(axis=1)

    r2_bh = (pos_np**2).sum(axis=1) + bh_eps2
    acc_ref -= (
        bh_mass
        * pos_np
        / (r2_bh * np.sqrt(r2_bh))[:, None]
    )

    np.testing.assert_allclose(acc_np, acc_ref, rtol=3e-3, atol=1e-3)


def test_initial_conditions_are_sane():
    cfg = SimConfig(n=2_000, seed=3)
    data = make_tde(cfg)

    assert data["pos"].shape == (2_000, 3)
    assert data["vel"].shape == (2_000, 3)
    assert data["mass"].shape == (2_000,)
    assert data["color"].shape == (2_000, 3)
    assert np.isfinite(data["pos"]).all()
    assert np.isfinite(data["vel"]).all()
    assert np.isfinite(data["color"]).all()
    assert np.all(data["color"] >= 0.0)

    mass = data["mass"].astype(np.float64)
    pos = data["pos"].astype(np.float64)
    vel = data["vel"].astype(np.float64)
    total_mass = mass.sum()

    r0, v_r, v_t, _ = orbit_velocity(cfg)
    com = (pos * mass[:, None]).sum(axis=0) / total_mass
    bulk = (vel * mass[:, None]).sum(axis=0) / total_mass

    np.testing.assert_allclose(com, [r0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(bulk, [v_r, v_t, 0.0], atol=1e-3)

    # Parabolic orbit: total COM energy is zero.
    energy = 0.5 * np.dot(bulk, bulk) - cfg.G * cfg.bh_mass / r0
    assert abs(energy) < 1e-3

    # The star itself has no net momentum in its rest frame.
    internal_p = (
        (vel - bulk) * mass[:, None]
    ).sum(axis=0)
    assert np.linalg.norm(internal_p) < 1e-3


def test_plummer_star_is_bound_and_virialised():
    n = 500
    star = sample_plummer_star(
        n=n,
        mass=1.0,
        radius=1.0,
        G=1.0,
        rng=np.random.default_rng(5),
    )
    pos = star["pos"].astype(np.float64)
    vel = star["vel"].astype(np.float64)
    mass = star["mass"].astype(np.float64)

    kinetic = 0.5 * (mass * (vel**2).sum(axis=1)).sum()

    diff = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt((diff**2).sum(axis=-1) + 0.035**2)
    np.fill_diagonal(r, np.inf)
    potential = -0.5 * (
        mass[None, :] * mass[:, None] / r
    ).sum()

    virial = 2.0 * kinetic / abs(potential)
    assert 0.7 < virial < 1.3
    assert kinetic + potential < 0.0


def test_energy_is_conserved_before_periapsis():
    cfg = SimConfig(
        n=384,
        dt=0.0025,
        softening=0.035,
        seed=11,
        capture_radius=0.0,
    )
    sim = Simulation(cfg)

    e0 = sim.total_energy()
    for _ in range(200):
        sim.step()
    e1 = sim.total_energy()

    drift = abs((e1 - e0) / e0)
    assert np.isfinite(e0)
    assert drift < 0.01, f"energy drift too large: {drift:.4%}"


def test_capture_removes_mass_and_records_energy():
    n = 2
    pos = ti.Vector.field(3, ti.f32, shape=n)
    vel = ti.Vector.field(3, ti.f32, shape=n)
    mass = ti.field(ti.f32, shape=n)
    hist = ti.field(ti.f32, shape=8)
    captured_energy = ti.field(ti.f32, shape=())
    captured_mass = ti.field(ti.f32, shape=())

    pos.from_numpy(
        np.array([[0.01, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    )
    vel.from_numpy(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    )
    mass.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
    hist.fill(0.0)
    captured_energy[None] = 0.0
    captured_mass[None] = 0.0

    capture(
        pos,
        vel,
        mass,
        hist,
        captured_energy,
        captured_mass,
        n,
        0.2,
        0.01,
        0.05,
        8,
        1.0,
        100.0,
        0.05**2,
    )

    mass_np = mass.to_numpy()
    assert mass_np[0] == pytest.approx(0.0)
    assert mass_np[1] == pytest.approx(3.0)
    assert captured_mass[None] == pytest.approx(2.0)
    assert hist.to_numpy().sum() == pytest.approx(2.0)
    assert captured_energy[None] < 0.0
