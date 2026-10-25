"""Initial conditions: a Plummer star on a parabolic orbit around a black hole.

The star is sampled from an isotropic Plummer model.  Its centre of mass is
placed on a zero-energy (parabolic) Keplerian orbit with a periapsis
``r_p = r_t / beta``.  The internal velocities are sampled from the exact
Plummer distribution function, so the star starts close to virial
equilibrium.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SimConfig

Array = np.ndarray


def sample_plummer_star(
    n: int,
    mass: float,
    radius: float,
    G: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Array]:
    """Sample an isotropic Plummer sphere.

    ``radius`` is the nominal stellar radius; the Plummer scale length is
    ``radius / 5``, so about 95% of the mass lies inside ``radius``.  A faint
    halo out to ``3 * radius`` is kept to avoid a sharp edge.
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = rng or np.random.default_rng(0)

    a = radius / 5.0

    # Inverse CDF of the Plummer density profile.  The clipping at 3 R_*
    # removes a tiny high-radius tail (well below 1% of the mass).
    u = np.clip(rng.random(n), 1e-9, 1.0 - 1e-9)
    r = a / np.sqrt(np.maximum(u ** (-2.0 / 3.0) - 1.0, 1e-12))
    r = np.minimum(r, 3.0 * radius)

    # Exact Plummer velocity distribution at radius r:
    #   f(v) ~ (1 - v^2/v_esc^2)^(7/2)
    # The speed fraction y = v^2/v_esc^2 follows Beta(3/2, 9/2).
    v_esc = np.sqrt(
        2.0 * G * mass / np.sqrt(r * r + a * a)
    )
    y = rng.beta(1.5, 4.5, n)
    speed = v_esc * np.sqrt(y)

    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)

    pos = np.column_stack(
        [
            r * sin_theta * np.cos(phi),
            r * sin_theta * np.sin(phi),
            r * cos_theta,
        ]
    )

    # Independent random directions for the velocity vectors.
    cos_theta_v = rng.uniform(-1.0, 1.0, n)
    sin_theta_v = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_v * cos_theta_v))
    phi_v = rng.uniform(0.0, 2.0 * np.pi, n)
    vel = np.column_stack(
        [
            speed * sin_theta_v * np.cos(phi_v),
            speed * sin_theta_v * np.sin(phi_v),
            speed * cos_theta_v,
        ]
    )

    # Remove the sampling offset and net momentum.  This keeps the star
    # centred and avoids a slow drift before the encounter.
    m = np.full(n, mass / n, dtype=np.float64)
    com = (pos * m[:, None]).sum(axis=0) / m.sum()
    p = (vel * m[:, None]).sum(axis=0)
    pos -= com
    vel -= p / m.sum()

    # Temperature-like colour: hot, white core -> cooler orange envelope.
    t = np.clip(r / radius, 0.0, 1.0)[:, None]
    core = np.array([1.00, 0.94, 0.80], dtype=np.float64)
    outer = np.array([1.00, 0.36, 0.10], dtype=np.float64)
    color = core * (1.0 - t) + outer * t
    color *= rng.uniform(0.75, 1.25, (n, 1))

    return {
        "pos": np.ascontiguousarray(pos, dtype=np.float32),
        "vel": np.ascontiguousarray(vel, dtype=np.float32),
        "mass": np.ascontiguousarray(m, dtype=np.float32),
        "color": np.ascontiguousarray(color, dtype=np.float32),
        "radius": np.ascontiguousarray(r, dtype=np.float32),
    }


def orbit_velocity(
    cfg: SimConfig,
    radius: float | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(r0, v_radial, v_tangential, angular_momentum)``.

    The orbit is parabolic: ``E = v^2/2 - G M_BH / r = 0``.  The velocity is
    split into a negative radial component (infall) and a positive tangential
    component in the xy plane.
    """
    r0 = cfg.start_radius if radius is None else radius
    rp = cfg.pericenter_radius
    gm = cfg.G * cfg.bh_mass
    if r0 <= rp:
        raise ValueError("start radius must be larger than pericenter")

    angular_momentum = np.sqrt(2.0 * gm * rp)
    v_t = angular_momentum / r0
    v_r = -np.sqrt(max(2.0 * gm / r0 - v_t * v_t, 0.0))
    return r0, float(v_r), float(v_t), float(angular_momentum)


def make_tde(
    cfg: SimConfig | None = None,
    n: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the star and place it on the parabolic orbit.

    The returned dictionary contains the particle arrays expected by
    :class:`tde.core.Simulation`, plus a ``meta`` dictionary with the
    characteristic radii and velocities.
    """
    cfg = cfg or SimConfig()
    n = cfg.n if n is None else n
    seed = cfg.seed if seed is None else seed
    if n < 2:
        raise ValueError("n must be at least 2")

    rng = np.random.default_rng(seed)
    star = sample_plummer_star(
        n=n,
        mass=cfg.star_mass,
        radius=cfg.star_radius,
        G=cfg.G,
        rng=rng,
    )

    r0, v_r, v_t, angular_momentum = orbit_velocity(cfg)
    offset = np.array([r0, 0.0, 0.0], dtype=np.float64)
    bulk_velocity = np.array([v_r, v_t, 0.0], dtype=np.float64)

    pos = star["pos"].astype(np.float64) + offset
    vel = star["vel"].astype(np.float64) + bulk_velocity

    meta = {
        "tidal_radius": cfg.tidal_radius,
        "pericenter": cfg.pericenter_radius,
        "start_radius": r0,
        "v_radial": v_r,
        "v_tangential": v_t,
        "angular_momentum": angular_momentum,
        "star_radius": cfg.star_radius,
    }

    return {
        "pos": np.ascontiguousarray(pos, dtype=np.float32),
        "vel": np.ascontiguousarray(vel, dtype=np.float32),
        "mass": star["mass"],
        "color": star["color"],
        "radius": star["radius"],
        "meta": meta,
    }
