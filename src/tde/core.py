"""Taichi core: softened N-body gravity, an external black hole and leapfrog.

The star is represented by equal-mass particles.  Their mutual gravity is
computed by direct O(N^2) summation on the GPU; the central black hole is a
fixed external Plummer-softened point mass.  A kick-drift-kick leapfrog keeps
the dynamics symplectic.

Particles that enter ``capture_radius`` are removed from the dynamical state.
Their mass and black-hole energy are stored in a reservoir, so the reported
total energy still includes the accreted debris.
"""

import numpy as np
import taichi as ti

from .config import SimConfig
from .initial_conditions import make_tde


@ti.kernel
def compute_acc(
    pos: ti.template(),
    mass: ti.template(),
    acc: ti.template(),
    n: ti.i32,
    G: ti.f32,
    eps2: ti.f32,
    bh_mass: ti.f32,
    bh_eps2: ti.f32,
):
    """Direct O(N^2) acceleration from the star plus a central point mass."""
    for i in range(n):
        if mass[i] > 0.0:
            p_i = pos[i]
            a = ti.Vector([0.0, 0.0, 0.0])

            # Self-gravity of the star.  Captured particles have mass == 0,
            # so they contribute nothing and need no branch here.
            for j in range(n):
                d = pos[j] - p_i
                r2 = d.dot(d) + eps2
                inv_r3 = 1.0 / (r2 * ti.sqrt(r2))
                a += (G * mass[j] * inv_r3) * d

            # Fixed central black hole, Plummer-softened at the origin.
            r2_bh = p_i.dot(p_i) + bh_eps2
            inv_r3_bh = 1.0 / (r2_bh * ti.sqrt(r2_bh))
            a += (-G * bh_mass * inv_r3_bh) * p_i

            acc[i] = a
        else:
            acc[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def kick(vel: ti.template(), acc: ti.template(), mass: ti.template(),
         half_dt: ti.f32, n: ti.i32):
    for i in range(n):
        if mass[i] > 0.0:
            vel[i] += acc[i] * half_dt


@ti.kernel
def drift(pos: ti.template(), vel: ti.template(), mass: ti.template(),
          dt: ti.f32, n: ti.i32):
    for i in range(n):
        if mass[i] > 0.0:
            pos[i] += vel[i] * dt


@ti.kernel
def capture(
    pos: ti.template(),
    vel: ti.template(),
    mass: ti.template(),
    hist: ti.template(),
    captured_energy: ti.template(),
    captured_mass: ti.template(),
    n: ti.i32,
    time: ti.f32,
    r_capture2: ti.f32,
    bin_width: ti.f32,
    n_bins: ti.i32,
    G: ti.f32,
    bh_mass: ti.f32,
    bh_eps2: ti.f32,
):
    """Accrete particles that fall inside the capture radius."""
    for i in range(n):
        if mass[i] > 0.0:
            r2 = pos[i].dot(pos[i])
            if r2 < r_capture2:
                m = mass[i]
                r = ti.sqrt(r2 + bh_eps2)
                energy = 0.5 * m * vel[i].dot(vel[i]) - (
                    G * bh_mass * m / r
                )
                ti.atomic_add(captured_energy[None], energy)
                ti.atomic_add(captured_mass[None], m)
                bin_id = ti.cast(time / bin_width, ti.i32)
                bin_id = ti.max(0, ti.min(bin_id, n_bins - 1))
                ti.atomic_add(hist[bin_id], m)
                mass[i] = 0.0


@ti.kernel
def update_colors(
    pos: ti.template(),
    vel: ti.template(),
    mass: ti.template(),
    base_color: ti.template(),
    color: ti.template(),
    n: ti.i32,
    com_x: ti.f32,
    com_y: ti.f32,
    com_z: ti.f32,
    star_radius: ti.f32,
    energy_scale: ti.f32,
    G: ti.f32,
    bh_mass: ti.f32,
    bh_eps2: ti.f32,
):
    """Tint unbound, separated debris blue; keep bound debris warm."""
    blue = ti.Vector([0.32, 0.62, 1.0])
    for i in range(n):
        if mass[i] > 0.0:
            p = pos[i]
            r = ti.sqrt(p.dot(p) + bh_eps2)
            energy = 0.5 * vel[i].dot(vel[i]) - G * bh_mass / r
            dx = p.x - com_x
            dy = p.y - com_y
            dz = p.z - com_z
            dist = ti.sqrt(dx * dx + dy * dy + dz * dz)

            a_energy = ti.max(0.0, ti.min(energy / (0.22 * energy_scale), 1.0))
            a_dist = ti.max(
                0.0,
                ti.min((dist - 1.4 * star_radius) / (1.8 * star_radius), 1.0),
            )
            alpha = a_energy * a_dist
            color[i] = base_color[i] * (1.0 - alpha) + blue * alpha


@ti.kernel
def kinetic_energy(vel: ti.template(), mass: ti.template(), n: ti.i32) -> ti.f32:
    energy = 0.0
    for i in range(n):
        energy += 0.5 * mass[i] * vel[i].dot(vel[i])
    return energy


@ti.kernel
def star_potential(
    pos: ti.template(),
    mass: ti.template(),
    n: ti.i32,
    G: ti.f32,
    eps2: ti.f32,
) -> ti.f32:
    energy = 0.0
    for i in range(n):
        if mass[i] > 0.0:
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = ti.sqrt(d.dot(d) + eps2)
                energy -= G * mass[i] * mass[j] / r
    return energy


@ti.kernel
def bh_potential(
    pos: ti.template(),
    mass: ti.template(),
    n: ti.i32,
    G: ti.f32,
    bh_mass: ti.f32,
    bh_eps2: ti.f32,
) -> ti.f32:
    energy = 0.0
    for i in range(n):
        if mass[i] > 0.0:
            r = ti.sqrt(pos[i].dot(pos[i]) + bh_eps2)
            energy -= G * bh_mass * mass[i] / r
    return energy


class Simulation:
    """State and time stepping for the tidal disruption event."""

    def __init__(self, cfg: SimConfig | None = None, seed: int | None = None):
        self.cfg = cfg or SimConfig()
        self.n = self.cfg.n

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.mass = ti.field(dtype=ti.f32, shape=self.n)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.base_color = ti.Vector.field(3, dtype=ti.f32, shape=self.n)

        self.capture_hist = ti.field(
            dtype=ti.f32, shape=self.cfg.capture_bins
        )
        self.captured_energy = ti.field(dtype=ti.f32, shape=())
        self.captured_mass = ti.field(dtype=ti.f32, shape=())

        self.time = 0.0
        self.steps = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-sample the initial conditions and reset diagnostics."""
        seed = self.cfg.seed if seed is None else seed
        data = make_tde(self.cfg, seed=seed)
        self.pos.from_numpy(data["pos"])
        self.vel.from_numpy(data["vel"])
        self.mass.from_numpy(data["mass"])
        self.color.from_numpy(data["color"])
        self.base_color.from_numpy(data["color"])
        self.capture_hist.fill(0.0)
        self.captured_energy[None] = 0.0
        self.captured_mass[None] = 0.0
        self.time = 0.0
        self.steps = 0
        self._compute_acc()

    def _compute_acc(self) -> None:
        compute_acc(
            self.pos,
            self.mass,
            self.acc,
            self.n,
            self.cfg.G,
            self.cfg.softening**2,
            self.cfg.bh_mass,
            self.cfg.bh_softening**2,
        )

    def step(self) -> None:
        """One kick-drift-kick leapfrog step plus accretion bookkeeping."""
        half_dt = 0.5 * self.cfg.dt
        kick(self.vel, self.acc, self.mass, half_dt, self.n)
        drift(self.pos, self.vel, self.mass, self.cfg.dt, self.n)
        self._compute_acc()
        kick(self.vel, self.acc, self.mass, half_dt, self.n)

        self.time += self.cfg.dt
        self.steps += 1

        if self.cfg.capture_radius > 0.0:
            capture(
                self.pos,
                self.vel,
                self.mass,
                self.capture_hist,
                self.captured_energy,
                self.captured_mass,
                self.n,
                self.time,
                self.cfg.capture_radius**2,
                self.cfg.capture_bin_width,
                self.cfg.capture_bins,
                self.cfg.G,
                self.cfg.bh_mass,
                self.cfg.bh_softening**2,
            )

    def kinetic_energy(self) -> float:
        return float(kinetic_energy(self.vel, self.mass, self.n))

    def potential_energy(self) -> float:
        """Star self-gravity plus the external black-hole potential."""
        return float(
            star_potential(
                self.pos,
                self.mass,
                self.n,
                self.cfg.G,
                self.cfg.softening**2,
            )
            + bh_potential(
                self.pos,
                self.mass,
                self.n,
                self.cfg.G,
                self.cfg.bh_mass,
                self.cfg.bh_softening**2,
            )
        )

    def total_energy(self) -> float:
        """Total energy including the accreted-particle reservoir."""
        return (
            self.kinetic_energy()
            + self.potential_energy()
            + float(self.captured_energy[None])
        )

    def star_com(self) -> np.ndarray:
        """Mass-weighted centre of mass of the surviving star/debris."""
        mass = self.mass.to_numpy()
        active = mass > 0.0
        if not active.any():
            return np.zeros(3, dtype=np.float64)
        pos = self.pos.to_numpy()[active].astype(np.float64)
        m = mass[active].astype(np.float64)
        return (pos * m[:, None]).sum(axis=0) / m.sum()

    def star_mass(self) -> float:
        mass = self.mass.to_numpy()
        return float(mass.sum())

    def debris_radius(self, percentile: float = 90.0) -> float:
        """Percentile radius of the surviving particles around their COM."""
        mass = self.mass.to_numpy()
        active = mass > 0.0
        if not active.any():
            return 0.0
        pos = self.pos.to_numpy()[active].astype(np.float64)
        com = self.star_com()
        r = np.linalg.norm(pos - com, axis=1)
        return float(np.percentile(r, percentile))

    def capture_rate(self, window: float = 1.0) -> float:
        """Recent accreted mass per unit code time."""
        hist = self.capture_hist.to_numpy()
        idx = int(self.time / self.cfg.capture_bin_width)
        lo = max(0, idx - int(window / self.cfg.capture_bin_width) + 1)
        hi = min(len(hist), idx + 1)
        if hi <= lo:
            return 0.0
        return float(hist[lo:hi].sum() / window)

    def update_colors(self) -> None:
        """Tint escaping, unbound debris blue for the renderer."""
        com = self.star_com()
        energy_scale = (
            self.cfg.G
            * self.cfg.bh_mass
            * self.cfg.star_radius
            / self.cfg.pericenter_radius**2
        )
        update_colors(
            self.pos,
            self.vel,
            self.mass,
            self.base_color,
            self.color,
            self.n,
            float(com[0]),
            float(com[1]),
            float(com[2]),
            self.cfg.star_radius,
            energy_scale,
            self.cfg.G,
            self.cfg.bh_mass,
            self.cfg.bh_softening**2,
        )

    def state(self) -> dict[str, np.ndarray]:
        """Copy the full dynamical state to NumPy."""
        return {
            "pos": self.pos.to_numpy(),
            "vel": self.vel.to_numpy(),
            "mass": self.mass.to_numpy(),
            "color": self.color.to_numpy(),
            "capture_hist": self.capture_hist.to_numpy(),
        }
