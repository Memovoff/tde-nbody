"""Simulation parameters and physical units.

The code is scale-free: ``G = 1``, masses are in solar masses, lengths in
solar radii and time in the stellar dynamical time
``sqrt(R_sun^3 / (G M_sun)) ~ 26.5 min``.

The default black hole mass is deliberately modest (10^3 Msun).  It keeps the
star large enough relative to the tidal radius to be resolved by a few tens of
thousands of particles, while staying in the regime where the disruption
happens outside the event horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SUN_RADIUS_M = 6.957e8
SUN_MASS_KG = 1.989e30
G_SI = 6.67430e-11

#: One code time unit in seconds (stellar dynamical time of the Sun).
TIME_UNIT_S = math.sqrt(SUN_RADIUS_M**3 / (G_SI * SUN_MASS_KG))
#: One code time unit in hours.
TIME_UNIT_H = TIME_UNIT_S / 3600.0
#: One code velocity unit in km/s.
VELOCITY_UNIT_KM_S = SUN_RADIUS_M / TIME_UNIT_S / 1000.0


@dataclass
class SimConfig:
    """Physical and numerical parameters of the TDE simulation."""

    n: int = 20_000
    dt: float = 0.0025
    G: float = 1.0
    softening: float = 0.035
    seed: int = 7

    # Binary: central black hole + one star.
    bh_mass: float = 1_000.0
    star_mass: float = 1.0
    star_radius: float = 1.0

    # Orbit: beta = r_t / r_p.  beta > 1 means the star crosses the tidal
    # radius and is disrupted.
    penetration: float = 1.4
    start_radius_rt: float = 4.0

    # Numerical treatment of the black hole.
    bh_softening: float = 0.05
    capture_radius: float = 0.10

    # Accretion light curve histogram.
    capture_bins: int = 512
    capture_bin_width: float = 0.05

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.G <= 0:
            raise ValueError("G must be positive")
        if self.softening <= 0:
            raise ValueError("softening must be positive")
        if self.bh_mass <= 0:
            raise ValueError("bh_mass must be positive")
        if self.star_mass <= 0:
            raise ValueError("star_mass must be positive")
        if self.star_radius <= 0:
            raise ValueError("star_radius must be positive")
        if self.penetration <= 0:
            raise ValueError("penetration must be positive")
        if self.start_radius_rt <= 1.0:
            raise ValueError("start_radius_rt must be > 1")
        if self.bh_softening <= 0:
            raise ValueError("bh_softening must be positive")
        if self.capture_radius < 0:
            raise ValueError("capture_radius must be non-negative")
        if self.capture_bins < 1 or self.capture_bin_width <= 0:
            raise ValueError("invalid capture histogram")

    @property
    def tidal_radius(self) -> float:
        """Classical tidal radius r_t = R_* (M_BH / M_*)^(1/3)."""
        return self.star_radius * (self.bh_mass / self.star_mass) ** (1.0 / 3.0)

    @property
    def pericenter_radius(self) -> float:
        return self.tidal_radius / self.penetration

    @property
    def start_radius(self) -> float:
        return self.start_radius_rt * self.tidal_radius

    @property
    def star_dynamical_time(self) -> float:
        return math.sqrt(
            self.star_radius**3 / (self.G * self.star_mass)
        )

    @property
    def orbital_time_at_rt(self) -> float:
        return 2.0 * math.pi * math.sqrt(
            self.tidal_radius**3 / (self.G * self.bh_mass)
        )
