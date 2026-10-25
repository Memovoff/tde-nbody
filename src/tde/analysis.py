"""Post-processing: binding energies, fallback rate and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TIME_UNIT_H, SimConfig


def particle_energies(sim) -> dict[str, np.ndarray]:
    """Specific orbital energy of every surviving particle relative to the BH.

    The external black-hole potential is the same Plummer-softened potential
    used by the force kernel, so ``E = 0`` is the escape boundary.
    """
    cfg = sim.cfg
    mass = sim.mass.to_numpy()
    active = mass > 0.0
    pos = sim.pos.to_numpy()[active].astype(np.float64)
    vel = sim.vel.to_numpy()[active].astype(np.float64)
    m = mass[active].astype(np.float64)

    r = np.linalg.norm(pos, axis=1)
    energy = 0.5 * (vel * vel).sum(axis=1) - (
        cfg.G * cfg.bh_mass / np.sqrt(r * r + cfg.bh_softening**2)
    )
    return {"energy": energy, "mass": m, "pos": pos, "vel": vel}


def fallback_curve(
    sim,
    bins: int = 160,
    t_max: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Estimate the mass fallback rate from the debris energy distribution.

    For a bound orbit with specific energy ``E < 0`` the semi-major axis is
    ``a = -G M_BH / (2E)`` and the return time is one orbital period
    ``P = 2 pi sqrt(a^3 / (G M_BH))``.  Binning the debris mass by ``P``
    gives a prediction for the late-time accretion light curve.
    """
    cfg = sim.cfg
    data = particle_energies(sim)
    energy = data["energy"]
    mass = data["mass"]

    bound = energy < 0.0
    bound_energy = energy[bound]
    bound_mass = mass[bound]

    if bound_mass.size == 0:
        raise ValueError("no bound debris: cannot build a fallback curve")

    gm = cfg.G * cfg.bh_mass
    semi_major = -gm / (2.0 * bound_energy)
    period = 2.0 * np.pi * np.sqrt(semi_major**3 / gm)

    finite = np.isfinite(period) & (period > 0.0)
    period = period[finite]
    bound_mass = bound_mass[finite]
    if period.size == 0:
        raise ValueError("no finite bound periods")

    lo = float(np.percentile(period, 1.0))
    hi = float(np.percentile(period, 99.0)) if t_max is None else t_max
    if hi <= lo:
        hi = lo * 5.0
    edges = np.geomspace(max(lo * 0.8, 1e-4), hi * 1.2, bins + 1)
    hist, edges = np.histogram(period, bins=edges, weights=bound_mass)
    widths = np.diff(edges)
    rate = hist / np.maximum(widths, 1e-12)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # Fit the late-time slope where the rate is well defined.  The
    # asymptotic t^(-5/3) regime is reached after the most bound debris has
    # fallen back, so use the upper part of the cumulative mass distribution.
    order = np.argsort(period)
    cumulative = np.cumsum(bound_mass[order])
    total_bound = cumulative[-1]
    frac = np.interp(
        centers,
        period[order],
        cumulative / max(total_bound, 1e-30),
    )
    good = (rate > 0.0) & (centers > 0.0)
    fit = good & (frac > 0.70) & (frac < 0.99)
    slope = float("nan")
    fit_time = np.empty(0, dtype=np.float64)
    fit_rate = np.empty(0, dtype=np.float64)
    if fit.sum() >= 4:
        fit_time = centers[fit]
        fit_rate = rate[fit]
        slope = float(
            np.polyfit(np.log(fit_time), np.log(fit_rate), 1)[0]
        )

    unbound_mass = float(mass[energy >= 0.0].sum())
    captured_mass = float(sim.captured_mass[None])
    total_mass = float(cfg.star_mass)

    # Energy histogram for the lower panel of the diagnostic figure.  Almost
    # all debris sits in a narrow band around E = 0, so pick the range from
    # robust quantiles instead of a fixed, mostly empty window.
    e_scale = cfg.G * cfg.bh_mass / cfg.tidal_radius
    scaled_energy = energy / e_scale
    e_lo, e_hi = np.percentile(scaled_energy, [0.2, 99.8])
    span = max(float(e_hi - e_lo), 0.15)
    pad = 0.10 * span
    e_range = (
        min(float(e_lo) - pad, -0.02),
        max(float(e_hi) + pad, 0.02),
    )
    e_hist, e_edges = np.histogram(
        scaled_energy,
        bins=100,
        range=e_range,
        weights=mass,
    )

    return {
        "time": centers,
        "time_hours": centers * TIME_UNIT_H,
        "rate": rate,
        "rate_per_hour": rate / TIME_UNIT_H,
        "mass": hist,
        "edges": edges,
        "period": period,
        "period_mass": bound_mass,
        "energy": energy,
        "energy_mass": mass,
        "energy_hist": e_hist,
        "energy_edges": e_edges,
        "energy_scale": e_scale,
        "slope": slope,
        "fit_time": fit_time,
        "fit_rate": fit_rate,
        "unbound_fraction": unbound_mass / total_mass,
        "bound_fraction": float(bound_mass.sum()) / total_mass,
        "captured_fraction": captured_mass / total_mass,
        "total_mass": total_mass,
        "t_min": float(period.min()),
        "t_max": float(hi),
        "period_max": float(period.max()),
        "median_period": float(np.median(period)),
    }


def summary(sim) -> dict[str, float]:
    """A compact dictionary of the current simulation state."""
    cfg = sim.cfg
    data = particle_energies(sim)
    energy = data["energy"]
    mass = data["mass"]
    total = float(cfg.star_mass)
    com = sim.star_com()
    r = float(np.linalg.norm(com))
    e_scale = cfg.G * cfg.bh_mass / cfg.tidal_radius
    return {
        "time": float(sim.time),
        "steps": float(sim.steps),
        "bh_distance": r,
        "debris_r90": sim.debris_radius(90.0),
        "unbound_fraction": float(mass[energy >= 0.0].sum() / total),
        "bound_fraction": float(mass[energy < 0.0].sum() / total),
        "captured_mass": float(sim.captured_mass[None]),
        "captured_fraction": float(sim.captured_mass[None] / total),
        "remaining_mass": float(sim.star_mass()),
        "energy_spread": float(np.std(energy / e_scale)),
        "total_energy": sim.total_energy(),
        "captured_energy": float(sim.captured_energy[None]),
        "tidal_radius": cfg.tidal_radius,
        "pericenter": cfg.pericenter_radius,
    }


def save_fallback_csv(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "time_code,time_hours,rate_msun_per_code,"
            "rate_msun_per_hour,mass_in_bin\n"
        )
        for t, th, rate, rate_h, m in zip(
            result["time"],
            result["time_hours"],
            result["rate"],
            result["rate_per_hour"],
            result["mass"],
        ):
            handle.write(
                f"{t:.8e},{th:.8e},{rate:.8e},{rate_h:.8e},{m:.8e}\n"
            )


def save_fallback_plot(
    result: dict,
    path: str | Path,
    cfg: SimConfig,
    title: str = "Tidal disruption event",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax_rate, ax_energy) = plt.subplots(
        2,
        1,
        figsize=(9.2, 8.0),
        dpi=160,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    t = result["time_hours"]
    rate = result["rate_per_hour"]
    good = (rate > 0.0) & np.isfinite(rate)
    ax_rate.loglog(t[good], rate[good], color="#ffb066", linewidth=1.8)
    fit_t = result["fit_time"] * TIME_UNIT_H
    fit_rate = result["fit_rate"] / TIME_UNIT_H
    if fit_t.size >= 2:
        t_ref = float(np.exp(np.mean(np.log(fit_t))))
        rate_ref = float(np.exp(np.mean(np.log(fit_rate))))
        ref = rate_ref * (t[good] / t_ref) ** (-5.0 / 3.0)
        ax_rate.loglog(
            t[good],
            ref,
            color="#7cc7ff",
            linewidth=1.2,
            linestyle="--",
            label=r"$t^{-5/3}$ reference",
        )
    ax_rate.set_xlabel("time after disruption, hours")
    ax_rate.set_ylabel(r"fallback rate, $M_\odot$ / hour")
    ax_rate.set_title(
        f"{title}: predicted fallback light curve",
        fontsize=12,
    )
    slope = result["slope"]
    ax_rate.text(
        0.03,
        0.95,
        (
            f"unbound debris: {result['unbound_fraction'] * 100:.0f}%\n"
            f"captured: {result['captured_fraction'] * 100:.1f}%\n"
            f"fitted slope: {slope:.2f}"
        ),
        transform=ax_rate.transAxes,
        va="top",
        fontsize=9.5,
        color="#d8dde8",
        bbox={"facecolor": "#141821", "edgecolor": "#3a4250", "alpha": 0.9},
    )
    ax_rate.legend(loc="lower left", fontsize=9, frameon=False)
    ax_rate.grid(alpha=0.18, which="both")

    e_edges = result["energy_edges"]
    e_hist = result["energy_hist"]
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    ax_energy.step(
        e_centers,
        e_hist,
        where="mid",
        color="#ff8a5c",
        linewidth=1.5,
    )
    ax_energy.axvline(0.0, color="#7cc7ff", linewidth=1.2, linestyle="--")
    ax_energy.set_xlabel(
        r"specific energy  $E \,/\, (G M_\mathrm{BH}/r_t)$"
    )
    ax_energy.set_ylabel(r"mass per bin, $M_\odot$")
    ax_energy.set_xlim(e_edges[0], e_edges[-1])
    ax_energy.set_yscale("log")
    ax_energy.set_title("Debris energy distribution", fontsize=11)
    ax_energy.text(
        0.02,
        0.92,
        "bound  ←   E = 0   →  unbound",
        transform=ax_energy.transAxes,
        fontsize=9,
        color="#aeb7c6",
        va="top",
    )
    ax_energy.grid(alpha=0.18, which="both")

    fig.suptitle(
        (
            f"$M_\\mathrm{{BH}} = {cfg.bh_mass:.0f}\\,M_\\odot$, "
            f"$\\beta = {cfg.penetration:.2f}$, "
            f"$N = {cfg.n:,}$"
        ),
        fontsize=10,
        color="#aeb7c6",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
