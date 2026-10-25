"""Headless runs: preview stills, frame sequences, diagnostics and benchmarks."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import taichi as ti

from .analysis import (
    fallback_curve,
    save_fallback_csv,
    save_fallback_plot,
    summary,
)
from .config import TIME_UNIT_H, SimConfig
from .core import Simulation
from .render import FollowCamera, Renderer


def _save_energy_plot(
    times: list[float], energies: list[float], path: Path, n: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_arr = np.asarray(times)
    energies_arr = np.asarray(energies)
    drift = (energies_arr - energies_arr[0]) / max(
        abs(energies_arr[0]), 1e-12
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    ax.plot(times_arr, drift * 100.0, color="#7cc7ff", linewidth=1.8)
    ax.axhline(0.0, color="#444a55", linewidth=1.0, linestyle="--")
    max_drift = float(np.max(np.abs(drift)) * 100.0)
    ax.text(
        0.03,
        0.06,
        f"max |drift| = {max_drift:.3f}%",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#aeb7c6",
    )
    ax.set_xlabel("simulation time")
    ax.set_ylabel("relative energy drift, %")
    ax.set_title(f"Energy conservation (N = {n:,})")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_headless(
    cfg: SimConfig | None = None,
    steps: int = 5_600,
    width: int = 1280,
    height: int = 720,
    exposure: float = 1.05,
    bloom_strength: float = 0.85,
    out: str | Path | None = "media/preview.png",
    frames_dir: str | Path | None = None,
    frame_every: int = 16,
    energy_csv: str | Path | None = None,
    fallback_csv: str | Path | None = "media/fallback.csv",
    fallback_plot: str | Path | None = "media/fallback.png",
    diag_every: int = 200,
    azimuth_deg: float = -38.0,
    elevation_deg: float = 28.0,
    zoom: float = 1.0,
    follow: bool = True,
    smoothing: float = 0.18,
) -> dict[str, float]:
    """Run without a window and optionally save frames and diagnostics."""
    cfg = cfg or SimConfig()
    sim = Simulation(cfg)
    renderer = Renderer(
        width,
        height,
        exposure=exposure,
        bloom_strength=bloom_strength,
    )
    camera = FollowCamera(
        width,
        height,
        star_radius=cfg.star_radius,
        start_radius=cfg.start_radius,
        zoom=zoom,
        follow=follow,
        smoothing=smoothing,
    )
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    camera.update(sim, azimuth, elevation)

    frames_path = Path(frames_dir) if frames_dir else None
    if frames_path is not None:
        frames_path.mkdir(parents=True, exist_ok=True)

    def save_frame(step: int) -> None:
        if frames_path is None:
            return
        sim.update_colors()
        renderer.render(sim, azimuth, elevation, camera)
        renderer.save(frames_path / f"frame_{step:05d}.png")

    make_frames = frames_path is not None and frame_every > 0
    camera_every = max(1, frame_every // 2) if make_frames else steps + 1

    energies: list[float] = []
    times: list[float] = []
    if energy_csv is not None:
        energies.append(sim.total_energy())
        times.append(0.0)

    if make_frames:
        save_frame(0)

    start = time.perf_counter()
    for step in range(1, steps + 1):
        sim.step()

        if make_frames:
            if step % camera_every == 0 or step % frame_every == 0:
                camera.update(sim, azimuth, elevation)
            if step % frame_every == 0:
                save_frame(step)

        if (
            energy_csv is not None
            and diag_every > 0
            and step % diag_every == 0
        ):
            energies.append(sim.total_energy())
            times.append(sim.time)
    elapsed = time.perf_counter() - start

    camera.update(sim, azimuth, elevation)
    sim.update_colors()
    renderer.render(sim, azimuth, elevation, camera)
    if out is not None:
        renderer.save(out)

    if energy_csv is not None:
        energy_path = Path(energy_csv)
        energy_path.parent.mkdir(parents=True, exist_ok=True)
        with energy_path.open("w", encoding="utf-8") as handle:
            handle.write("time,energy,relative_drift\n")
            e0 = energies[0]
            for t, e in zip(times, energies):
                drift = (e - e0) / max(abs(e0), 1e-12)
                handle.write(f"{t:.8f},{e:.10e},{drift:.10e}\n")
        _save_energy_plot(
            times, energies, energy_path.with_suffix(".png"), cfg.n
        )

    result = None
    try:
        result = fallback_curve(sim)
    except ValueError as exc:
        print(f"fallback curve skipped: {exc}")

    if result is not None:
        if fallback_csv is not None:
            save_fallback_csv(result, fallback_csv)
        if fallback_plot is not None:
            save_fallback_plot(result, fallback_plot, cfg)

    stats = summary(sim)
    stats["wall_time"] = elapsed
    stats["steps_per_second"] = steps / elapsed if elapsed > 0 else float("inf")

    print(
        f"n={cfg.n:,}  steps={steps}  t={sim.time:.2f}  "
        f"wall={elapsed:.2f}s  rate={stats['steps_per_second']:.1f} steps/s"
    )
    print(
        f"unbound={stats['unbound_fraction'] * 100:.1f}%  "
        f"bound={stats['bound_fraction'] * 100:.1f}%  "
        f"captured={stats['captured_mass']:.3e}"
    )
    if result is not None:
        print(
            f"fallback: slope={result['slope']:.2f}  "
            f"median={result['median_period'] * TIME_UNIT_H:.1f} h  "
            f"range={result['t_min'] * TIME_UNIT_H:.1f}.."
            f"{result['t_max'] * TIME_UNIT_H:.0f} h"
        )
    if out is not None:
        print(f"saved preview: {out}")
    if frames_path is not None:
        print(f"saved frames:  {frames_path}")
    if energy_csv is not None:
        print(f"saved energy:  {energy_csv}")
    if fallback_csv is not None and result is not None:
        print(f"saved fallback: {fallback_csv}")
    if fallback_plot is not None and result is not None:
        print(f"saved plot:     {fallback_plot}")

    return stats


def run_benchmark(
    sizes: list[int],
    steps: int = 20,
    base_cfg: SimConfig | None = None,
    warmup: int = 2,
) -> list[dict[str, float]]:
    """Measure raw stepping throughput for several particle counts."""
    base_cfg = base_cfg or SimConfig()
    results: list[dict[str, float]] = []

    print(f"{'N':>10}  {'steps/s':>10}  {'Mpair/s':>10}")
    print("-" * 36)
    for n in sizes:
        cfg = replace(base_cfg, n=n)
        sim = Simulation(cfg)
        for _ in range(warmup):
            sim.step()
        ti.sync()

        start = time.perf_counter()
        for _ in range(steps):
            sim.step()
        ti.sync()
        elapsed = time.perf_counter() - start

        rate = steps / elapsed if elapsed > 0 else float("inf")
        pair_rate = n * n * rate / 1e6
        results.append(
            {
                "n": float(n),
                "steps_per_second": rate,
                "mpair_per_second": pair_rate,
                "wall_time": elapsed,
            }
        )
        print(f"{n:>10,}  {rate:>10.2f}  {pair_rate:>10.1f}")

    return results
