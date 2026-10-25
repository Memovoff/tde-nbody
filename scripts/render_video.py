"""Render an MP4 of the tidal disruption without writing PNG frames to disk."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import taichi as ti

from tde.config import SimConfig
from tde.core import Simulation
from tde.render import FollowCamera, Renderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--steps", type=int, default=4_800)
    parser.add_argument("--frame-every", type=int, default=14)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--exposure", type=float, default=1.05)
    parser.add_argument("--bloom", type=float, default=0.85)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bh-mass", type=float, default=1_000.0)
    parser.add_argument("--penetration", type=float, default=1.4)
    parser.add_argument("--azimuth", type=float, default=-38.0, help="degrees")
    parser.add_argument("--elevation", type=float, default=28.0, help="degrees")
    parser.add_argument("--out", type=str, default="media/tde.mp4")
    parser.add_argument(
        "--arch",
        choices=["gpu", "cuda", "vulkan", "cpu"],
        default="gpu",
    )
    args = parser.parse_args()

    try:
        import imageio
    except ImportError as exc:
        raise SystemExit(
            "imageio is required: pip install -e '.[video]'"
        ) from exc

    arch = {
        "gpu": ti.gpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "cpu": ti.cpu,
    }[args.arch]
    ti.init(arch=arch, default_fp=ti.f32, kernel_profiler=False)

    cfg = SimConfig(
        n=args.n,
        dt=args.dt,
        seed=args.seed,
        bh_mass=args.bh_mass,
        penetration=args.penetration,
    )
    sim = Simulation(cfg)
    renderer = Renderer(
        args.width,
        args.height,
        exposure=args.exposure,
        bloom_strength=args.bloom,
    )
    camera = FollowCamera(
        args.width,
        args.height,
        star_radius=cfg.star_radius,
        start_radius=cfg.start_radius,
        zoom=args.zoom,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    azimuth = math.radians(args.azimuth)
    elevation = math.radians(args.elevation)
    camera.update(sim, azimuth, elevation)
    sim.update_colors()

    start = time.perf_counter()
    total_frames = args.steps // args.frame_every + 1
    written = 0
    camera_every = max(1, args.frame_every // 4)

    try:
        for step in range(args.steps + 1):
            if step % camera_every == 0:
                camera.update(sim, azimuth, elevation)
            if step % args.frame_every == 0:
                sim.update_colors()
                renderer.render(sim, azimuth, elevation, camera)
                frame = (
                    np.clip(renderer.image(), 0.0, 1.0) * 255.0
                ).astype(np.uint8)
                writer.append_data(frame)
                written += 1
                if written % 50 == 0 or written == total_frames:
                    elapsed = time.perf_counter() - start
                    print(
                        f"frame {written:4d}/{total_frames}  "
                        f"t={sim.time:6.2f}  elapsed={elapsed:5.1f}s",
                        end="\r",
                        flush=True,
                    )
            if step < args.steps:
                sim.step()
    finally:
        writer.close()

    elapsed = time.perf_counter() - start
    size_mb = out.stat().st_size / (1024 * 1024)
    print()
    print(
        f"saved {out}  {written} frames  {elapsed:.1f}s  "
        f"{size_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
