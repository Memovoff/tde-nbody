"""Render a compact GIF preview of the tidal disruption."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import taichi as ti

from tde.config import SimConfig
from tde.core import Simulation
from tde.render import FollowCamera, Renderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12_000)
    parser.add_argument("--steps", type=int, default=4_800)
    parser.add_argument("--frame-every", type=int, default=24)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=405)
    parser.add_argument("--out", type=str, default="media/preview.gif")
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

    cfg = SimConfig(n=args.n, dt=0.0025, seed=7)
    sim = Simulation(cfg)
    renderer = Renderer(args.width, args.height)
    camera = FollowCamera(
        args.width,
        args.height,
        star_radius=cfg.star_radius,
        start_radius=cfg.start_radius,
    )

    azimuth = math.radians(-38.0)
    elevation = math.radians(28.0)
    camera.update(sim, azimuth, elevation)
    sim.update_colors()

    frames: list[np.ndarray] = []
    camera_every = max(1, args.frame_every // 4)
    for step in range(args.steps + 1):
        if step % camera_every == 0:
            camera.update(sim, azimuth, elevation)
        if step % args.frame_every == 0:
            sim.update_colors()
            renderer.render(sim, azimuth, elevation, camera)
            frame = (np.clip(renderer.image(), 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            frames.append(frame)
        if step < args.steps:
            sim.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        out,
        frames,
        fps=args.fps,
        loop=0,
        palettesize=128,
        subrectangles=True,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"saved {out}  {len(frames)} frames  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
