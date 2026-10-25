"""Command line entry point."""

from __future__ import annotations

import argparse

import taichi as ti

from .app import run_interactive
from .config import SimConfig
from .headless import run_benchmark, run_headless

_ARCHES = {
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
    "cpu": ti.cpu,
}


def _init_taichi(arch_name: str) -> None:
    arch = _ARCHES[arch_name]
    try:
        ti.init(arch=arch, default_fp=ti.f32, kernel_profiler=False)
    except Exception:
        if arch_name != "gpu":
            raise
        print("GPU auto-detect failed, falling back to Vulkan")
        ti.init(arch=ti.vulkan, default_fp=ti.f32, kernel_profiler=False)
    backend = ti.lang.impl.current_cfg().arch
    print(f"Taichi backend: {backend}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=20_000, help="particles in the star")
    parser.add_argument("--dt", type=float, default=0.0025, help="time step")
    parser.add_argument("--softening", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bh-mass", type=float, default=1_000.0)
    parser.add_argument("--star-mass", type=float, default=1.0)
    parser.add_argument("--star-radius", type=float, default=1.0)
    parser.add_argument(
        "--penetration",
        type=float,
        default=1.4,
        help="beta = r_t / r_p; values above 1 disrupt the star",
    )
    parser.add_argument(
        "--start-radius",
        type=float,
        default=4.0,
        help="initial distance in units of the tidal radius",
    )
    parser.add_argument("--bh-softening", type=float, default=0.05)
    parser.add_argument("--capture-radius", type=float, default=0.10)
    parser.add_argument(
        "--arch",
        choices=sorted(_ARCHES),
        default="gpu",
        help="Taichi backend (default: auto GPU)",
    )


def _add_render(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--exposure", type=float, default=1.05)
    parser.add_argument("--bloom", type=float, default=0.85)
    parser.add_argument("--azimuth", type=float, default=-38.0, help="degrees")
    parser.add_argument("--elevation", type=float, default=28.0, help="degrees")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.18,
        help="camera smoothing per frame (0..1)",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="keep the camera centred on the black hole",
    )


def _config_from_args(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        n=args.n,
        dt=args.dt,
        softening=args.softening,
        seed=args.seed,
        bh_mass=args.bh_mass,
        star_mass=args.star_mass,
        star_radius=args.star_radius,
        penetration=args.penetration,
        start_radius_rt=args.start_radius,
        bh_softening=args.bh_softening,
        capture_radius=args.capture_radius,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tde",
        description="GPU N-body simulation of a tidal disruption event",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="open the interactive window")
    _add_common(run)
    _add_render(run)
    run.add_argument(
        "--no-rotate",
        action="store_true",
        help="disable auto camera orbit",
    )

    headless = sub.add_parser("headless", help="run without a window")
    _add_common(headless)
    _add_render(headless)
    headless.add_argument("--steps", type=int, default=5_600)
    headless.add_argument("--out", type=str, default="media/preview.png")
    headless.add_argument("--no-preview", action="store_true")
    headless.add_argument("--frames-dir", type=str, default=None)
    headless.add_argument("--frame-every", type=int, default=16)
    headless.add_argument("--energy-csv", type=str, default=None)
    headless.add_argument(
        "--fallback-csv", type=str, default="media/fallback.csv"
    )
    headless.add_argument(
        "--fallback-plot", type=str, default="media/fallback.png"
    )
    headless.add_argument(
        "--no-fallback",
        action="store_true",
        help="skip the fallback curve and its plot",
    )
    headless.add_argument("--diag-every", type=int, default=200)

    bench = sub.add_parser("bench", help="measure raw GPU throughput")
    bench.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5_000, 20_000, 50_000],
        help="particle counts to benchmark",
    )
    bench.add_argument("--steps", type=int, default=20)
    bench.add_argument("--dt", type=float, default=0.0025)
    bench.add_argument("--softening", type=float, default=0.035)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--bh-mass", type=float, default=1_000.0)
    bench.add_argument("--star-mass", type=float, default=1.0)
    bench.add_argument("--star-radius", type=float, default=1.0)
    bench.add_argument("--penetration", type=float, default=1.4)
    bench.add_argument("--start-radius", type=float, default=4.0)
    bench.add_argument(
        "--arch",
        choices=sorted(_ARCHES),
        default="gpu",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _init_taichi(args.arch)

    if args.command == "run":
        run_interactive(
            _config_from_args(args),
            width=args.width,
            height=args.height,
            exposure=args.exposure,
            bloom_strength=args.bloom,
            zoom=args.zoom,
            auto_rotate=not args.no_rotate,
        )
        return

    if args.command == "headless":
        fallback_csv = None if args.no_fallback else args.fallback_csv
        fallback_plot = None if args.no_fallback else args.fallback_plot
        run_headless(
            _config_from_args(args),
            steps=args.steps,
            width=args.width,
            height=args.height,
            exposure=args.exposure,
            bloom_strength=args.bloom,
            out=None if args.no_preview else args.out,
            frames_dir=args.frames_dir,
            frame_every=args.frame_every,
            energy_csv=args.energy_csv,
            fallback_csv=fallback_csv,
            fallback_plot=fallback_plot,
            diag_every=args.diag_every,
            azimuth_deg=args.azimuth,
            elevation_deg=args.elevation,
            zoom=args.zoom,
            follow=not args.no_follow,
            smoothing=args.smoothing,
        )
        return

    if args.command == "bench":
        base = SimConfig(
            dt=args.dt,
            softening=args.softening,
            seed=args.seed,
            bh_mass=args.bh_mass,
            star_mass=args.star_mass,
            star_radius=args.star_radius,
            penetration=args.penetration,
            start_radius_rt=args.start_radius,
        )
        run_benchmark(args.sizes, steps=args.steps, base_cfg=base)
        return

    parser.error(f"unknown command: {args.command}")
