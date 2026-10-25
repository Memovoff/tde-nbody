"""Interactive Taichi UI for the tidal disruption event.

Controls:
    SPACE      pause / resume
    R / ENTER  reset
    arrows     rotate camera
    mouse drag rotate camera
    F          toggle follow camera
    Z / X      zoom in / out
    + / -      simulation speed (steps per frame)
    ESC        quit
"""

from __future__ import annotations

import math
import time

import taichi as ti

from .config import SimConfig
from .core import Simulation
from .render import FollowCamera, Renderer


def run_interactive(
    cfg: SimConfig | None = None,
    width: int = 1280,
    height: int = 720,
    exposure: float = 1.05,
    bloom_strength: float = 0.85,
    zoom: float = 1.0,
    auto_rotate: bool = True,
) -> None:
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
    )

    gui = ti.GUI(
        "Tidal Disruption - GPU N-body",
        res=(width, height),
        fast_gui=True,
    )

    azimuth = math.radians(-38.0)
    elevation = math.radians(28.0)
    paused = False
    speed = 2
    previous_cursor: tuple[float, float] | None = None

    def reset_view() -> None:
        camera._initialized = False
        camera.zoom = zoom
        camera.update(sim, azimuth, elevation)
        sim.update_colors()

    camera.update(sim, azimuth, elevation)
    sim.update_colors()

    frames = 0
    last_report = time.perf_counter()

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            key = event.key
            if key == ti.GUI.ESCAPE:
                gui.running = False
            elif key == ti.GUI.SPACE:
                paused = not paused
            elif key == ti.GUI.RETURN or key in ("r", "R"):
                sim.reset()
                reset_view()
            elif key == ti.GUI.LEFT:
                azimuth -= 0.06
            elif key == ti.GUI.RIGHT:
                azimuth += 0.06
            elif key == ti.GUI.UP:
                elevation = min(elevation + 0.04, 1.50)
            elif key == ti.GUI.DOWN:
                elevation = max(elevation - 0.04, 0.05)
            elif key in ("f", "F"):
                camera.follow = not camera.follow
                camera._initialized = False
            elif key in ("z", "Z"):
                camera.zoom *= 1.12
            elif key in ("x", "X"):
                camera.zoom /= 1.12
            elif key in ("+", "="):
                speed = min(speed + 1, 12)
            elif key in ("-", "_"):
                speed = max(speed - 1, 1)

        if gui.is_pressed(ti.GUI.LMB):
            cursor = gui.get_cursor_pos()
            if previous_cursor is not None:
                dx = cursor[0] - previous_cursor[0]
                dy = cursor[1] - previous_cursor[1]
                azimuth += dx * 6.0
                elevation = min(max(elevation + dy * 4.0, 0.05), 1.50)
            previous_cursor = cursor
        else:
            previous_cursor = None
            if auto_rotate and not paused:
                azimuth += 0.0015

        if not paused:
            for _ in range(speed):
                sim.step()

        sim.update_colors()
        camera.update(sim, azimuth, elevation)
        renderer.render(sim, azimuth, elevation, camera)
        gui.set_image(renderer.pixels)
        gui.show()

        frames += 1
        now = time.perf_counter()
        if now - last_report >= 1.0:
            fps = frames / (now - last_report)
            r = float((sim.star_com() ** 2).sum() ** 0.5)
            print(
                f"t={sim.time:7.2f}  steps={sim.steps:6d}  "
                f"r={r:6.2f}  fps={fps:5.1f}  speed={speed}x",
                end="\r",
                flush=True,
            )
            frames = 0
            last_report = now

    print()
