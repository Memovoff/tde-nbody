import numpy as np

from tde.config import SimConfig
from tde.core import Simulation
from tde.render import Camera, FollowCamera, Renderer, view_coords


def test_view_coords_is_identity_for_zero():
    x, y = view_coords(np.zeros(3), 0.3, 0.5)
    assert abs(x) < 1e-12
    assert abs(y) < 1e-12


def test_camera_follows_the_star():
    cfg = SimConfig(n=300)
    sim = Simulation(cfg)
    camera = FollowCamera(320, 180, star_radius=cfg.star_radius)
    camera.update(sim, 0.0, 0.5)
    assert np.all(np.isfinite(camera.center))
    assert camera.scale > 0.0

    for _ in range(30):
        sim.step()
    camera.update(sim, 0.0, 0.5)
    assert np.all(np.isfinite(camera.center))
    assert camera.scale > 0.0


def test_renderer_produces_an_image():
    cfg = SimConfig(n=500)
    sim = Simulation(cfg)
    sim.update_colors()
    camera = Camera((0.0, 0.0), scale=8.0)
    renderer = Renderer(160, 90)

    renderer.render(sim, -0.6, 0.5, camera)
    image = renderer.image()

    assert image.shape == (90, 160, 3)
    assert np.isfinite(image).all()
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    assert image.max() > 0.01
