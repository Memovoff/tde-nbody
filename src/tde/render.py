"""Additive particle renderer and a follow camera for the TDE."""

import math

import numpy as np
import taichi as ti

_PI = 3.141592653589793
_GAMMA = 1.0 / 2.2


@ti.kernel
def copy_background(pixels: ti.template(), background: ti.template()):
    for i, j in pixels:
        pixels[i, j] = background[i, j]


@ti.kernel
def splat(
    pos: ti.template(),
    color: ti.template(),
    mass: ti.template(),
    pixels: ti.template(),
    width: ti.i32,
    height: ti.i32,
    scale: ti.f32,
    center_x: ti.f32,
    center_y: ti.f32,
    azimuth: ti.f32,
    elevation: ti.f32,
):
    """Project particles and additively splat 5x5 Gaussian sprites."""
    ca = ti.cos(azimuth)
    sa = ti.sin(azimuth)
    tilt = _PI * 0.5 - elevation
    ct = ti.cos(tilt)
    st = ti.sin(tilt)

    for i in range(pos.shape[0]):
        if mass[i] > 0.0:
            p = pos[i]
            x1 = ca * p.x + sa * p.y
            y1 = -sa * p.x + ca * p.y
            screen_y = ct * y1 - st * p.z

            px = width * 0.5 + (x1 - center_x) * scale
            py = height * 0.5 + (screen_y - center_y) * scale
            ix = ti.cast(px, ti.i32)
            iy = ti.cast(py, ti.i32)
            c = color[i] * 0.72

            for dy in ti.static(range(-2, 3)):
                for dx in ti.static(range(-2, 3)):
                    w = ti.exp(-0.5 * (dx * dx + dy * dy)) / 7.2
                    jx = ix + dx
                    jy = iy + dy
                    if 0 <= jx and jx < width and 0 <= jy and jy < height:
                        ti.atomic_add(pixels[jx, jy], c * w)


@ti.kernel
def draw_black_hole(
    pixels: ti.template(),
    width: ti.i32,
    height: ti.i32,
    cx: ti.f32,
    cy: ti.f32,
    radius: ti.f32,
    glow_radius: ti.f32,
    strength: ti.f32,
):
    """Draw a dark core with a warm photon-ring-like glow."""
    for i, j in pixels:
        dx = ti.cast(i, ti.f32) - cx
        dy = ti.cast(j, ti.f32) - cy
        r = ti.sqrt(dx * dx + dy * dy)
        if r < radius:
            pixels[i, j] *= 0.015
        else:
            halo = ti.exp(-((r - radius) / glow_radius) ** 2) * strength
            ring = (
                ti.exp(-((r - radius * 1.35) / (0.75 * radius + 1.0)) ** 2)
                * strength
                * 2.2
            )
            pixels[i, j] += ti.Vector([1.0, 0.52, 0.16]) * (halo + ring)


@ti.kernel
def extract_bright(src: ti.template(), dst: ti.template(), threshold: ti.f32):
    for i, j in src:
        c = src[i, j] - threshold
        dst[i, j] = ti.Vector(
            [ti.max(c.x, 0.0), ti.max(c.y, 0.0), ti.max(c.z, 0.0)]
        )


@ti.kernel
def blur_horizontal(
    src: ti.template(), dst: ti.template(), width: ti.i32, height: ti.i32
):
    for i, j in src:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in ti.static(range(-5, 6)):
            x = i + k
            if 0 <= x and x < width:
                w = ti.exp(-0.5 * (k / 2.2) ** 2)
                acc += src[x, j] * w
                wsum += w
        dst[i, j] = acc / wsum


@ti.kernel
def blur_vertical(
    src: ti.template(), dst: ti.template(), width: ti.i32, height: ti.i32
):
    for i, j in src:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in ti.static(range(-5, 6)):
            y = j + k
            if 0 <= y and y < height:
                w = ti.exp(-0.5 * (k / 2.2) ** 2)
                acc += src[i, y] * w
                wsum += w
        dst[i, j] = acc / wsum


@ti.kernel
def composite(pixels: ti.template(), bloom: ti.template(), strength: ti.f32):
    for i, j in pixels:
        pixels[i, j] += bloom[i, j] * strength


@ti.kernel
def tonemap(pixels: ti.template(), exposure: ti.f32):
    for i, j in pixels:
        c = pixels[i, j] * exposure
        c = ti.Vector(
            [
                1.0 - ti.exp(-c.x),
                1.0 - ti.exp(-c.y),
                1.0 - ti.exp(-c.z),
            ]
        )
        pixels[i, j] = ti.Vector(
            [
                ti.pow(c.x, _GAMMA),
                ti.pow(c.y, _GAMMA),
                ti.pow(c.z, _GAMMA),
            ]
        )


def view_coords(
    point: np.ndarray, azimuth: float, elevation: float
) -> tuple[float, float]:
    """Project a 3D point to the 2D camera plane (no translation)."""
    ca = math.cos(azimuth)
    sa = math.sin(azimuth)
    tilt = _PI * 0.5 - elevation
    ct = math.cos(tilt)
    st = math.sin(tilt)
    x1 = ca * point[0] + sa * point[1]
    y1 = -sa * point[0] + ca * point[1]
    return x1, ct * y1 - st * point[2]


class Camera:
    """A fixed orthographic camera in the rotated view plane."""

    def __init__(self, center=(0.0, 0.0), scale: float = 30.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = float(scale)

    def update(self, sim, azimuth: float, elevation: float) -> None:
        del sim, azimuth, elevation


class FollowCamera(Camera):
    """Follow the stellar COM and reframe the debris automatically."""

    def __init__(
        self,
        width: int,
        height: int,
        star_radius: float = 1.0,
        start_radius: float = 40.0,
        zoom: float = 1.0,
        follow: bool = True,
        smoothing: float = 0.18,
    ):
        super().__init__((0.0, 0.0), 30.0)
        self.width = width
        self.height = height
        self.star_radius = star_radius
        self.start_radius = start_radius
        self.zoom = zoom
        self.follow = follow
        self.smoothing = smoothing
        self._initialized = False

    def update(self, sim, azimuth: float, elevation: float) -> None:
        com = sim.star_com()
        r = float(np.linalg.norm(com))
        debris = sim.debris_radius(90.0)

        # Keep a close-up of the star on approach, but widen enough to show
        # the black hole near periapsis and the growing tidal stream later.
        half_width = max(
            5.0 * self.star_radius,
            1.5 * min(r, 5.0 * self.star_radius),
            1.2 * debris,
        )
        half_width = min(
            max(half_width, 0.25 * self.star_radius),
            80.0 * self.star_radius,
        )

        target_scale = (
            0.5 * min(self.width, self.height) / half_width * self.zoom
        )
        if self.follow:
            x1, y1 = view_coords(com, azimuth, elevation)
            target_center = np.array([x1, y1], dtype=np.float64)
        else:
            target_center = np.zeros(2, dtype=np.float64)
            target_scale = (
                0.5
                * min(self.width, self.height)
                / max(self.start_radius, 1.5 * debris)
                * self.zoom
            )

        if not self._initialized:
            self.center = target_center
            self.scale = float(target_scale)
            self._initialized = True
        else:
            a = self.smoothing
            self.center += a * (target_center - self.center)
            self.scale += a * (target_scale - self.scale)


class Renderer:
    """HDR framebuffer, particle splatting, bloom and tone mapping."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        exposure: float = 1.05,
        bloom_strength: float = 0.85,
    ):
        self.width = width
        self.height = height
        self.exposure = exposure
        self.bloom_strength = bloom_strength
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.background = ti.Vector.field(
            3, dtype=ti.f32, shape=(width, height)
        )
        self._bloom_a = ti.Vector.field(
            3, dtype=ti.f32, shape=(width, height)
        )
        self._bloom_b = ti.Vector.field(
            3, dtype=ti.f32, shape=(width, height)
        )
        self.bh_radius = max(3.0, 3.5 * min(width, height) / 540.0)
        self.bh_glow = 18.0 * min(width, height) / 540.0
        self._build_background()

    def _build_background(self) -> None:
        """A dim starfield with a soft blue vignette."""
        rng = np.random.default_rng(2026)
        width, height = self.width, self.height

        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[:, None]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :]
        vignette = np.clip(1.0 - 0.30 * (x * x + y * y), 0.0, 1.0)
        base = np.array([0.0018, 0.0032, 0.0070], dtype=np.float32)
        image = base[None, None, :] * vignette[:, :, None]

        n_stars = max(300, width * height // 350)
        xs = rng.integers(0, width, n_stars)
        ys = rng.integers(0, height, n_stars)
        brightness = (
            0.012 + 0.09 * rng.random(n_stars).astype(np.float32) ** 2.5
        )
        hue = rng.random(n_stars).astype(np.float32)
        colors = np.empty((n_stars, 3), dtype=np.float32)
        colors[:, 0] = 0.78 + 0.22 * hue
        colors[:, 1] = 0.84 + 0.12 * (1.0 - np.abs(hue - 0.5) * 2.0)
        colors[:, 2] = 1.00 - 0.22 * hue

        kernel = np.array(
            [[0.2, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 0.2]],
            dtype=np.float32,
        )
        kernel /= kernel.sum()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                xx = np.clip(xs + dx, 0, width - 1)
                yy = np.clip(ys + dy, 0, height - 1)
                contribution = colors * (
                    brightness * kernel[dy + 1, dx + 1]
                )[:, None]
                np.add.at(image, (xx, yy), contribution)

        self.background.from_numpy(np.ascontiguousarray(image))

    def render(self, sim, azimuth: float, elevation: float, camera: Camera):
        copy_background(self.pixels, self.background)
        splat(
            sim.pos,
            sim.color,
            sim.mass,
            self.pixels,
            self.width,
            self.height,
            camera.scale,
            float(camera.center[0]),
            float(camera.center[1]),
            azimuth,
            elevation,
        )

        bx = self.width * 0.5 - camera.center[0] * camera.scale
        by = self.height * 0.5 - camera.center[1] * camera.scale
        if -80.0 < bx < self.width + 80.0 and -80.0 < by < self.height + 80.0:
            capture_rate = sim.capture_rate(window=0.8)
            strength = 0.65 + 8.0 * min(capture_rate / max(sim.cfg.star_mass, 1e-9), 1.0)
            draw_black_hole(
                self.pixels,
                self.width,
                self.height,
                bx,
                by,
                self.bh_radius,
                self.bh_glow,
                strength,
            )

        extract_bright(self.pixels, self._bloom_a, 0.16)
        blur_horizontal(self._bloom_a, self._bloom_b, self.width, self.height)
        blur_vertical(self._bloom_b, self._bloom_a, self.width, self.height)
        composite(self.pixels, self._bloom_a, self.bloom_strength)
        tonemap(self.pixels, self.exposure)

    def image(self) -> np.ndarray:
        """Return the frame as an (H, W, 3) top-left NumPy image."""
        raw = self.pixels.to_numpy()
        return np.flipud(np.transpose(raw, (1, 0, 2)))

    def save(self, path: str) -> None:
        from pathlib import Path

        from PIL import Image

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image = np.clip(self.image(), 0.0, 1.0)
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(path)
