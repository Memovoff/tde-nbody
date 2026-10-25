import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, kernel_profiler=False)
    yield
