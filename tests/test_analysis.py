import numpy as np
import pytest

from tde.analysis import (
    fallback_curve,
    save_fallback_csv,
    save_fallback_plot,
    summary,
)
from tde.config import SimConfig
from tde.core import Simulation


@pytest.fixture(scope="module")
def disrupted_sim():
    cfg = SimConfig(
        n=500,
        dt=0.005,
        softening=0.05,
        seed=3,
        capture_radius=0.0,
    )
    sim = Simulation(cfg)
    for _ in range(1_200):
        sim.step()
    return sim


def test_fallback_curve_is_physical(disrupted_sim):
    result = fallback_curve(disrupted_sim, bins=60)

    assert np.all(result["time"] > 0.0)
    assert np.all(result["rate"] >= 0.0)
    assert np.isfinite(result["slope"])
    assert 0.0 <= result["unbound_fraction"] <= 1.0
    assert 0.0 <= result["bound_fraction"] <= 1.0
    assert result["unbound_fraction"] + result["bound_fraction"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert result["t_min"] > 0.0
    assert result["t_max"] >= result["t_min"]
    assert result["median_period"] > 0.0


def test_summary_matches_state(disrupted_sim):
    stats = summary(disrupted_sim)
    assert stats["time"] == pytest.approx(disrupted_sim.time)
    assert 0.0 <= stats["unbound_fraction"] <= 1.0
    assert 0.0 <= stats["bound_fraction"] <= 1.0
    assert stats["debris_r90"] > 0.0
    assert np.isfinite(stats["total_energy"])


def test_fallback_outputs(disrupted_sim, tmp_path):
    result = fallback_curve(disrupted_sim, bins=40)
    csv_path = tmp_path / "fallback.csv"
    plot_path = tmp_path / "fallback.png"
    save_fallback_csv(result, csv_path)
    save_fallback_plot(result, plot_path, disrupted_sim.cfg)

    text = csv_path.read_text(encoding="utf-8")
    assert "rate_msun_per_hour" in text
    assert len(text.splitlines()) == len(result["time"]) + 1
    assert plot_path.stat().st_size > 5_000
