from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from coalescence_core_3d import (  # type: ignore  # noqa: E402
    SimulationParams,
    make_3d_mesh,
    simulate,
    sweep_parameter,
)
from self_test import run_self_test  # type: ignore  # noqa: E402
from web_app import (  # type: ignore  # noqa: E402
    _animation_frame_indices,
    animation_scene_ranges_mm,
    build_simulation_animation_figure,
    plotly_chart_streamlit_kwargs,
    stage_display_frame_budgets,
    _surface_trace_from_mesh,
)


def test_params_validation_rejects_invalid_radius() -> None:
    params = SimulationParams(radius_mm=-1.0)
    with pytest.raises(ValueError):
        params.validate()


def test_simulate_returns_monotone_time_and_finite_merge() -> None:
    result = simulate(SimulationParams())
    assert result.t.ndim == 1
    assert result.neck_radius.shape == result.t.shape
    assert result.mode_amplitude.shape == result.t.shape
    assert np.all(np.diff(result.t) >= 0.0)
    assert np.isfinite(result.merge_time)
    assert result.merge_time > 0.0


def test_make_3d_mesh_returns_consistent_surface() -> None:
    result = simulate(SimulationParams())
    mesh = make_3d_mesh(result, len(result.t) // 2, n_axial=90, n_azimuth=48)
    assert mesh.x.shape == mesh.y.shape == mesh.z.shape
    assert mesh.x.ndim == 2
    assert mesh.x.shape[0] >= 20
    assert mesh.x.shape[1] == 48
    assert np.isfinite(mesh.x).all()
    assert np.isfinite(mesh.y).all()
    assert np.isfinite(mesh.z).all()


def test_sweep_parameter_returns_requested_number_of_rows() -> None:
    rows = sweep_parameter(SimulationParams(), "radius_mm", 0.8, 1.4, 5)
    assert len(rows) == 5
    assert all("merge_time_ms" in row for row in rows)
    assert all("period_ms" in row for row in rows)


def test_build_simulation_animation_figure_contains_frames_and_controls() -> None:
    result = simulate(SimulationParams())
    figure = build_simulation_animation_figure(result, frame_count=12, frame_duration_ms=180)
    assert len(figure.frames) >= 8
    assert len(figure.data) == 5
    assert figure.data[0].type == "surface"
    assert figure.layout.updatemenus
    labels = [button.label for button in figure.layout.updatemenus[0].buttons]
    assert "Старт" in "".join(labels)
    assert "Пауза" in "".join(labels)
    assert figure.layout.sliders
    play_button = figure.layout.updatemenus[0].buttons[0]
    assert play_button.args[1]["frame"]["duration"] == 180
    assert figure.data[2].marker.color is None
    assert figure.data[4].marker.color is None


def test_animation_frame_indices_emphasize_early_coalescence_stage() -> None:
    result = simulate(SimulationParams())
    indices = _animation_frame_indices(result, frame_count=24)
    focus_time = result.merge_time + result.transition_time
    assert indices[0] == 0
    assert indices[-1] == len(result.t) - 1
    assert np.sum(result.t[indices] <= focus_time) >= 6


def test_animation_frame_indices_emphasize_early_post_merge_window() -> None:
    result = simulate(SimulationParams())
    indices = _animation_frame_indices(result, frame_count=30)
    focus_start = result.merge_time + result.transition_time
    focus_end = min(result.t[-1], focus_start + max(5.0 * result.oscillation_period, 0.30 * result.t[-1]))
    assert np.sum((result.t[indices] >= focus_start) & (result.t[indices] <= focus_end)) >= 8


def test_stage_display_frame_budgets_grow_with_requested_screen_time() -> None:
    early_small, post_small = stage_display_frame_budgets(400, 1.0, 2.0)
    early_big, post_big = stage_display_frame_budgets(400, 3.0, 6.0)
    assert early_big > early_small
    assert post_big > post_small


def test_animation_scene_ranges_are_fixed_and_symmetric() -> None:
    result = simulate(SimulationParams())
    x_range, y_range, z_range = animation_scene_ranges_mm(result)
    assert x_range[0] < 0 < x_range[1]
    assert y_range[0] < 0 < y_range[1]
    assert z_range[0] < 0 < z_range[1]
    assert abs(abs(x_range[0]) - abs(x_range[1])) < 1e-9
    assert abs(abs(y_range[0]) - abs(y_range[1])) < 1e-9
    assert y_range == z_range


def test_surface_trace_uses_serializable_sequences() -> None:
    result = simulate(SimulationParams())
    mesh = make_3d_mesh(result, 0, n_axial=40, n_azimuth=16)
    trace = _surface_trace_from_mesh(mesh)
    assert isinstance(trace.x, tuple)
    assert isinstance(trace.y, tuple)
    assert isinstance(trace.z, tuple)
    assert isinstance(trace.surfacecolor, tuple)


def test_surface_trace_has_visible_surface_contours() -> None:
    result = simulate(SimulationParams())
    mesh = make_3d_mesh(result, 0, n_axial=40, n_azimuth=16)
    trace = _surface_trace_from_mesh(mesh)
    assert trace.contours.x.show is True
    assert trace.contours.y.show is True


def test_plotly_chart_kwargs_do_not_use_deprecated_plotly_kwargs_channel() -> None:
    kwargs = plotly_chart_streamlit_kwargs()
    assert kwargs == {"use_container_width": True}
    assert "width" not in kwargs


def test_self_test_generates_artifacts(tmp_path: Path) -> None:
    paths = run_self_test(tmp_path)
    assert len(paths) >= 3
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
