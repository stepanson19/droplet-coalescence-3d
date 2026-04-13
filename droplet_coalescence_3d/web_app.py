from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coalescence_core_3d import (
    SimulationParams,
    SweepVariable,
    frame_label,
    make_3d_mesh,
    model_description_text,
    simulate,
    sweep_parameter,
)

_ = pd.Series


PARAMETER_LABELS = {
    "radius_mm": "Радиус R, мм",
    "mu_mpas": "Вязкость μ, мПа·с",
    "sigma_mnm": "Поверхностное натяжение σ, мН/м",
    "rho": "Плотность ρ, кг/м³",
}

MAINTENANCE_MODE = False


def plotly_chart_streamlit_kwargs() -> dict[str, bool]:
    """Совместимые kwargs для st.plotly_chart в текущей версии Streamlit."""
    return {"use_container_width": True}


def _surface_trace_from_mesh(mesh) -> go.Surface:
    return go.Surface(
        x=(mesh.x * 1e3).tolist(),
        y=(mesh.y * 1e3).tolist(),
        z=(mesh.z * 1e3).tolist(),
        surfacecolor=(mesh.x * 1e3).tolist(),
        colorscale=[
            [0.0, "#90e0ef"],
            [0.45, "#48cae4"],
            [0.7, "#00b4d8"],
            [1.0, "#0077b6"],
        ],
        showscale=False,
        contours={
            "x": {"show": True, "color": "rgba(2, 48, 71, 0.28)", "width": 2},
            "y": {"show": True, "color": "rgba(2, 48, 71, 0.28)", "width": 2},
            "z": {"show": False},
        },
        lighting={"ambient": 0.68, "diffuse": 0.72, "roughness": 0.95, "specular": 0.05},
    )


def stage_display_frame_budgets(
    frame_duration_ms: int,
    coalescence_display_s: float,
    post_merge_display_s: float,
) -> tuple[int, int]:
    frame_duration_ms = int(np.clip(frame_duration_ms, 80, 3000))
    coalescence_display_s = float(np.clip(coalescence_display_s, 0.4, 12.0))
    post_merge_display_s = float(np.clip(post_merge_display_s, 0.8, 20.0))

    early_frames = max(6, int(round(coalescence_display_s * 1000.0 / frame_duration_ms)))
    post_merge_frames = max(10, int(round(post_merge_display_s * 1000.0 / frame_duration_ms)))
    return early_frames, post_merge_frames


def _animation_frame_indices(
    result,
    frame_count: int,
    frame_duration_ms: int = 480,
    coalescence_display_s: float = 2.0,
    post_merge_display_s: float = 6.0,
) -> np.ndarray:
    frame_count = int(np.clip(frame_count, 12, min(120, len(result.t))))
    total_time = float(result.t[-1])

    if total_time <= 0.0 or not np.isfinite(total_time):
        indices = np.linspace(0, len(result.t) - 1, frame_count).astype(int)
        return np.unique(indices)

    focus_start = result.merge_time + result.transition_time
    if not np.isfinite(focus_start) or focus_start <= 0.0:
        focus_start = 0.10 * total_time
    focus_start = float(np.clip(focus_start, total_time / 160.0, 0.18 * total_time))

    post_merge_window = 0.30 * total_time
    if np.isfinite(result.oscillation_period) and result.oscillation_period > 0.0:
        post_merge_window = max(post_merge_window, 5.0 * float(result.oscillation_period))
    if np.isfinite(result.damping_time) and result.damping_time > 0.0:
        post_merge_window = min(max(post_merge_window, 0.35 * float(result.damping_time)), 0.65 * total_time)
    focus_end = float(np.clip(focus_start + post_merge_window, focus_start + 0.12 * total_time, total_time))

    early_budget, middle_budget = stage_display_frame_budgets(
        frame_duration_ms,
        coalescence_display_s,
        post_merge_display_s,
    )
    effective_frame_count = max(frame_count, early_budget + middle_budget + 4)
    effective_frame_count = int(np.clip(effective_frame_count, 12, min(180, len(result.t))))

    early_frames = min(max(early_budget, 6), effective_frame_count - 12)
    middle_frames = min(max(middle_budget, 10), effective_frame_count - early_frames - 4)
    late_frames = max(effective_frame_count - early_frames - middle_frames + 2, 4)

    early_times = np.linspace(0.0, focus_start, early_frames, endpoint=True)
    middle_times = np.linspace(focus_start, focus_end, middle_frames, endpoint=True)
    late_times = np.linspace(focus_end, total_time, late_frames, endpoint=True)
    sample_times = np.unique(np.concatenate([early_times, middle_times, late_times]))

    right_indices = np.searchsorted(result.t, sample_times, side="left")
    right_indices = np.clip(right_indices, 0, len(result.t) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(result.t) - 1)
    choose_left = np.abs(result.t[left_indices] - sample_times) <= np.abs(result.t[right_indices] - sample_times)
    indices = np.where(choose_left, left_indices, right_indices)
    indices = np.unique(indices)

    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)
    if indices[-1] != len(result.t) - 1:
        indices = np.append(indices, len(result.t) - 1)
    return indices


def animation_scene_ranges_mm(result) -> tuple[list[float], list[float], list[float]]:
    radial_extent_mm = 1.25 * float(
        max(result.params.radius_m, result.equivalent_radius * (1.0 + max(0.0, result.params.initial_mode_amplitude))) * 1e3
    )
    axial_extent_mm = 1.25 * float(max(2.1 * result.params.radius_m, 1.35 * result.equivalent_radius) * 1e3)
    x_range = [-axial_extent_mm, axial_extent_mm]
    y_range = [-radial_extent_mm, radial_extent_mm]
    z_range = [-radial_extent_mm, radial_extent_mm]
    return x_range, y_range, z_range


def build_simulation_animation_figure(
    result,
    frame_count: int = 72,
    frame_duration_ms: int = 480,
    coalescence_display_s: float = 2.0,
    post_merge_display_s: float = 6.0,
) -> go.Figure:
    indices = _animation_frame_indices(
        result,
        frame_count,
        frame_duration_ms=frame_duration_ms,
        coalescence_display_s=coalescence_display_s,
        post_merge_display_s=post_merge_display_s,
    )
    initial_idx = int(indices[0])
    initial_mesh = make_3d_mesh(result, initial_idx, n_axial=96, n_azimuth=30)
    x_range, y_range, z_range = animation_scene_ranges_mm(result)
    frame_duration_ms = int(np.clip(frame_duration_ms, 80, 3000))

    t_ms = result.t * 1e3

    fig = go.Figure(data=[_surface_trace_from_mesh(initial_mesh)])

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, object]] = []
    for order, idx in enumerate(indices):
        mesh = make_3d_mesh(result, int(idx), n_axial=96, n_azimuth=30)
        frame_name = f"frame_{order}"
        frames.append(
            go.Frame(
                name=frame_name,
                traces=[0],
                data=[_surface_trace_from_mesh(mesh)],
                layout=go.Layout(title={"text": frame_label(result, int(idx))}),
            )
        )
        slider_steps.append(
            {
                "label": f"{t_ms[idx]:.1f}",
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.frames = frames
    fig.update_layout(
        title=frame_label(result, initial_idx),
        height=760,
        margin={"l": 10, "r": 10, "t": 80, "b": 20},
        scene={
            "xaxis_title": "x, мм",
            "yaxis_title": "y, мм",
            "zaxis_title": "z, мм",
            "xaxis": {"range": x_range, "autorange": False},
            "yaxis": {"range": y_range, "autorange": False},
            "zaxis": {"range": z_range, "autorange": False},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.65, "y": 1.0, "z": 1.0},
            "camera": {"eye": {"x": 1.7, "y": 1.45, "z": 0.9}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.04,
                "y": 1.12,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Старт",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Пауза",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": -0.06,
                "len": 0.88,
                "currentvalue": {"prefix": "t, мс: "},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def build_surface_figure(result, idx: int) -> go.Figure:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    mesh = make_3d_mesh(result, idx, n_axial=180, n_azimuth=72)
    fig = go.Figure(
        data=[_surface_trace_from_mesh(mesh)]
    )
    fig.update_layout(
        title=mesh.title,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        scene={
            "xaxis_title": "x, мм",
            "yaxis_title": "y, мм",
            "zaxis_title": "z, мм",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.7, "y": 1.45, "z": 0.9}},
        },
    )
    return fig


def build_timeseries_figure(result, idx: int) -> go.Figure:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    t_ms = result.t * 1e3
    t_cur_ms = float(result.t[idx] * 1e3)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Рост перешейка", "Релаксация моды l = 2"),
        vertical_spacing=0.16,
    )

    fig.add_trace(
        go.Scatter(x=t_ms, y=result.neck_radius / result.params.radius_m, mode="lines", name="r_n/R"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t_cur_ms],
            y=[result.neck_radius[idx] / result.params.radius_m],
            mode="markers",
            marker={"size": 9},
            name="Текущий кадр",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_ms, y=result.mode_amplitude, mode="lines", name="A(t)"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t_cur_ms],
            y=[result.mode_amplitude[idx]],
            mode="markers",
            marker={"size": 9},
            name="Текущий кадр 2",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if math.isfinite(result.merge_time):
        merge_ms = result.merge_time * 1e3
        fig.add_vline(x=merge_ms, line_dash="dash", line_width=1.2, row=1, col=1)
        fig.add_vline(x=merge_ms, line_dash="dash", line_width=1.2, row=2, col=1)

    fig.add_vline(x=t_cur_ms, line_dash="dot", line_width=1.0, row=1, col=1)
    fig.add_vline(x=t_cur_ms, line_dash="dot", line_width=1.0, row=2, col=1)

    fig.update_yaxes(title_text="r_n/R", row=1, col=1)
    fig.update_yaxes(title_text="A(t)", row=2, col=1)
    fig.update_xaxes(title_text="t, мс", row=2, col=1)
    fig.update_layout(height=560, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return fig


def build_experiment_figure(rows: list[dict[str, float | str]], variable: SweepVariable) -> go.Figure:
    x = np.array([float(row["value"]) for row in rows], dtype=float)
    merge = np.array([float(row["merge_time_ms"]) for row in rows], dtype=float)
    period = np.array([float(row["period_ms"]) for row in rows], dtype=float)
    damping = np.array([float(row["damping_time_ms"]) for row in rows], dtype=float)
    oh = np.array([float(row["oh"]) for row in rows], dtype=float)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Время завершения слияния", "Колебания и затухание"),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(x=x, y=merge, mode="lines+markers", name="t_merge, мс"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=period, mode="lines+markers", name="T₂, мс"),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=damping, mode="lines+markers", name="τ_damp, мс"),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=oh, mode="lines+markers", name="Oh", line={"dash": "dash"}),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.update_xaxes(title_text=PARAMETER_LABELS[variable], row=1, col=1)
    fig.update_xaxes(title_text=PARAMETER_LABELS[variable], row=1, col=2)
    fig.update_yaxes(title_text="t_merge, мс", row=1, col=1)
    fig.update_yaxes(title_text="мс", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Oh", row=1, col=2, secondary_y=True)
    fig.update_layout(height=440, margin={"l": 20, "r": 20, "t": 64, "b": 20})
    return fig


def build_experiment_csv_bytes(rows: list[dict[str, float | str]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _collect_simulation_params(st: Any, key_prefix: str) -> SimulationParams:
    defaults = SimulationParams()
    col1, col2 = st.columns(2)

    with col1:
        radius_mm = st.number_input(
            "Радиус R, мм",
            min_value=0.05,
            value=float(defaults.radius_mm),
            step=0.05,
            format="%.3f",
            key=f"{key_prefix}_radius_mm",
        )
        rho = st.number_input(
            "Плотность ρ, кг/м³",
            min_value=100.0,
            value=float(defaults.rho),
            step=1.0,
            format="%.1f",
            key=f"{key_prefix}_rho",
        )
        mu_mpas = st.number_input(
            "Вязкость μ, мПа·с",
            min_value=0.05,
            value=float(defaults.mu_mpas),
            step=0.05,
            format="%.3f",
            key=f"{key_prefix}_mu",
        )
        sigma_mnm = st.number_input(
            "Поверхностное натяжение σ, мН/м",
            min_value=1.0,
            value=float(defaults.sigma_mnm),
            step=1.0,
            format="%.3f",
            key=f"{key_prefix}_sigma",
        )

    with col2:
        initial_neck_ratio = st.slider(
            "Начальный радиус мостика r0/R",
            min_value=0.005,
            max_value=0.200,
            value=float(defaults.initial_neck_ratio),
            step=0.005,
            key=f"{key_prefix}_neck",
        )
        initial_mode_amplitude = st.slider(
            "Начальная амплитуда моды A0",
            min_value=0.0,
            max_value=0.5,
            value=float(defaults.initial_mode_amplitude),
            step=0.01,
            key=f"{key_prefix}_amp",
        )
        bridge_time_scale = st.slider(
            "Коэффициент замедления k_slow",
            min_value=0.5,
            max_value=8.0,
            value=float(defaults.bridge_time_scale),
            step=0.1,
            key=f"{key_prefix}_slow",
        )
        total_time_ms = st.number_input(
            "Длительность расчета, мс",
            min_value=5.0,
            value=float(defaults.total_time_ms),
            step=5.0,
            format="%.1f",
            key=f"{key_prefix}_time",
        )

    return SimulationParams(
        radius_mm=float(radius_mm),
        rho=float(rho),
        mu_mpas=float(mu_mpas),
        sigma_mnm=float(sigma_mnm),
        initial_neck_ratio=float(initial_neck_ratio),
        initial_mode_amplitude=float(initial_mode_amplitude),
        bridge_time_scale=float(bridge_time_scale),
        total_time_ms=float(total_time_ms),
    )


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Слипание капель в невесомости — 3D",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if MAINTENANCE_MODE:
        st.title("Слипание пары капель воды в невесомости — 3D")
        st.warning("Приложение временно недоступно.")
        st.info("Доступ по ссылке временно отключен. После снятия режима техработ приложение заработает снова.")
        st.stop()

    st.title("Слипание пары капель воды в невесомости — 3D")
    st.caption("Учебная 3D-модель: reduced-order физика, интерактивная визуализация и вычислительный эксперимент.")

    sim_tab, exp_tab, model_tab = st.tabs(["Симуляция 3D", "Вычислительный эксперимент", "Модель"])

    with sim_tab:
        st.subheader("Управление скоростью процесса")
        defaults = SimulationParams()
        speed_factor = st.slider(
            "Скорость слипания",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Больше значение — быстрее происходит слипание.",
        )
        params = SimulationParams(
            radius_mm=defaults.radius_mm,
            rho=defaults.rho,
            mu_mpas=defaults.mu_mpas,
            sigma_mnm=defaults.sigma_mnm,
            initial_neck_ratio=defaults.initial_neck_ratio,
            initial_mode_amplitude=defaults.initial_mode_amplitude,
            total_time_ms=defaults.total_time_ms,
            merge_ratio=defaults.merge_ratio,
            bridge_time_scale=max(0.15, defaults.bridge_time_scale / speed_factor),
            inertial_const=defaults.inertial_const,
            ilv_const=defaults.ilv_const,
            animation_frames=defaults.animation_frames,
        )
        result = simulate(params)
        st.plotly_chart(
            build_simulation_animation_figure(
                result,
                frame_count=72,
                frame_duration_ms=480,
                coalescence_display_s=2.0,
                post_merge_display_s=6.0,
            ),
            **plotly_chart_streamlit_kwargs(),
        )

    with exp_tab:
        st.subheader("Параметры вычислительного эксперимента")
        base_params = _collect_simulation_params(st, "exp")

        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        with control_col1:
            variable = st.selectbox(
                "Изменяемый параметр",
                options=["radius_mm", "mu_mpas", "sigma_mnm", "rho"],
                format_func=lambda item: PARAMETER_LABELS[item],
            )
        with control_col2:
            start = st.number_input("Начало диапазона", min_value=0.01, value=0.6, step=0.1, format="%.3f")
        with control_col3:
            stop = st.number_input("Конец диапазона", min_value=0.02, value=1.6, step=0.1, format="%.3f")
        with control_col4:
            points = st.slider("Число точек", min_value=3, max_value=15, value=6)

        rows = sweep_parameter(base_params, variable, float(start), float(stop), int(points))
        st.dataframe(rows, width="stretch")
        st.download_button(
            "Скачать CSV",
            data=build_experiment_csv_bytes(rows),
            file_name=f"experiment_{variable}.csv",
            mime="text/csv",
        )

    with model_tab:
        st.subheader("Математическая модель")
        st.write(model_description_text())


if __name__ == "__main__":
    main()
