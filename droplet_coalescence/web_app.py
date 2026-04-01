from __future__ import annotations

import csv
import io
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from coalescence_core import (
    SimulationParams,
    infer_experiment_comment,
    make_spatial_grid,
    model_description_text,
    post_merge_boundary,
    post_merge_radial_field,
    result_summary_lines,
    simulate,
    soft_union_field,
    sweep_parameter,
)


PARAMETER_LABELS = {
    "radius_mm": "Радиус R, мм",
    "mu_mpas": "Вязкость μ, мПа·с",
    "sigma_mnm": "Поверхностное натяжение σ, мН/м",
    "rho": "Плотность ρ, кг/м³",
}


def frame_index_from_progress(result, progress: float) -> int:
    progress = float(np.clip(progress, 0.0, 1.0))
    idx = int(round(progress * (len(result.t) - 1)))
    return int(np.clip(idx, 0, len(result.t) - 1))


def _draw_shape_axis(axis, result, idx: int, X, Y, extent) -> None:
    t_ms = result.t[idx] * 1e3
    radius = result.params.radius_m
    blend = float(result.blend_weight[idx]) if hasattr(result, "blend_weight") else float(result.stage[idx])

    axis.clear()
    axis.set_aspect("equal")
    axis.set_xlim(extent[0] * 1e3, extent[1] * 1e3)
    axis.set_ylim(extent[2] * 1e3, extent[3] * 1e3)
    axis.set_xlabel("x, мм")
    axis.set_ylabel("y, мм")
    axis.grid(True, alpha=0.15)

    if blend < 1e-3:
        field = soft_union_field(X, Y, radius, result.neck_radius[idx])
        axis.contourf(X * 1e3, Y * 1e3, field, levels=[field.min(), 0.0], colors=["#8ecae6"], alpha=0.92)
        axis.contour(X * 1e3, Y * 1e3, field, levels=[0.0], colors=["#023047"], linewidths=2.0)
        axis.set_title(
            f"Стадия 1: рост мостика\n"
            f"t = {t_ms:.3f} мс, rₙ/R = {result.neck_radius[idx] / radius:.3f}"
        )
    elif blend > 1.0 - 1e-3:
        x, y = post_merge_boundary(result.equivalent_radius, result.mode_amplitude[idx])
        axis.fill(x * 1e3, y * 1e3, color="#8ecae6", alpha=0.92)
        axis.plot(x * 1e3, y * 1e3, color="#023047", linewidth=2.0)
        axis.set_title(
            f"Стадия 2: релаксация объединенной капли\n"
            f"t = {t_ms:.3f} мс, A = {result.mode_amplitude[idx]:.3f}"
        )
    else:
        field_pre = soft_union_field(X, Y, radius, result.neck_radius[idx])
        field_post = post_merge_radial_field(X, Y, result.equivalent_radius, result.mode_amplitude[idx])
        field = (1.0 - blend) * field_pre + blend * field_post
        axis.contourf(
            X * 1e3,
            Y * 1e3,
            field,
            levels=[field.min(), 0.0],
            colors=["#8ecae6"],
            alpha=0.92,
        )
        axis.contour(
            X * 1e3,
            Y * 1e3,
            field,
            levels=[0.0],
            colors=["#023047"],
            linewidths=2.0,
            alpha=1.0,
        )
        axis.set_title(
            f"Переход стадий (сглаживание)\n"
            f"t = {t_ms:.3f} мс, w = {blend:.2f}"
        )


def build_shape_frame_figure(result, idx: int) -> Figure:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    X, Y, extent = make_spatial_grid(result.params.radius_m)

    fig = Figure(figsize=(7.6, 5.4), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    _draw_shape_axis(ax, result, idx, X, Y, extent)
    fig.tight_layout()
    return fig


def build_timeseries_figure(result, idx: int) -> Figure:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    t_ms = result.t * 1e3
    t_cur_ms = float(result.t[idx] * 1e3)

    fig = Figure(figsize=(7.6, 5.4), dpi=120)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(t_ms, result.neck_radius / result.params.radius_m, linewidth=2.0)
    if math.isfinite(result.merge_time):
        ax1.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.2)
    ax1.axvline(t_cur_ms, linestyle=":", linewidth=1.1)
    ax1.plot([t_cur_ms], [result.neck_radius[idx] / result.params.radius_m], marker="o")
    ax1.set_title("Рост перешейка")
    ax1.set_ylabel("rₙ/R")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ms, result.mode_amplitude, linewidth=2.0)
    if math.isfinite(result.merge_time):
        ax2.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.2)
    ax2.axvline(t_cur_ms, linestyle=":", linewidth=1.1)
    ax2.plot([t_cur_ms], [result.mode_amplitude[idx]], marker="o")
    ax2.set_title("Релаксация моды l = 2")
    ax2.set_xlabel("t, мс")
    ax2.set_ylabel("A(t)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def build_experiment_figure(rows: list[dict[str, float | str]], variable: str) -> Figure:
    x = np.array([float(r["value"]) for r in rows], dtype=float)
    merge = np.array([float(r["merge_time_ms"]) for r in rows], dtype=float)
    period = np.array([float(r["period_ms"]) for r in rows], dtype=float)
    damping = np.array([float(r["damping_time_ms"]) for r in rows], dtype=float)
    oh = np.array([float(r["oh"]) for r in rows], dtype=float)

    label = PARAMETER_LABELS[variable]
    fig = Figure(figsize=(10.6, 4.8), dpi=120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(x, merge, marker="o", linewidth=2.0)
    ax1.set_title("Время завершения слияния")
    ax1.set_xlabel(label)
    ax1.set_ylabel("t_merge, мс")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, period, marker="o", linewidth=2.0, label="T₂, мс")
    ax2.plot(x, damping, marker="s", linewidth=2.0, label="τ_damp, мс")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, oh, marker="^", linestyle="--", linewidth=1.8, label="Oh")

    ax2.set_title("Колебания и затухание")
    ax2.set_xlabel(label)
    ax2.set_ylabel("мс")
    ax2_twin.set_ylabel("Oh")
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    return buf.getvalue()


def build_experiment_csv_bytes(rows: list[dict[str, float | str]]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _sample_frame_indices(result, frame_count: int) -> np.ndarray:
    frame_count = int(np.clip(frame_count, 8, 320))
    indices = np.linspace(0, len(result.t) - 1, frame_count).astype(int)
    return np.unique(indices)


def _render_shape_frames(result, indices: np.ndarray) -> list[np.ndarray]:
    X, Y, extent = make_spatial_grid(result.params.radius_m)
    fig = Figure(figsize=(7.6, 5.4), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvasAgg(fig)

    frames: list[np.ndarray] = []
    for idx in indices:
        _draw_shape_axis(ax, result, int(idx), X, Y, extent)
        fig.tight_layout()
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
    fig.clear()
    return frames


def build_simulation_gif_bytes(result, frame_count: int = 96) -> bytes:
    import imageio.v2 as imageio

    indices = _sample_frame_indices(result, frame_count)
    frames = _render_shape_frames(result, indices)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", fps=20)
    return buf.getvalue()


def build_simulation_mp4_bytes(result, frame_count: int = 120) -> bytes:
    import imageio.v2 as imageio

    indices = _sample_frame_indices(result, frame_count)
    frames = _render_shape_frames(result, indices)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    writer = imageio.get_writer(str(tmp_path), fps=24, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return data


def _collect_simulation_params(st: Any, key_prefix: str) -> SimulationParams:
    defaults = SimulationParams()
    c1, c2 = st.columns(2)
    with c1:
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
            min_value=50.0,
            value=float(defaults.rho),
            step=10.0,
            format="%.1f",
            key=f"{key_prefix}_rho",
        )
        mu_mpas = st.number_input(
            "Вязкость μ, мПа·с",
            min_value=0.01,
            value=float(defaults.mu_mpas),
            step=0.1,
            format="%.3f",
            key=f"{key_prefix}_mu_mpas",
        )
        sigma_mnm = st.number_input(
            "Поверхностное натяжение σ, мН/м",
            min_value=1.0,
            value=float(defaults.sigma_mnm),
            step=1.0,
            format="%.3f",
            key=f"{key_prefix}_sigma_mnm",
        )
        initial_neck_ratio = st.number_input(
            "Начальный перешеек r₀/R",
            min_value=0.001,
            max_value=0.9,
            value=float(defaults.initial_neck_ratio),
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_initial_neck_ratio",
        )
    with c2:
        initial_mode_amplitude = st.number_input(
            "Начальная амплитуда A₀",
            min_value=0.0,
            max_value=0.79,
            value=float(defaults.initial_mode_amplitude),
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_initial_mode_amplitude",
        )
        total_time_ms = st.number_input(
            "Общее время, мс",
            min_value=1.0,
            value=float(defaults.total_time_ms),
            step=1.0,
            format="%.2f",
            key=f"{key_prefix}_total_time_ms",
        )
        merge_ratio = st.number_input(
            "Порог слияния η",
            min_value=0.50,
            max_value=1.00,
            value=float(defaults.merge_ratio),
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_merge_ratio",
        )
        bridge_time_scale = st.number_input(
            "Коэф. замедления k_slow",
            min_value=1.0,
            max_value=20.0,
            value=float(defaults.bridge_time_scale),
            step=0.5,
            format="%.2f",
            key=f"{key_prefix}_bridge_time_scale",
        )
        animation_frames = st.number_input(
            "Кадров анимации",
            min_value=20,
            max_value=1000,
            value=int(defaults.animation_frames),
            step=10,
            key=f"{key_prefix}_animation_frames",
        )

    return SimulationParams(
        radius_mm=float(radius_mm),
        rho=float(rho),
        mu_mpas=float(mu_mpas),
        sigma_mnm=float(sigma_mnm),
        initial_neck_ratio=float(initial_neck_ratio),
        initial_mode_amplitude=float(initial_mode_amplitude),
        total_time_ms=float(total_time_ms),
        merge_ratio=float(merge_ratio),
        bridge_time_scale=float(bridge_time_scale),
        animation_frames=int(animation_frames),
    )


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Слипание пары капель воды", layout="wide")
    st.title("Слипание пары капель воды в невесомости")
    st.caption("Web-интерфейс без Tk: параметры, симуляция, вычислительный эксперимент, экспорт PNG/GIF/MP4/CSV.")

    tab_sim, tab_exp, tab_help = st.tabs(["Симуляция", "Вычислительный эксперимент", "Модель и справка"])

    with tab_sim:
        st.subheader("Параметры модели")
        params = _collect_simulation_params(st, key_prefix="sim")

        col_run, col_info = st.columns([1, 2])
        with col_run:
            run_sim = st.button("Запустить симуляцию", width="stretch")
        with col_info:
            st.info("Изменяйте параметры и запускайте пересчет. k_slow > 1 делает слипание медленнее.")

        if run_sim or "sim_result" not in st.session_state:
            try:
                params.validate()
                st.session_state["sim_params"] = asdict(params)
                st.session_state["sim_result"] = simulate(params)
            except Exception as exc:
                st.error(f"Ошибка параметров: {exc}")

        result = st.session_state.get("sim_result")
        if result is not None:
            progress = st.slider("Положение по времени", min_value=0.0, max_value=1.0, value=0.0, step=0.001)
            idx = frame_index_from_progress(result, progress)

            left, right = st.columns([1.3, 1.0])
            with left:
                fig_shape = build_shape_frame_figure(result, idx)
                st.pyplot(fig_shape, width="stretch")
            with right:
                fig_curves = build_timeseries_figure(result, idx)
                st.pyplot(fig_curves, width="stretch")

            st.markdown("**Численные характеристики**")
            st.code("\n".join(result_summary_lines(result)))

            png_bytes = figure_to_png_bytes(fig_shape)
            st.download_button(
                "Скачать PNG текущего кадра",
                data=png_bytes,
                file_name="droplet_simulation_frame.png",
                mime="image/png",
            )

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                if st.button("Сгенерировать GIF"):
                    with st.spinner("Генерация GIF..."):
                        try:
                            st.session_state["sim_gif_bytes"] = build_simulation_gif_bytes(result)
                            st.success("GIF готов.")
                        except Exception as exc:
                            st.error(f"Не удалось сгенерировать GIF: {exc}")
                if st.session_state.get("sim_gif_bytes"):
                    st.download_button(
                        "Скачать GIF симуляции",
                        data=st.session_state["sim_gif_bytes"],
                        file_name="droplet_simulation.gif",
                        mime="image/gif",
                    )
            with exp_col2:
                if st.button("Сгенерировать MP4"):
                    with st.spinner("Генерация MP4..."):
                        try:
                            st.session_state["sim_mp4_bytes"] = build_simulation_mp4_bytes(result)
                            st.success("MP4 готов.")
                        except Exception as exc:
                            st.error(f"Не удалось сгенерировать MP4: {exc}")
                if st.session_state.get("sim_mp4_bytes"):
                    st.download_button(
                        "Скачать MP4 симуляции",
                        data=st.session_state["sim_mp4_bytes"],
                        file_name="droplet_simulation.mp4",
                        mime="video/mp4",
                    )

            fig_shape.clear()
            fig_curves.clear()

    with tab_exp:
        st.subheader("Вычислительный эксперимент (sweep)")
        base_params = _collect_simulation_params(st, key_prefix="exp")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            variable = st.selectbox("Варьируемый параметр", list(PARAMETER_LABELS.keys()), format_func=lambda k: PARAMETER_LABELS[k])
        with col2:
            start = st.number_input("От", min_value=0.001, value=0.5, step=0.1, format="%.4f")
        with col3:
            stop = st.number_input("До", min_value=0.001, value=3.0, step=0.1, format="%.4f")
        with col4:
            points = st.number_input("Точек", min_value=2, max_value=60, value=8, step=1)

        if st.button("Запустить эксперимент", width="stretch"):
            try:
                rows = sweep_parameter(base_params, variable, float(start), float(stop), int(points))
                st.session_state["exp_rows"] = rows
                st.session_state["exp_var"] = variable
            except Exception as exc:
                st.error(f"Ошибка эксперимента: {exc}")

        rows = st.session_state.get("exp_rows", [])
        exp_var = st.session_state.get("exp_var", variable)
        if rows:
            fig_exp = build_experiment_figure(rows, exp_var)
            st.pyplot(fig_exp, width="stretch")
            st.info(infer_experiment_comment(exp_var, rows))
            st.dataframe(rows, width="stretch")

            csv_bytes = build_experiment_csv_bytes(rows)
            st.download_button(
                "Скачать CSV эксперимента",
                data=csv_bytes,
                file_name="droplet_experiment.csv",
                mime="text/csv",
            )
            fig_exp.clear()

    with tab_help:
        st.subheader("Математическая модель")
        st.code(model_description_text())
        st.markdown(
            "- Интерфейс web-формата повторяет ключевые возможности desktop-версии.\n"
            "- Для локального запуска используйте `streamlit run web_app.py`.\n"
            "- Экспорт поддерживает PNG/GIF/MP4 и CSV."
        )


if __name__ == "__main__":
    run_streamlit_app()
