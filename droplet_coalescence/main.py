from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from coalescence_core import (
    SimulationParams,
    export_sweep_to_csv,
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


class DropletCoalescenceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Слипание пары капель воды в невесомости")
        self.geometry("1460x920")
        self.minsize(1240, 760)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.result = None
        self.grid_cache = None
        self.frame_indices = None
        self.current_frame = 0
        self.anim_job = None
        self.animating = False
        self.slider_guard = False
        self.sweep_rows = []

        self._build_ui()
        self._load_defaults()
        self.after(150, self.run_simulation)
        self.after(350, self.run_default_experiment)




    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 6))

        self.sim_tab = ttk.Frame(notebook)
        self.exp_tab = ttk.Frame(notebook)
        self.help_tab = ttk.Frame(notebook)

        notebook.add(self.sim_tab, text="Симуляция")
        notebook.add(self.exp_tab, text="Вычислительный эксперимент")
        notebook.add(self.help_tab, text="Модель и справка")

        self._build_simulation_tab()
        self._build_experiment_tab()
        self._build_help_tab()

        self.status_var = tk.StringVar(value="Готово")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 10))

    def _build_simulation_tab(self) -> None:
        self.sim_tab.columnconfigure(1, weight=1)
        self.sim_tab.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.sim_tab, padding=10)
        controls.grid(row=0, column=0, sticky="nsw")

        fig_frame = ttk.Frame(self.sim_tab, padding=(0, 10, 10, 10))
        fig_frame.grid(row=0, column=1, sticky="nsew")
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self.param_vars: dict[str, tk.StringVar] = {}
        self._entry(controls, 0, "Радиус капли R, мм", "radius_mm")
        self._entry(controls, 1, "Плотность ρ, кг/м³", "rho")
        self._entry(controls, 2, "Вязкость μ, мПа·с", "mu_mpas")
        self._entry(controls, 3, "Поверхностное натяжение σ, мН/м", "sigma_mnm")
        self._entry(controls, 4, "Начальный перешеек r₀/R", "initial_neck_ratio")
        self._entry(controls, 5, "Начальная амплитуда A₀", "initial_mode_amplitude")
        self._entry(controls, 6, "Общее время моделирования, мс", "total_time_ms")
        self._entry(controls, 7, "Порог завершения η", "merge_ratio")
        self._entry(controls, 8, "Коэф. замедления k_slow", "bridge_time_scale")
        self._entry(controls, 9, "Число кадров анимации", "animation_frames")

        btn_frame = ttk.Frame(controls)
        btn_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Запустить", command=self.run_simulation).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.pause_btn = ttk.Button(btn_frame, text="Пауза", command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(btn_frame, text="Сбросить к воде", command=self.reset_to_water).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(btn_frame, text="Сохранить PNG", command=self.save_simulation_png).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(btn_frame, text="Сохранить MP4", command=self.save_simulation_mp4).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )

        ttk.Separator(controls, orient="horizontal").grid(row=11, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Label(controls, text="Перемотка по времени").grid(row=12, column=0, columnspan=2, sticky="w")
        self.frame_var = tk.DoubleVar(value=0)
        self.frame_scale = ttk.Scale(
            controls,
            from_=0,
            to=1,
            variable=self.frame_var,
            orient="horizontal",
            command=self.on_scale_move,
        )
        self.frame_scale.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(4, 2))

        self.time_label_var = tk.StringVar(value="t = 0.000 мс")
        ttk.Label(controls, textvariable=self.time_label_var).grid(row=14, column=0, columnspan=2, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(row=15, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(controls, text="Численные характеристики").grid(row=16, column=0, columnspan=2, sticky="w")

        self.summary_text = tk.Text(controls, width=38, height=20, wrap="word")
        self.summary_text.grid(row=17, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        self.summary_text.config(state="disabled")
        controls.rowconfigure(17, weight=1)

        self.sim_fig = Figure(figsize=(10.8, 6.9), dpi=100)
        gs = self.sim_fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[1, 1])
        self.ax_shape = self.sim_fig.add_subplot(gs[:, 0])
        self.ax_neck = self.sim_fig.add_subplot(gs[0, 1])
        self.ax_mode = self.sim_fig.add_subplot(gs[1, 1])
        self.sim_fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.94, wspace=0.30, hspace=0.35)

        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=fig_frame)
        self.sim_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_experiment_tab(self) -> None:
        self.exp_tab.columnconfigure(0, weight=1)
        self.exp_tab.rowconfigure(1, weight=1)
        self.exp_tab.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.exp_tab, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for c in range(10):
            controls.columnconfigure(c, weight=1 if c in (1, 3, 5, 7) else 0)

        ttk.Label(controls, text="Варьировать").grid(row=0, column=0, sticky="w")
        self.exp_var = tk.StringVar(value=PARAMETER_LABELS["radius_mm"])
        var_box = ttk.Combobox(
            controls,
            textvariable=self.exp_var,
            values=list(PARAMETER_LABELS.values()),
            state="readonly",
            width=24,
        )
        var_box.grid(row=0, column=1, sticky="ew", padx=(5, 12))

        self.exp_start = tk.StringVar(value="0.5")
        self.exp_stop = tk.StringVar(value="3.0")
        self.exp_points = tk.StringVar(value="8")

        ttk.Label(controls, text="От").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.exp_start, width=10).grid(row=0, column=3, sticky="ew", padx=(5, 12))
        ttk.Label(controls, text="До").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.exp_stop, width=10).grid(row=0, column=5, sticky="ew", padx=(5, 12))
        ttk.Label(controls, text="Точек").grid(row=0, column=6, sticky="w")
        ttk.Entry(controls, textvariable=self.exp_points, width=8).grid(row=0, column=7, sticky="ew", padx=(5, 12))

        ttk.Button(controls, text="Запустить эксперимент", command=self.run_experiment).grid(row=0, column=8, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Экспорт CSV", command=self.export_experiment_csv).grid(row=0, column=9, sticky="ew")

        self.exp_comment_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.exp_comment_var, wraplength=1200, justify="left").grid(
            row=1, column=0, columnspan=10, sticky="ew", pady=(8, 0)
        )

        fig_frame = ttk.Frame(self.exp_tab, padding=(10, 0, 10, 10))
        fig_frame.grid(row=1, column=0, sticky="nsew")
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self.exp_fig = Figure(figsize=(10.8, 5.4), dpi=100)
        self.exp_ax1 = self.exp_fig.add_subplot(1, 2, 1)
        self.exp_ax2 = self.exp_fig.add_subplot(1, 2, 2)
        self.exp_fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.90, wspace=0.28)
        self.exp_canvas = FigureCanvasTkAgg(self.exp_fig, master=fig_frame)
        self.exp_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.Frame(self.exp_tab, padding=(10, 0, 10, 10))
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("value", "oh", "merge", "period", "damp")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headers = {
            "value": "Параметр",
            "oh": "Oh",
            "merge": "t_merge, мс",
            "period": "T₂, мс",
            "damp": "τ_damp, мс",
        }
        widths = {"value": 140, "oh": 90, "merge": 110, "period": 110, "damp": 120}
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

    def _build_help_tab(self) -> None:
        text = tk.Text(self.help_tab, wrap="word")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", model_description_text())
        text.insert(
            "end",
            "\n\nИНСТРУКЦИЯ ПО РАБОТЕ\n====================\n\n"
            "1. На вкладке «Симуляция» задайте свойства жидкости и размер капли.\n"
            "   Параметр k_slow управляет скоростью слипания: больше k_slow => медленнее стадия 1.\n"
            "2. Нажмите «Запустить». Слева будет анимация формы, справа — графики r_n/R и A(t).\n"
            "3. Ползунком «Перемотка по времени» можно вручную смотреть состояние в любой момент.\n"
            "   Кнопки «Сохранить PNG» и «Сохранить MP4» экспортируют визуализацию.\n"
            "4. На вкладке «Вычислительный эксперимент» выберите параметр для sweep, диапазон и число точек.\n"
            "5. Нажмите «Запустить эксперимент» — программа построит графики и таблицу результатов.\n\n"
            "ЧТО МОЖНО НАПИСАТЬ В ОТЧЕТ\n==========================\n\n"
            "- Привести математическую модель (рост шейки + релаксация моды l=2).\n"
            "- Показать интерфейс программы.\n"
            "- Провести sweep по R или μ и интерпретировать графики.\n"
            "- Сравнить капиллярное время τ_c, вязкостное время τ_v и число Онезорге Oh.\n"
            "- Отдельно отметить, что при невесомости Bo = 0, поэтому ведущим механизмом является поверхностное натяжение.\n"
        )
        text.config(state="disabled")

    def _entry(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
        parent.columnconfigure(1, weight=1)
        self.param_vars[key] = var




    def _load_defaults(self) -> None:
        defaults = SimulationParams()
        self._fill_param_fields(defaults)

    def reset_to_water(self) -> None:
        self._fill_param_fields(SimulationParams())
        self.status_var.set("Параметры воды восстановлены")

    def _fill_param_fields(self, p: SimulationParams) -> None:
        self.param_vars["radius_mm"].set(f"{p.radius_mm}")
        self.param_vars["rho"].set(f"{p.rho}")
        self.param_vars["mu_mpas"].set(f"{p.mu_mpas}")
        self.param_vars["sigma_mnm"].set(f"{p.sigma_mnm}")
        self.param_vars["initial_neck_ratio"].set(f"{p.initial_neck_ratio}")
        self.param_vars["initial_mode_amplitude"].set(f"{p.initial_mode_amplitude}")
        self.param_vars["total_time_ms"].set(f"{p.total_time_ms}")
        self.param_vars["merge_ratio"].set(f"{p.merge_ratio}")
        self.param_vars["bridge_time_scale"].set(f"{p.bridge_time_scale}")
        self.param_vars["animation_frames"].set(f"{p.animation_frames}")

    def collect_params(self) -> SimulationParams:
        try:
            params = SimulationParams(
                radius_mm=float(self.param_vars["radius_mm"].get().replace(",", ".")),
                rho=float(self.param_vars["rho"].get().replace(",", ".")),
                mu_mpas=float(self.param_vars["mu_mpas"].get().replace(",", ".")),
                sigma_mnm=float(self.param_vars["sigma_mnm"].get().replace(",", ".")),
                initial_neck_ratio=float(self.param_vars["initial_neck_ratio"].get().replace(",", ".")),
                initial_mode_amplitude=float(self.param_vars["initial_mode_amplitude"].get().replace(",", ".")),
                total_time_ms=float(self.param_vars["total_time_ms"].get().replace(",", ".")),
                merge_ratio=float(self.param_vars["merge_ratio"].get().replace(",", ".")),
                bridge_time_scale=float(self.param_vars["bridge_time_scale"].get().replace(",", ".")),
                animation_frames=int(float(self.param_vars["animation_frames"].get().replace(",", "."))),
            )
            params.validate()
            return params
        except Exception as exc:
            raise ValueError(str(exc)) from exc




    def run_simulation(self) -> None:
        self.stop_animation()
        try:
            params = self.collect_params()
            self.result = simulate(params)
        except Exception as exc:
            messagebox.showerror("Ошибка параметров", str(exc), parent=self)
            return

        self.grid_cache = make_spatial_grid(self.result.params.radius_m)
        self.frame_indices = self.result.frame_indices
        self.current_frame = 0
        self.frame_scale.configure(to=max(len(self.frame_indices) - 1, 1))
        self.frame_var.set(0)

        self._update_summary()
        self._prepare_static_plots()
        self.draw_frame(0)
        self.animating = True
        self.pause_btn.configure(text="Пауза")
        self.status_var.set("Симуляция рассчитана")
        self.schedule_next_frame()

    def _prepare_static_plots(self) -> None:
        result = self.result
        assert result is not None

        t_ms = result.t * 1e3
        self.ax_neck.clear()
        self.ax_mode.clear()

        self.ax_neck.plot(t_ms, result.neck_radius / result.params.radius_m, linewidth=2.0)
        if math.isfinite(result.merge_time):
            self.ax_neck.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
        self.ax_neck.set_title("Рост перешейка")
        self.ax_neck.set_xlabel("t, мс")
        self.ax_neck.set_ylabel("rₙ / R")
        self.ax_neck.grid(True, alpha=0.3)

        self.ax_mode.plot(t_ms, result.mode_amplitude, linewidth=2.0)
        if math.isfinite(result.merge_time):
            self.ax_mode.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
        self.ax_mode.set_title("Релаксация моды l = 2")
        self.ax_mode.set_xlabel("t, мс")
        self.ax_mode.set_ylabel("A(t)")
        self.ax_mode.grid(True, alpha=0.3)

        (self.neck_marker,) = self.ax_neck.plot([t_ms[0]], [result.neck_radius[0] / result.params.radius_m], marker="o")
        (self.mode_marker,) = self.ax_mode.plot([t_ms[0]], [result.mode_amplitude[0]], marker="o")
        self.neck_cursor = self.ax_neck.axvline(t_ms[0], linestyle=":", linewidth=1.2)
        self.mode_cursor = self.ax_mode.axvline(t_ms[0], linestyle=":", linewidth=1.2)

        self.sim_canvas.draw_idle()

    def draw_frame(self, frame_number: int) -> None:
        if self.result is None or self.frame_indices is None:
            return

        frame_number = max(0, min(frame_number, len(self.frame_indices) - 1))
        idx = int(self.frame_indices[frame_number])
        result = self.result
        t_ms = result.t[idx] * 1e3
        R = result.params.radius_m
        X, Y, extent = self.grid_cache

        self._draw_shape_axis(self.ax_shape, result, idx, X, Y, extent)

        self.neck_marker.set_data([t_ms], [result.neck_radius[idx] / R])
        self.mode_marker.set_data([t_ms], [result.mode_amplitude[idx]])
        self.neck_cursor.set_xdata([t_ms, t_ms])
        self.mode_cursor.set_xdata([t_ms, t_ms])

        self.time_label_var.set(f"t = {t_ms:.3f} мс")
        self.current_frame = frame_number

        if not self.slider_guard:
            self.slider_guard = True
            self.frame_var.set(frame_number)
            self.slider_guard = False

        self.sim_canvas.draw_idle()

    def schedule_next_frame(self) -> None:
        if not self.animating or self.result is None or self.frame_indices is None:
            return
        delay_ms = 35
        self.anim_job = self.after(delay_ms, self._animate_step)

    def _animate_step(self) -> None:
        if not self.animating or self.frame_indices is None:
            return
        next_frame = self.current_frame + 1
        if next_frame >= len(self.frame_indices):
            self.animating = False
            self.pause_btn.configure(text="Продолжить")
            self.status_var.set("Анимация завершена")
            return
        self.draw_frame(next_frame)
        self.schedule_next_frame()

    def stop_animation(self) -> None:
        if self.anim_job is not None:
            try:
                self.after_cancel(self.anim_job)
            except tk.TclError:
                pass
            self.anim_job = None
        self.animating = False

    def toggle_pause(self) -> None:
        if self.result is None:
            return
        if self.animating:
            self.stop_animation()
            self.pause_btn.configure(text="Продолжить")
            self.status_var.set("Анимация поставлена на паузу")
        else:
            self.animating = True
            self.pause_btn.configure(text="Пауза")
            self.schedule_next_frame()
            self.status_var.set("Анимация продолжена")

    def on_scale_move(self, *_args) -> None:
        if self.slider_guard or self.frame_indices is None:
            return
        frame_number = int(round(self.frame_var.get()))
        self.stop_animation()
        self.pause_btn.configure(text="Продолжить")
        self.draw_frame(frame_number)
        self.status_var.set("Показано состояние для выбранного момента времени")

    def _update_summary(self) -> None:
        if self.result is None:
            return
        text = "\n".join(result_summary_lines(self.result))
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state="disabled")

    def save_simulation_png(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile="droplet_coalescence_simulation.png",
        )
        if not path:
            return
        self.sim_fig.savefig(path, dpi=170)
        self.status_var.set(f"PNG сохранен: {Path(path).name}")

    def save_simulation_mp4(self) -> None:
        if self.result is None:
            messagebox.showinfo("Сохранение MP4", "Сначала выполните симуляцию.", parent=self)
            return
        try:
            import imageio.v2 as imageio
        except Exception:
            messagebox.showerror(
                "Сохранение MP4",
                "Не хватает зависимостей для MP4.\nУстановите: pip install imageio imageio-ffmpeg",
                parent=self,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
            initialfile="droplet_coalescence_simulation.mp4",
        )
        if not path:
            return

        result = self.result
        frame_indices = result.frame_indices
        X, Y, extent = make_spatial_grid(result.params.radius_m)
        t_ms = result.t * 1e3
        r = result.params.radius_m

        fig = Figure(figsize=(11, 6.5), dpi=120)
        canvas = FigureCanvasAgg(fig)
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1])
        ax_shape = fig.add_subplot(gs[:, 0])
        ax_neck = fig.add_subplot(gs[0, 1])
        ax_mode = fig.add_subplot(gs[1, 1])

        ax_neck.plot(t_ms, result.neck_radius / r, linewidth=2.0)
        if math.isfinite(result.merge_time):
            ax_neck.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
        ax_neck.set_title("Рост перешейка")
        ax_neck.set_xlabel("t, мс")
        ax_neck.set_ylabel("rₙ / R")
        ax_neck.grid(True, alpha=0.3)

        ax_mode.plot(t_ms, result.mode_amplitude, linewidth=2.0)
        if math.isfinite(result.merge_time):
            ax_mode.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
        ax_mode.set_title("Релаксация моды l = 2")
        ax_mode.set_xlabel("t, мс")
        ax_mode.set_ylabel("A(t)")
        ax_mode.grid(True, alpha=0.3)

        (neck_marker,) = ax_neck.plot([t_ms[0]], [result.neck_radius[0] / r], marker="o")
        (mode_marker,) = ax_mode.plot([t_ms[0]], [result.mode_amplitude[0]], marker="o")
        neck_cursor = ax_neck.axvline(t_ms[0], linestyle=":", linewidth=1.2)
        mode_cursor = ax_mode.axvline(t_ms[0], linestyle=":", linewidth=1.2)

        try:
            writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8)
        except Exception as exc:
            messagebox.showerror("Сохранение MP4", f"Не удалось создать MP4: {exc}", parent=self)
            return

        try:
            for idx in frame_indices:
                idx = int(idx)
                self._draw_shape_axis(ax_shape, result, idx, X, Y, extent)
                t_cur_ms = result.t[idx] * 1e3
                neck_marker.set_data([t_cur_ms], [result.neck_radius[idx] / r])
                mode_marker.set_data([t_cur_ms], [result.mode_amplitude[idx]])
                neck_cursor.set_xdata([t_cur_ms, t_cur_ms])
                mode_cursor.set_xdata([t_cur_ms, t_cur_ms])

                fig.tight_layout()
                canvas.draw()
                frame = np.asarray(canvas.buffer_rgba())[:, :, :3]
                writer.append_data(frame)
        finally:
            writer.close()
            fig.clear()

        self.status_var.set(f"MP4 сохранен: {Path(path).name}")

    def _draw_shape_axis(self, axis, result, idx: int, X, Y, extent) -> None:
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




    def current_sweep_variable(self) -> str:
        reverse = {v: k for k, v in PARAMETER_LABELS.items()}
        return reverse[self.exp_var.get()]

    def run_default_experiment(self) -> None:
        self.exp_var.set(PARAMETER_LABELS["radius_mm"])
        self.exp_start.set("0.5")
        self.exp_stop.set("3.0")
        self.exp_points.set("8")
        self.run_experiment()

    def run_experiment(self) -> None:
        try:
            base_params = self.collect_params()
            variable = self.current_sweep_variable()
            start = float(self.exp_start.get().replace(",", "."))
            stop = float(self.exp_stop.get().replace(",", "."))
            points = int(float(self.exp_points.get().replace(",", ".")))
            self.sweep_rows = sweep_parameter(base_params, variable, start, stop, points)
        except Exception as exc:
            messagebox.showerror("Ошибка эксперимента", str(exc), parent=self)
            return

        self._plot_experiment(variable)
        self._fill_tree(variable)
        self.exp_comment_var.set(infer_experiment_comment(variable, self.sweep_rows))
        self.status_var.set("Вычислительный эксперимент выполнен")

    def _plot_experiment(self, variable: str) -> None:
        x = [float(r["value"]) for r in self.sweep_rows]
        merge = [float(r["merge_time_ms"]) for r in self.sweep_rows]
        period = [float(r["period_ms"]) for r in self.sweep_rows]
        damping = [float(r["damping_time_ms"]) for r in self.sweep_rows]
        oh = [float(r["oh"]) for r in self.sweep_rows]

        label = PARAMETER_LABELS[variable]
        self.exp_fig.clear()
        self.exp_ax1 = self.exp_fig.add_subplot(1, 2, 1)
        self.exp_ax2 = self.exp_fig.add_subplot(1, 2, 2)

        self.exp_ax1.plot(x, merge, marker="o", linewidth=2.0)
        self.exp_ax1.set_title("Время завершения слияния")
        self.exp_ax1.set_xlabel(label)
        self.exp_ax1.set_ylabel("t_merge, мс")
        self.exp_ax1.grid(True, alpha=0.3)

        self.exp_ax2.plot(x, period, marker="o", linewidth=2.0, label="T₂, мс")
        self.exp_ax2.plot(x, damping, marker="s", linewidth=2.0, label="τ_damp, мс")
        exp_ax2_twin = self.exp_ax2.twinx()
        exp_ax2_twin.plot(x, oh, marker="^", linewidth=1.8, linestyle="--", label="Oh")
        self.exp_ax2.set_title("Колебания и затухание")
        self.exp_ax2.set_xlabel(label)
        self.exp_ax2.set_ylabel("мс")
        exp_ax2_twin.set_ylabel("Oh")
        self.exp_ax2.grid(True, alpha=0.3)

        lines1, labels1 = self.exp_ax2.get_legend_handles_labels()
        lines2, labels2 = exp_ax2_twin.get_legend_handles_labels()
        self.exp_ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        self.exp_fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.90, wspace=0.28)
        self.exp_canvas.draw_idle()

    def _fill_tree(self, variable: str) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self.sweep_rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    f"{row['value']:.4g}",
                    f"{row['oh']:.5f}",
                    f"{row['merge_time_ms']:.4f}",
                    (f"{row['period_ms']:.4f}" if math.isfinite(float(row['period_ms'])) else "—"),
                    f"{row['damping_time_ms']:.4f}",
                ),
            )
        self.tree.heading("value", text=PARAMETER_LABELS[variable])

    def export_experiment_csv(self) -> None:
        if not self.sweep_rows:
            messagebox.showinfo("Экспорт CSV", "Сначала выполните эксперимент.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="droplet_coalescence_experiment.csv",
        )
        if not path:
            return
        export_sweep_to_csv(self.sweep_rows, path)
        self.status_var.set(f"CSV сохранен: {Path(path).name}")


if __name__ == "__main__":
    app = DropletCoalescenceApp()
    app.mainloop()
