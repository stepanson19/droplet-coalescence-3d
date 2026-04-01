from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


if sys.version_info >= (3, 10):
    model_dataclass = partial(dataclass, slots=True)
else:
    model_dataclass = dataclass


SweepVariable = Literal["radius_mm", "mu_mpas", "sigma_mnm", "rho"]


@model_dataclass
class SimulationParams:
    """Параметры учебной 3D reduced-order модели."""

    radius_mm: float = 1.0
    rho: float = 997.0
    mu_mpas: float = 1.0
    sigma_mnm: float = 72.0
    initial_neck_ratio: float = 0.02
    initial_mode_amplitude: float = 0.22
    total_time_ms: float = 300.0
    merge_ratio: float = 0.95
    bridge_time_scale: float = 3.0
    inertial_const: float = 1.62
    ilv_const: float = 1.0
    animation_frames: int = 180

    def validate(self) -> None:
        positive_values = {
            "radius_mm": self.radius_mm,
            "rho": self.rho,
            "mu_mpas": self.mu_mpas,
            "sigma_mnm": self.sigma_mnm,
            "total_time_ms": self.total_time_ms,
            "merge_ratio": self.merge_ratio,
            "bridge_time_scale": self.bridge_time_scale,
            "animation_frames": self.animation_frames,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"Параметр {name} должен быть положительным.")
        if not (0.0 < self.initial_neck_ratio < 1.0):
            raise ValueError("Начальный радиус перешейка должен удовлетворять 0 < r0/R < 1.")
        if not (0.0 < self.merge_ratio <= 1.0):
            raise ValueError("Порог завершения слияния должен лежать в пределах (0, 1].")
        if not (0.0 <= self.initial_mode_amplitude < 0.8):
            raise ValueError("Начальная амплитуда моды должна лежать в диапазоне [0, 0.8).")
        if int(self.animation_frames) < 10:
            raise ValueError("Число кадров должно быть не меньше 10.")

    @property
    def radius_m(self) -> float:
        return self.radius_mm * 1e-3

    @property
    def mu(self) -> float:
        return self.mu_mpas * 1e-3

    @property
    def sigma(self) -> float:
        return self.sigma_mnm * 1e-3

    @property
    def total_time_s(self) -> float:
        return self.total_time_ms * 1e-3


@model_dataclass
class SimulationResult:
    params: SimulationParams
    t: np.ndarray
    neck_radius: np.ndarray
    mode_amplitude: np.ndarray
    blend_weight: np.ndarray
    stage: np.ndarray
    merge_time: float
    transition_time: float
    capillary_time: float
    viscous_time: float
    ohnesorge: float
    bond: float
    equivalent_radius: float
    omega2: float
    beta2: float
    damped_omega2: float
    oscillation_period: float
    damping_time: float
    oscillations_until_1e: float
    bridge_steps: int
    notes: list[str]

    @property
    def frame_indices(self) -> np.ndarray:
        idx = np.linspace(0, len(self.t) - 1, self.params.animation_frames)
        idx = np.unique(idx.astype(int))
        if idx[-1] != len(self.t) - 1:
            idx = np.append(idx, len(self.t) - 1)
        return idx


@model_dataclass
class BridgeIntegration:
    t: np.ndarray
    r: np.ndarray
    merge_time: float
    steps: int
    merged: bool


@model_dataclass
class SurfaceMesh:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    axial_x: np.ndarray
    radial_rho: np.ndarray
    title: str


def capillary_time(params: SimulationParams) -> float:
    return math.sqrt(params.rho * params.radius_m**3 / params.sigma)


def viscous_time(params: SimulationParams) -> float:
    return params.mu * params.radius_m / params.sigma


def ohnesorge(params: SimulationParams) -> float:
    return params.mu / math.sqrt(params.rho * params.sigma * params.radius_m)


def bond_number_microgravity() -> float:
    return 0.0


def equivalent_radius(params: SimulationParams) -> float:
    return (2.0 ** (1.0 / 3.0)) * params.radius_m


def bridge_velocity(neck_radius: float, params: SimulationParams) -> float:
    r = max(neck_radius, 1e-12)
    v_linear = params.ilv_const * params.sigma / params.mu
    v_inertial = 0.5 * params.inertial_const**2 * math.sqrt(params.sigma * params.radius_m / params.rho) / r
    v_comp = 1.0 / (1.0 / v_linear + 1.0 / v_inertial)
    return v_comp / max(params.bridge_time_scale, 1e-12)


def integrate_bridge(
    params: SimulationParams,
    t_stop: float | None = None,
    max_steps: int = 200_000,
) -> BridgeIntegration:
    params.validate()
    radius = params.radius_m
    r_target = params.merge_ratio * radius
    t_limit = params.total_time_s if t_stop is None else float(t_stop)

    t_values = [0.0]
    r_values = [params.initial_neck_ratio * radius]

    t_cur = 0.0
    r_cur = params.initial_neck_ratio * radius
    merged = r_cur >= r_target

    steps = 0
    while t_cur < t_limit and not merged and steps < max_steps:
        velocity = bridge_velocity(r_cur, params)
        dt_rel = 0.015 * max(r_cur, 1e-12) / max(velocity, 1e-12)
        dt_cap = max(t_limit / 5000.0, 2.5e-7)
        dt = min(dt_rel, dt_cap, 2.5e-4)
        dt = max(dt, 1e-9)

        r_cur = min(r_cur + velocity * dt, r_target)
        t_cur = min(t_cur + dt, t_limit)
        t_values.append(t_cur)
        r_values.append(r_cur)
        merged = r_cur >= r_target * (1.0 - 1e-9)
        steps += 1

    merge_time = t_cur if merged else math.nan
    return BridgeIntegration(
        t=np.asarray(t_values, dtype=float),
        r=np.asarray(r_values, dtype=float),
        merge_time=merge_time,
        steps=steps,
        merged=merged,
    )


def oscillation_characteristics(params: SimulationParams) -> tuple[float, float, float, float, float]:
    r_eq = equivalent_radius(params)
    omega0 = math.sqrt(8.0 * params.sigma / (params.rho * r_eq**3))
    beta = 5.0 * params.mu / (params.rho * r_eq**2)
    if beta < omega0:
        omega_d = math.sqrt(max(omega0**2 - beta**2, 0.0))
        period = 2.0 * math.pi / omega_d
    else:
        omega_d = 0.0
        period = math.nan
    return r_eq, omega0, beta, omega_d, period


def mode_amplitude(time_from_merge: np.ndarray, params: SimulationParams) -> np.ndarray:
    a0 = params.initial_mode_amplitude
    _, omega0, beta, omega_d, _ = oscillation_characteristics(params)

    if beta < omega0 and omega_d > 0.0:
        return a0 * np.exp(-beta * time_from_merge) * (
            np.cos(omega_d * time_from_merge) + (beta / omega_d) * np.sin(omega_d * time_from_merge)
        )

    disc = max(beta**2 - omega0**2, 0.0)
    root = math.sqrt(disc)
    lam1 = -beta + root
    lam2 = -beta - root
    if abs(lam1 - lam2) < 1e-12:
        return a0 * np.exp(lam1 * time_from_merge) * (1.0 - lam1 * time_from_merge)

    c1 = -a0 * lam2 / (lam1 - lam2)
    c2 = a0 * lam1 / (lam1 - lam2)
    return c1 * np.exp(lam1 * time_from_merge) + c2 * np.exp(lam2 * time_from_merge)


def smoothstep01(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.clip(x, 0.0, 1.0)
    return x_arr * x_arr * (3.0 - 2.0 * x_arr)


def transition_time(params: SimulationParams) -> float:
    tau_c = capillary_time(params)
    tau_v = viscous_time(params)
    return max(0.15 * tau_c, 0.60 * tau_v, params.total_time_s / 400.0)


def simulate(params: SimulationParams) -> SimulationResult:
    params.validate()

    t_uniform = np.linspace(0.0, params.total_time_s, max(1500, int(params.total_time_ms * 40)))
    bridge = integrate_bridge(params, t_stop=params.total_time_s)

    notes: list[str] = []
    if not bridge.merged:
        notes.append(
            "За выбранное время моделирования шейка не достигла порога merge_ratio. "
            "Увеличьте общее время расчета."
        )

    neck_uniform = np.interp(t_uniform, bridge.t, bridge.r, left=bridge.r[0], right=bridge.r[-1])
    stage = np.zeros_like(t_uniform, dtype=int)
    amplitude = np.zeros_like(t_uniform)
    blend = np.zeros_like(t_uniform)
    transition = transition_time(params)

    if bridge.merged:
        merge_time = bridge.merge_time
        post_mask = t_uniform >= merge_time
        tau = (t_uniform[post_mask] - merge_time) / max(transition, 1e-12)
        blend_post = smoothstep01(tau)
        blend[post_mask] = blend_post

        neck_uniform[post_mask] = params.radius_m * (
            params.merge_ratio + (1.0 - params.merge_ratio) * blend_post
        )

        amplitude_raw = mode_amplitude(t_uniform[post_mask] - merge_time, params)
        amplitude[post_mask] = blend_post * amplitude_raw
        stage = (blend >= 0.5).astype(int)
    else:
        merge_time = math.nan

    r_eq, omega0, beta, omega_d, period = oscillation_characteristics(params)
    damping_time = 1.0 / beta if beta > 0.0 else math.inf
    oscillations_until_1e = omega_d * damping_time / (2.0 * math.pi) if omega_d > 0.0 else math.nan

    return SimulationResult(
        params=params,
        t=t_uniform,
        neck_radius=neck_uniform,
        mode_amplitude=amplitude,
        blend_weight=blend,
        stage=stage,
        merge_time=merge_time,
        transition_time=transition if bridge.merged else 0.0,
        capillary_time=capillary_time(params),
        viscous_time=viscous_time(params),
        ohnesorge=ohnesorge(params),
        bond=bond_number_microgravity(),
        equivalent_radius=r_eq,
        omega2=omega0,
        beta2=beta,
        damped_omega2=omega_d,
        oscillation_period=period,
        damping_time=damping_time,
        oscillations_until_1e=oscillations_until_1e,
        bridge_steps=bridge.steps,
        notes=notes,
    )


def epsilon_from_neck(neck_radius: float, radius: float) -> float:
    ln2 = math.log(2.0)
    return max((math.sqrt(radius * radius + neck_radius * neck_radius) - radius) / ln2, 1e-12)


def soft_union_field_axial(x_grid: np.ndarray, rho_grid: np.ndarray, radius: float, neck_radius: float) -> np.ndarray:
    eps = epsilon_from_neck(neck_radius, radius)
    d1 = np.hypot(x_grid - radius, rho_grid) - radius
    d2 = np.hypot(x_grid + radius, rho_grid) - radius
    minimum = np.minimum(d1, d2)
    return minimum - eps * np.log(np.exp(-(d1 - minimum) / eps) + np.exp(-(d2 - minimum) / eps))


def post_merge_radial_field_axial(
    x_grid: np.ndarray,
    rho_grid: np.ndarray,
    equivalent_radius_value: float,
    amplitude: float,
) -> np.ndarray:
    theta = np.arctan2(rho_grid, x_grid)
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    radius_boundary = equivalent_radius_value * (1.0 + amplitude * p2)
    radius_boundary = np.clip(radius_boundary, 0.05 * equivalent_radius_value, None)
    return np.hypot(x_grid, rho_grid) - radius_boundary


def make_axial_grid(
    radius: float,
    equivalent_radius_value: float,
    n_axial: int = 220,
    n_radial: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_extent = max(2.35 * radius, 1.85 * equivalent_radius_value)
    rho_extent = max(1.75 * radius, 1.55 * equivalent_radius_value)
    x = np.linspace(-x_extent, x_extent, n_axial)
    rho = np.linspace(0.0, rho_extent, n_radial)
    x_grid, rho_grid = np.meshgrid(x, rho)
    return x, rho, x_grid, rho_grid


def profile_from_field(x: np.ndarray, rho: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.zeros_like(x)

    for idx in range(len(x)):
        column = field[:, idx]
        if column[0] > 0.0:
            boundary[idx] = 0.0
            continue

        positive_idx = np.flatnonzero(column > 0.0)
        if len(positive_idx) == 0:
            boundary[idx] = rho[-1]
            continue

        top = int(positive_idx[0])
        if top == 0:
            boundary[idx] = rho[0]
            continue

        rho0 = rho[top - 1]
        rho1 = rho[top]
        f0 = column[top - 1]
        f1 = column[top]
        if abs(f1 - f0) < 1e-12:
            boundary[idx] = rho0
        else:
            boundary[idx] = rho0 - f0 * (rho1 - rho0) / (f1 - f0)

    active = np.flatnonzero(boundary > 1e-10)
    if len(active) == 0:
        raise ValueError("Не удалось извлечь осесимметричный профиль капли.")

    start = max(int(active[0]) - 1, 0)
    stop = min(int(active[-1]) + 2, len(x))
    x_profile = x[start:stop]
    rho_profile = boundary[start:stop]
    rho_profile[0] = 0.0
    rho_profile[-1] = 0.0
    return x_profile, rho_profile


def frame_label(result: SimulationResult, idx: int) -> str:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    t_ms = result.t[idx] * 1e3
    radius = result.params.radius_m
    blend = float(result.blend_weight[idx])

    if blend < 1e-3:
        return f"Стадия 1: рост мостика, t = {t_ms:.3f} мс, r_n/R = {result.neck_radius[idx] / radius:.3f}"
    if blend > 1.0 - 1e-3:
        return f"Стадия 2: релаксация капли, t = {t_ms:.3f} мс, A = {result.mode_amplitude[idx]:.3f}"
    return f"Переход стадий, t = {t_ms:.3f} мс, w = {blend:.2f}"


def make_meridional_profile(
    result: SimulationResult,
    idx: int,
    n_axial: int = 220,
    n_radial: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    idx = int(np.clip(idx, 0, len(result.t) - 1))
    x, rho, x_grid, rho_grid = make_axial_grid(
        radius=result.params.radius_m,
        equivalent_radius_value=result.equivalent_radius,
        n_axial=n_axial,
        n_radial=n_radial,
    )

    blend = float(result.blend_weight[idx])
    field_pre = soft_union_field_axial(x_grid, rho_grid, result.params.radius_m, result.neck_radius[idx])
    field_post = post_merge_radial_field_axial(
        x_grid,
        rho_grid,
        result.equivalent_radius,
        result.mode_amplitude[idx],
    )

    if blend < 1e-3:
        field = field_pre
    elif blend > 1.0 - 1e-3:
        field = field_post
    else:
        field = (1.0 - blend) * field_pre + blend * field_post

    return profile_from_field(x, rho, field)


def make_3d_mesh(
    result: SimulationResult,
    idx: int,
    n_axial: int = 180,
    n_azimuth: int = 72,
) -> SurfaceMesh:
    axial_x, radial_rho = make_meridional_profile(result, idx, n_axial=n_axial)
    phi = np.linspace(0.0, 2.0 * math.pi, n_azimuth)
    x_surface = np.repeat(axial_x[:, None], len(phi), axis=1)
    y_surface = radial_rho[:, None] * np.cos(phi)[None, :]
    z_surface = radial_rho[:, None] * np.sin(phi)[None, :]
    return SurfaceMesh(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        axial_x=axial_x,
        radial_rho=radial_rho,
        title=frame_label(result, idx),
    )


def result_summary_lines(result: SimulationResult) -> list[str]:
    params = result.params
    lines = [
        f"R = {params.radius_mm:.3f} мм",
        f"ρ = {params.rho:.1f} кг/м³",
        f"μ = {params.mu_mpas:.3f} мПа·с",
        f"σ = {params.sigma_mnm:.3f} мН/м",
        f"k_slow = {params.bridge_time_scale:.3f}",
        f"Oh = {result.ohnesorge:.5f}",
        f"Bo = {result.bond:.1f} (невесомость)",
        f"τ_c = {result.capillary_time * 1e3:.3f} мс",
        f"τ_v = {result.viscous_time * 1e3:.3f} мс",
        (
            f"t_merge ≈ {result.merge_time * 1e3:.3f} мс"
            if math.isfinite(result.merge_time)
            else "t_merge: не достигнуто в выбранном интервале"
        ),
        f"R_eq = {result.equivalent_radius * 1e3:.3f} мм",
        f"ω₂ = {result.omega2:.2f} рад/с",
        f"β₂ = {result.beta2:.2f} 1/с",
        (
            f"T₂ ≈ {result.oscillation_period * 1e3:.3f} мс"
            if math.isfinite(result.oscillation_period)
            else "T₂: режим без колебаний"
        ),
        f"τ_damp = {result.damping_time * 1e3:.3f} мс",
        (
            f"N_e ≈ {result.oscillations_until_1e:.2f}"
            if math.isfinite(result.oscillations_until_1e)
            else "N_e: не определено"
        ),
        f"Шагов интегрирования мостика: {result.bridge_steps}",
    ]
    lines.extend(result.notes)
    return lines


def sweep_parameter(
    base_params: SimulationParams,
    variable: SweepVariable,
    start: float,
    stop: float,
    points: int,
) -> list[dict[str, float | str]]:
    if points < 2:
        raise ValueError("Для эксперимента нужно минимум 2 точки.")
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("Границы диапазона должны быть положительными.")

    values = np.linspace(start, stop, points)
    rows: list[dict[str, float | str]] = []
    for value in values:
        params = SimulationParams(**asdict(base_params))
        setattr(params, variable, float(value))
        params.validate()

        bridge = integrate_bridge(
            params,
            t_stop=max(0.5, 20.0 * capillary_time(params), 20.0 * viscous_time(params), params.total_time_s),
        )
        r_eq, omega0, beta, omega_d, period = oscillation_characteristics(params)
        rows.append(
            {
                "variable": variable,
                "value": float(value),
                "oh": ohnesorge(params),
                "merge_time_ms": bridge.merge_time * 1e3 if math.isfinite(bridge.merge_time) else math.nan,
                "capillary_time_ms": capillary_time(params) * 1e3,
                "viscous_time_ms": viscous_time(params) * 1e3,
                "equivalent_radius_mm": r_eq * 1e3,
                "omega2_rad_s": omega0,
                "beta2_s_inv": beta,
                "period_ms": period * 1e3 if math.isfinite(period) else math.nan,
                "damping_time_ms": (1.0 / beta) * 1e3 if beta > 0.0 else math.inf,
                "oscillations_until_1e": (omega_d / beta / (2.0 * math.pi)) if (beta > 0.0 and omega_d > 0.0) else math.nan,
            }
        )
    return rows


def infer_experiment_comment(variable: SweepVariable, rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return "Нет данных эксперимента."

    values = np.array([float(row["value"]) for row in rows], dtype=float)
    merge_ms = np.array([float(row["merge_time_ms"]) for row in rows], dtype=float)
    period_ms = np.array([float(row["period_ms"]) for row in rows], dtype=float)
    damping_ms = np.array([float(row["damping_time_ms"]) for row in rows], dtype=float)

    def monotone_trend(array: np.ndarray) -> str:
        finite = array[np.isfinite(array)]
        diffs = np.diff(finite)
        if len(diffs) == 0:
            return "не удалось определить"
        if np.all(diffs >= -1e-10):
            return "монотонно растет"
        if np.all(diffs <= 1e-10):
            return "монотонно убывает"
        return "меняется немонотонно"

    if variable == "radius_mm":
        valid = np.isfinite(merge_ms) & (merge_ms > 0.0) & (values > 0.0)
        slope_msg = ""
        if valid.sum() >= 2:
            slope = np.polyfit(np.log(values[valid]), np.log(merge_ms[valid]), 1)[0]
            slope_msg = f" Оценка степенного закона для t_merge: ~ R^{slope:.2f}."
        return (
            f"При изменении радиуса время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damping_ms)}.{slope_msg}"
        )
    if variable == "mu_mpas":
        return (
            f"При изменении вязкости время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damping_ms)}. Рост μ усиливает демпфирование."
        )
    if variable == "sigma_mnm":
        return (
            f"При изменении поверхностного натяжения время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damping_ms)}. Большое σ ускоряет процесс."
        )
    return (
        f"При изменении плотности время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
        f"а время затухания {monotone_trend(damping_ms)}."
    )


def export_sweep_to_csv(rows: Iterable[dict[str, float | str]], path: str | Path) -> None:
    row_list = list(rows)
    if not row_list:
        raise ValueError("Нечего экспортировать: список результатов пуст.")

    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0].keys()))
        writer.writeheader()
        writer.writerows(row_list)


def model_description_text() -> str:
    return (
        "Учебная 3D-модель использует reduced-order описание слипания капель в невесомости.\n\n"
        "1. На первой стадии растет жидкий мостик между двумя одинаковыми каплями. "
        "Радиус шейки r_n(t) вычисляется по композитному закону, который сочетает вязкостно-капиллярный "
        "и инерционно-капиллярный режимы.\n"
        "2. После достижения порога merge_ratio система переходит к объединенной капле эквивалентного "
        "радиуса R_eq = 2^(1/3) R.\n"
        "3. Форма объединенной капли задается осесимметричным возмущением r(θ,t) = R_eq [1 + A(t) P2(cos θ)], "
        "где P2(cos θ) = (3 cos^2 θ - 1)/2.\n"
        "4. Амплитуда A(t) удовлетворяет линейному уравнению затухающей моды A'' + 2βA' + ω0^2 A = 0.\n"
        "5. Невесомость учитывается через Bo = 0, поэтому форма не деформируется силой тяжести.\n\n"
        "3D-визуализация строится как тело вращения осесимметричного профиля вокруг оси слияния."
    )
