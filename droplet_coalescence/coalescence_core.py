from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


if sys.version_info >= (3, 10):
    model_dataclass = partial(dataclass, slots=True)
else:
    model_dataclass = dataclass


@model_dataclass
class SimulationParams:
    """Параметры reduced-order модели слипания капель."""

    radius_mm: float = 1.0
    rho: float = 997.0
    mu_mpas: float = 1.0
    sigma_mnm: float = 72.0
    initial_neck_ratio: float = 0.02
    initial_mode_amplitude: float = 0.22
    total_time_ms: float = 80.0
    merge_ratio: float = 0.95
    bridge_time_scale: float = 3.0
    inertial_const: float = 1.62
    ilv_const: float = 1.0
    animation_frames: int = 180

    def validate(self) -> None:
        checks = {
            "radius_mm": self.radius_mm,
            "rho": self.rho,
            "mu_mpas": self.mu_mpas,
            "sigma_mnm": self.sigma_mnm,
            "initial_neck_ratio": self.initial_neck_ratio,
            "total_time_ms": self.total_time_ms,
            "merge_ratio": self.merge_ratio,
            "bridge_time_scale": self.bridge_time_scale,
            "animation_frames": self.animation_frames,
        }
        for name, value in checks.items():
            if value <= 0:
                raise ValueError(f"Параметр {name} должен быть положительным.")
        if not (0 < self.initial_neck_ratio < 1):
            raise ValueError("Начальный радиус перешейка должен удовлетворять 0 < r0/R < 1.")
        if not (0 < self.merge_ratio <= 1):
            raise ValueError("Порог завершения слияния должен лежать в пределах (0, 1].")
        if not (0 <= self.initial_mode_amplitude < 0.8):
            raise ValueError("Начальная амплитуда моды должна лежать в диапазоне [0, 0.8).")
        if self.animation_frames < 10:
            raise ValueError("Число кадров анимации должно быть не меньше 10.")

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
    oscillation_period: float | float
    damping_time: float
    oscillations_until_1e: float | float
    bridge_steps: int
    notes: list[str]

    @property
    def frame_indices(self) -> np.ndarray:
        idx = np.linspace(0, len(self.t) - 1, self.params.animation_frames)
        idx = np.unique(idx.astype(int))
        if idx[-1] != len(self.t) - 1:
            idx = np.append(idx, len(self.t) - 1)
        return idx


SweepVariable = Literal["radius_mm", "mu_mpas", "sigma_mnm", "rho"]



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
    """Скорость роста радиуса мостика.

    Используется composite-law closure:
    - ранний линейный режим (ILV / вязкостно-инерционный): r ~ t,
    - последующий инерционно-капиллярный режим: r ~ t^{1/2}.

    Скорости объединяются через гармоническое среднее.
    """
    r = max(neck_radius, 1e-12)
    v_linear = params.ilv_const * params.sigma / params.mu
    v_inertial = 0.5 * params.inertial_const**2 * math.sqrt(params.sigma * params.radius_m / params.rho) / r
    v_comp = 1.0 / (1.0 / v_linear + 1.0 / v_inertial)
    return v_comp / max(params.bridge_time_scale, 1e-12)


@model_dataclass
class BridgeIntegration:
    t: np.ndarray
    r: np.ndarray
    merge_time: float
    steps: int
    merged: bool



def integrate_bridge(
    params: SimulationParams,
    t_stop: float | None = None,
    max_steps: int = 200_000,
) -> BridgeIntegration:
    """Адаптивное интегрирование роста шейки до времени t_stop.

    Шаг подбирается так, чтобы относительное изменение r на одном шаге было малым.
    """
    params.validate()
    R = params.radius_m
    r_target = params.merge_ratio * R
    t_limit = params.total_time_s if t_stop is None else t_stop

    t_values = [0.0]
    r_values = [params.initial_neck_ratio * R]

    t = 0.0
    r = params.initial_neck_ratio * R
    merged = r >= r_target

    steps = 0
    while t < t_limit and not merged and steps < max_steps:
        v = bridge_velocity(r, params)

        dt_rel = 0.015 * max(r, 1e-12) / max(v, 1e-12)
        dt_cap = max(t_limit / 5000.0, 2.5e-7)
        dt = min(dt_rel, dt_cap, 2.5e-4)
        dt = max(dt, 1e-9)

        r = min(r + v * dt, r_target)
        t = min(t + dt, t_limit)
        t_values.append(t)
        r_values.append(r)
        merged = r >= r_target * (1.0 - 1e-9)
        steps += 1

    merge_time = t if merged else math.nan
    return BridgeIntegration(
        t=np.asarray(t_values, dtype=float),
        r=np.asarray(r_values, dtype=float),
        merge_time=merge_time,
        steps=steps,
        merged=merged,
    )



def oscillation_characteristics(params: SimulationParams) -> tuple[float, float, float, float, float]:
    """Возвращает (R_eq, omega0, beta, omega_d, T_damped)."""
    R_eq = equivalent_radius(params)
    omega0 = math.sqrt(8.0 * params.sigma / (params.rho * R_eq**3))
    beta = 5.0 * params.mu / (params.rho * R_eq**2)
    if beta < omega0:
        omega_d = math.sqrt(max(omega0**2 - beta**2, 0.0))
        T = 2.0 * math.pi / omega_d
    else:
        omega_d = 0.0
        T = math.nan
    return R_eq, omega0, beta, omega_d, T



def mode_amplitude(time_from_merge: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Решение линейного уравнения A'' + 2βA' + ω0^2 A = 0, A(0)=A0, A'(0)=0."""
    A0 = params.initial_mode_amplitude
    R_eq, omega0, beta, omega_d, _ = oscillation_characteristics(params)
    _ = R_eq
    tau = time_from_merge

    if beta < omega0 and omega_d > 0:
        return A0 * np.exp(-beta * tau) * (np.cos(omega_d * tau) + (beta / omega_d) * np.sin(omega_d * tau))


    disc = max(beta**2 - omega0**2, 0.0)
    s = math.sqrt(disc)
    lam1 = -beta + s
    lam2 = -beta - s
    if abs(lam1 - lam2) < 1e-12:
        return A0 * np.exp(lam1 * tau) * (1.0 - lam1 * tau)
    c1 = -A0 * lam2 / (lam1 - lam2)
    c2 = A0 * lam1 / (lam1 - lam2)
    return c1 * np.exp(lam1 * tau) + c2 * np.exp(lam2 * tau)


def smoothstep01(x: np.ndarray | float) -> np.ndarray | float:
    """Гладкая функция 0..1 с нулевой производной на концах."""
    x_arr = np.clip(x, 0.0, 1.0)
    return x_arr * x_arr * (3.0 - 2.0 * x_arr)


def transition_time(params: SimulationParams) -> float:
    """Характерная длительность сглаженного перехода между стадиями."""
    tc = capillary_time(params)
    tv = viscous_time(params)
    return max(0.15 * tc, 0.60 * tv, params.total_time_s / 400.0)



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
    A = np.zeros_like(t_uniform)
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


        A_raw = mode_amplitude(t_uniform[post_mask] - merge_time, params)
        A[post_mask] = blend_post * A_raw

        stage = (blend >= 0.5).astype(int)
    else:
        merge_time = math.nan

    R_eq, omega0, beta, omega_d, T = oscillation_characteristics(params)
    damping_time = 1.0 / beta if beta > 0 else math.inf
    osc_until_1e = (omega_d * damping_time / (2.0 * math.pi)) if omega_d > 0 else math.nan

    return SimulationResult(
        params=params,
        t=t_uniform,
        neck_radius=neck_uniform,
        mode_amplitude=A,
        blend_weight=blend,
        stage=stage,
        merge_time=merge_time,
        transition_time=transition if bridge.merged else 0.0,
        capillary_time=capillary_time(params),
        viscous_time=viscous_time(params),
        ohnesorge=ohnesorge(params),
        bond=bond_number_microgravity(),
        equivalent_radius=R_eq,
        omega2=omega0,
        beta2=beta,
        damped_omega2=omega_d,
        oscillation_period=T,
        damping_time=damping_time,
        oscillations_until_1e=osc_until_1e,
        bridge_steps=bridge.steps,
        notes=notes,
    )



def epsilon_from_neck(neck_radius: float, radius: float) -> float:
    """Параметр soft-min level-set, обеспечивающий заданный радиус шейки в x=0."""
    ln2 = math.log(2.0)
    return max((math.sqrt(radius * radius + neck_radius * neck_radius) - radius) / ln2, 1e-12)



def soft_union_field(X: np.ndarray, Y: np.ndarray, radius: float, neck_radius: float) -> np.ndarray:
    eps = epsilon_from_neck(neck_radius, radius)
    d1 = np.hypot(X - radius, Y) - radius
    d2 = np.hypot(X + radius, Y) - radius
    m = np.minimum(d1, d2)
    field = m - eps * np.log(np.exp(-(d1 - m) / eps) + np.exp(-(d2 - m) / eps))
    return field



def post_merge_boundary(
    equivalent_radius_value: float,
    amplitude: float,
    n_points: int = 721,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, n_points)
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    r = equivalent_radius_value * (1.0 + amplitude * p2)
    r = np.clip(r, 0.05 * equivalent_radius_value, None)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y



def post_merge_radial_field(
    X: np.ndarray,
    Y: np.ndarray,
    equivalent_radius_value: float,
    amplitude: float,
) -> np.ndarray:
    """Аппроксимация signed-distance поля для формы после слияния."""
    theta = np.arctan2(Y, X)
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    rb = equivalent_radius_value * (1.0 + amplitude * p2)
    rb = np.clip(rb, 0.05 * equivalent_radius_value, None)
    return np.hypot(X, Y) - rb


def make_spatial_grid(radius: float, nx: int = 260, ny: int = 220) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x_lim = 2.25 * radius
    y_lim = 1.55 * radius
    x = np.linspace(-x_lim, x_lim, nx)
    y = np.linspace(-y_lim, y_lim, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y, (-x_lim, x_lim, -y_lim, y_lim)



def result_summary_lines(result: SimulationResult) -> list[str]:
    p = result.params
    lines = [
        f"R = {p.radius_mm:.3f} мм",
        f"ρ = {p.rho:.1f} кг/м³",
        f"μ = {p.mu_mpas:.3f} мПа·с",
        f"σ = {p.sigma_mnm:.3f} мН/м",
        f"k_slow = {p.bridge_time_scale:.3f}",
        f"Oh = {result.ohnesorge:.5f}",
        f"Bo = {result.bond:.1f} (невесомость)",
        f"τ_c = {result.capillary_time * 1e3:.3f} мс",
        f"τ_v = {result.viscous_time * 1e3:.3f} мс",
        (
            f"t_merge ≈ {result.merge_time * 1e3:.3f} мс"
            if math.isfinite(result.merge_time)
            else "t_merge: не достигнуто в выбранном интервале"
        ),
        (
            f"Δt_transition ≈ {result.transition_time * 1e3:.3f} мс"
            if result.transition_time > 0
            else "Δt_transition: не используется"
        ),
        f"R_eq = {result.equivalent_radius * 1e3:.3f} мм",
        f"ω₂ = {result.omega2:.2f} рад/с",
        f"β₂ = {result.beta2:.2f} 1/с",
        (
            f"T₂ ≈ {result.oscillation_period * 1e3:.3f} мс"
            if math.isfinite(result.oscillation_period)
            else "T₂: режим без колебаний (пере/критическое затухание)"
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
    if start <= 0 or stop <= 0:
        raise ValueError("Границы диапазона должны быть положительными.")

    values = np.linspace(start, stop, points)
    rows: list[dict[str, float | str]] = []
    for value in values:
        params = SimulationParams(**asdict(base_params))
        setattr(params, variable, float(value))
        params.validate()


        bridge = integrate_bridge(params, t_stop=max(0.5, 20.0 * capillary_time(params), 20.0 * viscous_time(params), params.total_time_s))
        R_eq, omega0, beta, omega_d, T = oscillation_characteristics(params)
        rows.append(
            {
                "variable": variable,
                "value": float(value),
                "oh": ohnesorge(params),
                "merge_time_ms": bridge.merge_time * 1e3 if math.isfinite(bridge.merge_time) else math.nan,
                "capillary_time_ms": capillary_time(params) * 1e3,
                "viscous_time_ms": viscous_time(params) * 1e3,
                "equivalent_radius_mm": R_eq * 1e3,
                "omega2_rad_s": omega0,
                "beta2_s_inv": beta,
                "period_ms": T * 1e3 if math.isfinite(T) else math.nan,
                "damping_time_ms": (1.0 / beta) * 1e3 if beta > 0 else math.inf,
                "oscillations_until_1e": (omega_d / beta / (2.0 * math.pi)) if (beta > 0 and omega_d > 0) else math.nan,
            }
        )
    return rows



def infer_experiment_comment(variable: SweepVariable, rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return "Нет данных эксперимента."
    values = np.array([float(row["value"]) for row in rows], dtype=float)
    merge_ms = np.array([float(row["merge_time_ms"]) for row in rows], dtype=float)
    period_ms = np.array([float(row["period_ms"]) for row in rows], dtype=float)
    damp_ms = np.array([float(row["damping_time_ms"]) for row in rows], dtype=float)

    def monotone_trend(arr: np.ndarray) -> str:
        diffs = np.diff(arr[np.isfinite(arr)])
        if len(diffs) == 0:
            return "не удалось определить"
        if np.all(diffs >= -1e-10):
            return "монотонно растет"
        if np.all(diffs <= 1e-10):
            return "монотонно убывает"
        return "меняется немонотонно"

    if variable == "radius_mm":
        valid = np.isfinite(merge_ms) & (merge_ms > 0) & (values > 0)
        slope_msg = ""
        if valid.sum() >= 2:
            slope = np.polyfit(np.log(values[valid]), np.log(merge_ms[valid]), 1)[0]
            slope_msg = f" Оценка степенного закона для t_merge: ~ R^{slope:.2f}."
        return (
            f"При изменении радиуса время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damp_ms)}.{slope_msg}"
        )
    if variable == "mu_mpas":
        return (
            f"При изменении вязкости время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damp_ms)}. Обычно рост μ усиливает демпфирование."
        )
    if variable == "sigma_mnm":
        return (
            f"При изменении поверхностного натяжения время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
            f"а время затухания {monotone_trend(damp_ms)}. Более высокое σ обычно ускоряет процесс."
        )
    return (
        f"При изменении плотности время слияния {monotone_trend(merge_ms)}, период колебаний {monotone_trend(period_ms)}, "
        f"а время затухания {monotone_trend(damp_ms)}."
    )



def export_sweep_to_csv(rows: Iterable[dict[str, float | str]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Нет данных для экспорта.")
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def model_description_text() -> str:
    return """\
МАТЕМАТИЧЕСКАЯ МОДЕЛЬ
=====================

1) Постановка
-------------
Рассматриваются две одинаковые капли воды радиуса R в условиях невесомости.
Приняты допущения:
- капли одинаковые и осесимметричные;
- внешняя среда — газ с малой плотностью и вязкостью;
- гравитация отсутствует, поэтому число Бонда Bo = 0;
- жидкость ньютоновская, свойства постоянные.

2) Ранняя стадия: рост жидкого мостика r_n(t)
---------------------------------------------
Используется reduced-order composite law:

    dr_n/dt = [ ( 1 / v_lin + 1 / v_in )^(-1) ] / k_slow,

где

    v_lin = C_v * sigma / mu,
    v_in  = (C_i^2 / 2) * sqrt(sigma * R / rho) / r_n.

Здесь sigma — поверхностное натяжение, mu — динамическая вязкость,
rho — плотность, а k_slow >= 1 — коэффициент замедления визуально-расчетного
темпа слипания (по умолчанию 3.0). Такая формула специально объединяет два характерных режима:
- ранний линейный (r_n ~ t),
- инерционно-капиллярный (r_n ~ t^(1/2)).

3) Поздняя стадия: релаксация одной капли
-----------------------------------------
После достижения r_n >= eta * R (по умолчанию eta = 0.95) считаем,
что образовалась одна капля эквивалентного радиуса

    R_eq = 2^(1/3) * R.

Ее форма описывается доминирующей осесимметричной модой l = 2:

    A'' + 2*beta*A' + omega_2^2 * A = 0,

где

    omega_2^2 = 8 * sigma / (rho * R_eq^3),
    beta      = 5 * mu / (rho * R_eq^2).

Чтобы убрать нефизичный разрыв между стадиями, используется гладкая
переходная функция w(t) = smoothstep((t - t_merge)/Delta_t), 0 <= w <= 1,
где Delta_t выбирается из характерных времен (tau_c, tau_v).

Тогда в переходном окне:
    r_n/R = eta + (1 - eta) * w(t),
    A_eff(t) = w(t) * A_raw(t).

Геометрия поверхности в меридиональном сечении задается как

    r(theta, t) = R_eq * [1 + A_eff(t) * P2(cos(theta))],
    P2(x) = (3x^2 - 1)/2.

4) Визуализация ранней стадии
-----------------------------
Для плавного отображения двух слипающихся капель используется level-set модель
soft-union двух окружностей:

    d_eps = -eps * ln( exp(-d1/eps) + exp(-d2/eps) ),

где d1 и d2 — signed distance до каждой капли. Параметр eps выбирается так,
чтобы в сечении x = 0 радиус перешейка совпадал с текущим r_n(t).

5) Характерные числа
--------------------
Программа автоматически считает:
- капиллярное время  tau_c = sqrt(rho * R^3 / sigma),
- вязкостное время   tau_v = mu * R / sigma,
- число Онезорге     Oh    = mu / sqrt(rho * sigma * R),
- число Бонда        Bo    = 0.

6) Что именно моделирует программа
----------------------------------
Это не DNS/VOF/Level-Set решение полной системы Навье—Стокса,
а физически осмысленная reduced-order модель для учебного и инженерного анализа.
Она хорошо передает:
- характерные времена процесса;
- закон роста мостика;
- последующие затухающие колебания объединенной капли;
- влияние R, mu, sigma и rho на динамику.
"""
