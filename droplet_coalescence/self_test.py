from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coalescence_core import (
    SimulationParams,
    make_spatial_grid,
    post_merge_boundary,
    post_merge_radial_field,
    simulate,
    soft_union_field,
    sweep_parameter,
    export_sweep_to_csv,
)


def save_demo_plots(out_dir: str | Path = ".") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SimulationParams()
    result = simulate(params)
    X, Y, extent = make_spatial_grid(params.radius_m)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1.35, 1])

    frame_ids = [0, len(result.frame_indices) // 3, -1]
    titles = ["Начало", "Рост мостика", "Релаксация"]
    for k, fid in enumerate(frame_ids):
        ax = fig.add_subplot(gs[0, k])
        idx = result.frame_indices[fid]
        blend = float(result.blend_weight[idx]) if hasattr(result, "blend_weight") else float(result.stage[idx])
        if blend < 1e-3:
            field = soft_union_field(X, Y, params.radius_m, result.neck_radius[idx])
            ax.contourf(X * 1e3, Y * 1e3, field, levels=[field.min(), 0.0], colors=["#8ecae6"], alpha=0.95)
            ax.contour(X * 1e3, Y * 1e3, field, levels=[0.0], colors=["#023047"], linewidths=2.0)
            ax.set_title(f"{titles[k]}\nt = {result.t[idx] * 1e3:.3f} мс")
        elif blend > 1.0 - 1e-3:
            x, y = post_merge_boundary(result.equivalent_radius, result.mode_amplitude[idx])
            ax.fill(x * 1e3, y * 1e3, color="#8ecae6", alpha=0.95)
            ax.plot(x * 1e3, y * 1e3, color="#023047", linewidth=2.0)
            ax.set_title(f"{titles[k]}\nt = {result.t[idx] * 1e3:.3f} мс")
        else:
            field_pre = soft_union_field(X, Y, params.radius_m, result.neck_radius[idx])
            field_post = post_merge_radial_field(X, Y, result.equivalent_radius, result.mode_amplitude[idx])
            field = (1.0 - blend) * field_pre + blend * field_post
            ax.contourf(
                X * 1e3,
                Y * 1e3,
                field,
                levels=[field.min(), 0.0],
                colors=["#8ecae6"],
                alpha=0.95,
            )
            ax.contour(
                X * 1e3,
                Y * 1e3,
                field,
                levels=[0.0],
                colors=["#023047"],
                linewidths=2.0,
                alpha=1.0,
            )
            ax.set_title(f"{titles[k]} (переход)\nt = {result.t[idx] * 1e3:.3f} мс")
        ax.set_aspect("equal")
        ax.set_xlim(extent[0] * 1e3, extent[1] * 1e3)
        ax.set_ylim(extent[2] * 1e3, extent[3] * 1e3)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("x, мм")
        ax.set_ylabel("y, мм")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(result.t * 1e3, result.neck_radius / params.radius_m, linewidth=2.2)
    if math.isfinite(result.merge_time):
        ax1.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
    ax1.set_title("Рост перешейка")
    ax1.set_xlabel("t, мс")
    ax1.set_ylabel("rₙ / R")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1:3])
    ax2.plot(result.t * 1e3, result.mode_amplitude, linewidth=2.2)
    if math.isfinite(result.merge_time):
        ax2.axvline(result.merge_time * 1e3, linestyle="--", linewidth=1.3)
    ax2.set_title("Релаксация моды l=2")
    ax2.set_xlabel("t, мс")
    ax2.set_ylabel("A(t)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "self_test_simulation.png", dpi=180)
    plt.close(fig)

    rows = sweep_parameter(params, "radius_mm", 0.5, 3.0, 8)
    export_sweep_to_csv(rows, out_dir / "self_test_radius_sweep.csv")
    x = np.array([row["value"] for row in rows], dtype=float)
    merge = np.array([row["merge_time_ms"] for row in rows], dtype=float)
    period = np.array([row["period_ms"] for row in rows], dtype=float)
    damping = np.array([row["damping_time_ms"] for row in rows], dtype=float)
    oh = np.array([row["oh"] for row in rows], dtype=float)

    fig2 = plt.figure(figsize=(12, 4.8))
    ax21 = fig2.add_subplot(1, 2, 1)
    ax22 = fig2.add_subplot(1, 2, 2)
    ax21.plot(x, merge, marker="o", linewidth=2.0)
    ax21.set_title("Sweep по радиусу: t_merge")
    ax21.set_xlabel("R, мм")
    ax21.set_ylabel("t_merge, мс")
    ax21.grid(True, alpha=0.3)

    ax22.plot(x, period, marker="o", linewidth=2.0, label="T₂, мс")
    ax22.plot(x, damping, marker="s", linewidth=2.0, label="τ_damp, мс")
    ax22b = ax22.twinx()
    ax22b.plot(x, oh, marker="^", linestyle="--", linewidth=1.8, label="Oh")
    ax22.set_title("Sweep по радиусу: колебания")
    ax22.set_xlabel("R, мм")
    ax22.set_ylabel("мс")
    ax22b.set_ylabel("Oh")
    ax22.grid(True, alpha=0.3)

    lines1, labels1 = ax22.get_legend_handles_labels()
    lines2, labels2 = ax22b.get_legend_handles_labels()
    ax22.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig2.tight_layout()
    fig2.savefig(out_dir / "self_test_experiment.png", dpi=180)
    plt.close(fig2)


if __name__ == "__main__":
    save_demo_plots(Path(__file__).resolve().parent)
    print("Self-test completed. Files written to project directory.")
