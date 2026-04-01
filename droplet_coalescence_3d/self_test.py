from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from coalescence_core_3d import (
    SimulationParams,
    export_sweep_to_csv,
    make_3d_mesh,
    simulate,
    sweep_parameter,
)


def _save_surface_figure(output_path: Path) -> None:
    result = simulate(SimulationParams())
    frame_idx = int(0.75 * (len(result.t) - 1))
    mesh = make_3d_mesh(result, frame_idx, n_axial=160, n_azimuth=72)

    fig = plt.figure(figsize=(8.2, 6.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mesh.x * 1e3,
        mesh.y * 1e3,
        mesh.z * 1e3,
        cmap="Blues",
        linewidth=0.0,
        antialiased=True,
        alpha=0.96,
    )
    ax.set_title(mesh.title)
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_experiment_figure(output_path: Path) -> None:
    rows = sweep_parameter(SimulationParams(), "radius_mm", 0.6, 1.6, 6)
    x = np.array([float(row["value"]) for row in rows], dtype=float)
    merge = np.array([float(row["merge_time_ms"]) for row in rows], dtype=float)
    period = np.array([float(row["period_ms"]) for row in rows], dtype=float)
    damping = np.array([float(row["damping_time_ms"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=150)

    axes[0].plot(x, merge, marker="o", linewidth=2.0)
    axes[0].set_title("Время завершения слияния")
    axes[0].set_xlabel("R, мм")
    axes[0].set_ylabel("t_merge, мс")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, period, marker="o", linewidth=2.0, label="T₂, мс")
    axes[1].plot(x, damping, marker="s", linewidth=2.0, label="τ_damp, мс")
    axes[1].set_title("Осцилляции и затухание")
    axes[1].set_xlabel("R, мм")
    axes[1].set_ylabel("мс")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_self_test(output_dir: str | Path | None = None) -> list[Path]:
    output_root = Path(output_dir) if output_dir is not None else Path.cwd()
    output_root.mkdir(parents=True, exist_ok=True)

    surface_png = output_root / "self_test_3d_surface.png"
    experiment_png = output_root / "self_test_experiment.png"
    sweep_csv = output_root / "self_test_radius_sweep.csv"

    _save_surface_figure(surface_png)
    _save_experiment_figure(experiment_png)
    export_sweep_to_csv(
        sweep_parameter(SimulationParams(), "radius_mm", 0.6, 1.6, 6),
        sweep_csv,
    )
    return [surface_png, experiment_png, sweep_csv]


if __name__ == "__main__":
    for artifact in run_self_test():
        print(artifact)
