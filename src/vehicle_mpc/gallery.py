"""Build a reproducible website gallery for the path-tracking benchmark."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .core import run_episode

INK = "#0f172a"
BLUE = "#2563eb"
ORANGE = "#f59e0b"
SLATE = "#64748b"
GRID = "#cbd5e1"
PAPER = "#f8fafc"
CONTROLLERS = ["pure-pursuit", "lqr", "mpc"]
SCENARIOS = {
    "nominal": {},
    "low friction": {"friction_scale": 0.65},
    "200 ms latency": {"delay_steps": 2},
}


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=SLATE)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=125, facecolor=PAPER)
    plt.close(figure)


def benchmark_results() -> dict[str, dict[str, dict]]:
    return {
        controller: {
            scenario: run_episode(controller, **settings) for scenario, settings in SCENARIOS.items()
        }
        for controller in CONTROLLERS
    }


def _hero(results: dict[str, dict[str, dict]], output: Path) -> None:
    nominal = {name: results[name]["nominal"] for name in CONTROLLERS}
    reference = nominal["mpc"]["references"]
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), gridspec_kw={"height_ratios": [2.0, 1.0]})
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("Constrained path tracking on a curved road", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "Three controller families share one plant, reference, lane boundary, and metric contract.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    axes[0].fill_between(
        reference[:, 0],
        reference[:, 1] - 1.5,
        reference[:, 1] + 1.5,
        color=BLUE,
        alpha=0.08,
        label="lane envelope",
    )
    axes[0].plot(
        reference[:, 0], reference[:, 1], color=INK, linestyle="--", linewidth=1.7, label="reference"
    )
    styles = {
        "pure-pursuit": (BLUE, "-"),
        "lqr": (ORANGE, "--"),
        "mpc": (INK, "-."),
    }
    for name, result in nominal.items():
        color, linestyle = styles[name]
        axes[0].plot(
            result["states"][:, 0],
            result["states"][:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=name,
        )
    axes[0].set(xlabel="longitudinal position x [m]", ylabel="lateral position y [m]")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")
    for name, result in nominal.items():
        error = result["states"][:, 1] - result["references"][:, 1]
        color, linestyle = styles[name]
        axes[1].plot(result["time"], error, color=color, linestyle=linestyle, linewidth=1.9, label=name)
    axes[1].axhspan(-1.5, 1.5, color=BLUE, alpha=0.08)
    axes[1].set(xlabel="time [s]", ylabel="lateral error [m]")
    for axis in axes:
        _style(axis)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.10, hspace=0.30)
    _save(figure, output / "hero.png")


def _benchmark(results: dict[str, dict[str, dict]], output: Path) -> None:
    rmse = np.array(
        [
            [results[controller][scenario]["lateral_rmse_m"] for scenario in SCENARIOS]
            for controller in CONTROLLERS
        ]
    )
    violations = np.array(
        [
            [results[controller][scenario]["constraint_violations"] for scenario in SCENARIOS]
            for controller in CONTROLLERS
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("Tracking robustness and solver timing", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "Heatmap cells show RMSE with constraint violations in parentheses; timing uses measured solves.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    image = axes[0].imshow(rmse, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(SCENARIOS)), list(SCENARIOS), rotation=20, ha="right")
    axes[0].set_yticks(range(len(CONTROLLERS)), CONTROLLERS)
    axes[0].set_title("Lateral RMSE [m]  (violations)", color=INK, pad=14)
    for row in range(rmse.shape[0]):
        for column in range(rmse.shape[1]):
            color = "white" if rmse[row, column] > 0.75 * rmse.max() else INK
            axes[0].text(
                column,
                row,
                f"{rmse[row, column]:.2f}  ({violations[row, column]})",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("RMSE [m]", color=INK)
    mpc = results["mpc"]
    median = [mpc[name]["median_solve_time_ms"] for name in SCENARIOS]
    p99 = [mpc[name]["p99_solve_time_ms"] for name in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    width = 0.34
    axes[1].bar(x - width / 2, median, width, color=BLUE, edgecolor=INK, linewidth=0.6, label="median")
    axes[1].bar(
        x + width / 2, p99, width, color="#bfdbfe", edgecolor=INK, linewidth=0.6, hatch="//", label="p99"
    )
    axes[1].axhline(20.0, color=ORANGE, linestyle="--", linewidth=2, label="20 ms target")
    axes[1].set(xticks=x, xticklabels=list(SCENARIOS), ylabel="MPC solve time [ms]")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)
    _style(axes[1])
    axes[0].set_facecolor(PAPER)
    figure.subplots_adjust(left=0.09, right=0.96, top=0.85, bottom=0.18, wspace=0.35)
    _save(figure, output / "benchmark.png")


def _animation(result: dict, output: Path) -> None:
    indices = np.linspace(0, len(result["time"]) - 1, 80, dtype=int)
    reference = result["references"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    figure.patch.set_facecolor(PAPER)
    axis.set_facecolor(PAPER)
    axis.fill_between(reference[:, 0], reference[:, 1] - 1.5, reference[:, 1] + 1.5, color=BLUE, alpha=0.08)
    axis.plot(reference[:, 0], reference[:, 1], color=INK, linestyle="--", linewidth=1.5)
    axis.set(
        xlim=(reference[:, 0].min() - 1, reference[:, 0].max() + 2),
        ylim=(reference[:, 1].min() - 2, reference[:, 1].max() + 2),
        xlabel="x [m]",
        ylabel="y [m]",
    )
    axis.grid(color=GRID, linewidth=0.8)
    (trail,) = axis.plot([], [], color=BLUE, linewidth=2.0)
    (vehicle,) = axis.plot([], [], color=ORANGE, linewidth=6, marker="o", markersize=5)
    status = axis.text(0.03, 0.92, "", transform=axis.transAxes, color=INK, fontsize=11)

    def update(frame: int):
        index = indices[frame]
        state = result["states"][index]
        length = 1.0
        dx, dy = length * np.cos(state[2]), length * np.sin(state[2])
        trail.set_data(result["states"][: index + 1, 0], result["states"][: index + 1, 1])
        vehicle.set_data([state[0] - dx / 2, state[0] + dx / 2], [state[1] - dy / 2, state[1] + dy / 2])
        error = state[1] - result["references"][index, 1]
        status.set_text(f"t = {result['time'][index]:.1f} s   eᵧ = {error:+.2f} m")
        return trail, vehicle, status

    animation = FuncAnimation(figure, update, frames=len(indices), interval=70, blit=True)
    animation.save(output / "demo.gif", writer=PillowWriter(fps=14), dpi=90)
    plt.close(figure)


def _architecture(output: Path) -> None:
    nodes = [
        (35, "Reference path", "position · yaw · speed"),
        (285, "State error", "lateral + heading"),
        (535, "Nonlinear MPC", "horizon + constraints"),
        (785, "Steer / accel", "bounded inputs"),
        (1035, "Bicycle model", "RK4 + latency queue"),
    ]
    elements = []
    for index, (x, title, subtitle) in enumerate(nodes):
        elements.append(
            f'<rect x="{x}" y="84" width="200" height="88" rx="14" fill="white" stroke="{BLUE if index == 2 else GRID}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{x + 100}" y="119" text-anchor="middle" fill="{INK}" font-family="Arial" font-size="15">{html.escape(title)}</text>'
        )
        elements.append(
            f'<text x="{x + 100}" y="145" text-anchor="middle" fill="{SLATE}" font-family="Arial" font-size="12">{html.escape(subtitle)}</text>'
        )
        if index < len(nodes) - 1:
            elements.append(
                f'<line x1="{x + 200}" y1="128" x2="{nodes[index + 1][0] - 12}" y2="128" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1270" height="270" viewBox="0 0 1270 270"><rect width="1270" height="270" fill="{PAPER}"/><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{INK}"/></marker></defs><text x="35" y="42" fill="{INK}" font-family="Arial" font-size="22" font-weight="700">Receding-horizon path-tracking loop</text>{"".join(elements)}<path d="M1135 184 C1135 234, 385 234, 385 184" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/><text x="760" y="254" text-anchor="middle" fill="{SLATE}" font-family="Arial" font-size="13">measured vehicle state</text></svg>'''
    (output / "architecture.svg").write_text(svg)


def gallery_contract(results: dict[str, dict[str, dict]]) -> dict:
    nominal = results["mpc"]["nominal"]
    return {
        "schema_version": 1,
        "repository": "vehicle-path-tracking-mpc",
        "title": "Vehicle Path Tracking MPC",
        "tagline": "Pure Pursuit, LQR, and constrained nonlinear MPC on one benchmark.",
        "accent": BLUE,
        "highlights": [
            {"label": "nominal MPC RMSE", "value": f"{nominal['lateral_rmse_m']:.3f} m"},
            {"label": "constraint violations", "value": str(nominal["constraint_violations"])},
            {"label": "median solve", "value": f"{nominal['median_solve_time_ms']:.1f} ms"},
        ],
        "assets": [
            {
                "path": "hero.png",
                "role": "hero",
                "width": 1600,
                "height": 900,
                "alt": "Three controllers tracking a shared curved road reference.",
            },
            {
                "path": "benchmark.png",
                "role": "analysis",
                "width": 1600,
                "height": 900,
                "alt": "Tracking-error heatmap and MPC solve-time comparison.",
            },
            {
                "path": "demo.gif",
                "role": "animation",
                "width": 720,
                "height": 405,
                "alt": "Animated nonlinear MPC vehicle following the lane center.",
            },
            {
                "path": "architecture.svg",
                "role": "diagram",
                "width": 1270,
                "height": 270,
                "alt": "Reference, MPC, constrained actuation, and bicycle-model loop.",
            },
        ],
        "reproduce": "python -m vehicle_mpc.gallery --output artifacts/gallery",
    }


def generate_gallery(output: str | Path, animation: bool = True) -> dict:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = benchmark_results()
    _hero(results, path)
    _benchmark(results, path)
    _architecture(path)
    if animation:
        _animation(results["mpc"]["nominal"], path)
    rows = [
        {
            "controller": controller,
            "scenario": scenario,
            "lateral_rmse_m": result["lateral_rmse_m"],
            "constraint_violations": result["constraint_violations"],
            "median_solve_time_ms": result["median_solve_time_ms"],
            "p99_solve_time_ms": result["p99_solve_time_ms"],
        }
        for controller, scenarios in results.items()
        for scenario, result in scenarios.items()
    ]
    with (path / "benchmark_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    contract = gallery_contract(results)
    (path / "showcase.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gallery")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate_gallery(args.output, not args.no_animation)["highlights"], indent=2))


if __name__ == "__main__":
    main()
