import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_result(result, output, manifest):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"time", "states", "controls", "references"}
    metrics = {k: v for k, v in result.items() if k not in arrays}
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (path / "timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_s", "x_m", "y_m", "yaw_rad", "speed_mps", "accel_mps2", "steer_rad", "ref_y_m"]
        )
        writer.writerows(
            np.column_stack(
                (result["time"], result["states"], result["controls"], result["references"][:, 1])
            )
        )
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].plot(result["states"][:, 0], result["states"][:, 1], label="vehicle")
    axes[0].plot(result["references"][:, 0], result["references"][:, 1], "--", label="reference")
    axes[0].set(xlabel="x [m]", ylabel="y [m]")
    axes[0].grid()
    axes[0].legend()
    axes[1].plot(result["time"], result["controls"][:, 1])
    axes[1].set(xlabel="time [s]", ylabel="steer [rad]")
    axes[1].grid()
    fig.tight_layout()
    fig.savefig(path / "response.png", dpi=150)
    plt.close(fig)
