import argparse
from pathlib import Path

import yaml

from .core import run_episode
from .reporting import write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nominal.yaml")
    parser.add_argument("--output", default="artifacts/nominal")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = run_episode(
        config["controller"],
        config["duration_s"],
        config["sample_time_s"],
        config.get("friction_scale", 1.0),
        config.get("delay_steps", 0),
    )
    write_result(result, args.output, {"config": config, "solver": "scipy-lbfgsb"})
    print({k: v for k, v in result.items() if not hasattr(v, "shape")})


if __name__ == "__main__":
    main()
