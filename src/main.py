from __future__ import annotations

import argparse

from engine import build_simulation_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10c_stream.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine, dataset = build_simulation_from_yaml(args.config)
    result = engine.run(dataset, max_steps=args.max_steps)
    metrics = result.state.metrics
    print("Simulation metrics")
    print(f"Steps: {result.state.step}")
    if "efficiency" in metrics:
        print(f"Efficiency: {metrics['efficiency']:.6f}")
        print(f"Informativeness: {metrics['informativeness']:.6f}")
        print(f"Coverage by classes: {metrics['coverage_by_class']}")
        print(f"Coverage gaps by classes: {metrics['coverage_gap_by_class']}")
    else:
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
