"""Generate fixed canonical grid maps for DDID-Bench.

Run this script once to create map YAML files. The generated files should be
committed to Git and reused unchanged in experiments.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any

import yaml

Coordinate = tuple[int, int]


def generate_map(
    *,
    size: int,
    obstacle_probability: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one valid map with a reachable target."""

    while True:
        agent_start: Coordinate = (0, 0)
        target_location: Coordinate = (size - 2, size - 2)

        obstacles: set[Coordinate] = set()

        for row in range(size):
            for column in range(size):
                coordinate = (row, column)

                if coordinate in {
                    agent_start,
                    target_location,
                }:
                    continue

                if rng.random() < obstacle_probability:
                    obstacles.add(coordinate)

        if _path_exists(
            size=size,
            start=agent_start,
            target=target_location,
            obstacles=obstacles,
        ):
            break

    risk_field = [
        [0.0 for _ in range(size)]
        for _ in range(size)
    ]

    return {
        "schema_version": "1.0",
        "environment": {
            "grid_size": [size, size],
            "obstacles": [
                [row, column]
                for row, column in sorted(obstacles)
            ],
            "target_locations": [
                list(target_location)
            ],
            "risk_field": risk_field,
            "initial_exogenous_state": {},
        },
        "agents": {
            "initial_poses": [
                list(agent_start)
            ],
            "initial_health": [1.0],
        },
    }


def _path_exists(
    *,
    size: int,
    start: Coordinate,
    target: Coordinate,
    obstacles: set[Coordinate],
) -> bool:
    """Return whether the target is reachable from the start."""

    frontier: deque[Coordinate] = deque([start])
    visited = {start}

    movements = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    while frontier:
        row, column = frontier.popleft()

        if (row, column) == target:
            return True

        for row_delta, column_delta in movements:
            neighbor = (
                row + row_delta,
                column + column_delta,
            )

            neighbor_row, neighbor_column = neighbor

            inside_map = (
                0 <= neighbor_row < size
                and 0 <= neighbor_column < size
            )

            if (
                inside_map
                and neighbor not in obstacles
                and neighbor not in visited
            ):
                visited.add(neighbor)
                frontier.append(neighbor)

    return False


def generate_map_collection(
    *,
    count: int,
    size: int,
    obstacle_probability: float,
    seed: int,
    output_directory: Path,
) -> None:
    """Generate and save a collection of fixed maps."""

    if count <= 0:
        raise ValueError("count must be positive.")

    if size not in {8, 16, 32}:
        raise ValueError(
            "Supported map sizes are 8, 16, and 32."
        )

    if not 0.0 <= obstacle_probability < 1.0:
        raise ValueError(
            "obstacle_probability must be in [0.0, 1.0)."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(seed)

    for map_index in range(1, count + 1):
        map_config = generate_map(
            size=size,
            obstacle_probability=obstacle_probability,
            rng=rng,
        )

        map_config["map_id"] = f"map_{map_index:03d}"
        map_config["generation"] = {
            "collection_seed": seed,
            "map_index": map_index,
        }

        output_path = (
            output_directory
            / f"map_{map_index:03d}.yaml"
        )

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing map: {output_path}"
            )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                map_config,
                file,
                sort_keys=False,
            )

        print(f"Created {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed DDID-Bench maps."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=(8, 16, 32),
        default=8,
    )
    parser.add_argument(
        "--obstacle-probability",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/maps/8x8"),
    )

    arguments = parser.parse_args()

    generate_map_collection(
        count=arguments.count,
        size=arguments.size,
        obstacle_probability=arguments.obstacle_probability,
        seed=arguments.seed,
        output_directory=arguments.output,
    )


if __name__ == "__main__":
    main()