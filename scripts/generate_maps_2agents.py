"""Generate fixed 16x16 two-agent maps for DDID-Bench.

Each generated map contains:
- a 16x16 grid;
- two agent initial positions;
- one static target;
- random obstacles;
- guaranteed reachability from both agents to the target;
- a YAML file and matching PNG visualization.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle

Coordinate = tuple[int, int]


def _path_exists(
    *,
    size: int,
    start: Coordinate,
    target: Coordinate,
    obstacles: set[Coordinate],
) -> bool:
    """Return whether target is reachable from start."""

    frontier: deque[Coordinate] = deque([start])
    visited: set[Coordinate] = {start}

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


def generate_map(
    *,
    size: int,
    obstacle_probability: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Generate one valid two-agent map."""

    agent_0_start: Coordinate = (0, 0)
    agent_1_start: Coordinate = (0, size - 1)

    target_location: Coordinate = (
        size - 2,
        size // 2,
    )

    while True:
        obstacles: set[Coordinate] = set()

        protected_cells = {
            agent_0_start,
            agent_1_start,
            target_location,
        }

        for row in range(size):
            for column in range(size):
                coordinate = (row, column)

                if coordinate in protected_cells:
                    continue

                if rng.random() < obstacle_probability:
                    obstacles.add(coordinate)

        agent_0_reachable = _path_exists(
            size=size,
            start=agent_0_start,
            target=target_location,
            obstacles=obstacles,
        )

        agent_1_reachable = _path_exists(
            size=size,
            start=agent_1_start,
            target=target_location,
            obstacles=obstacles,
        )

        if agent_0_reachable and agent_1_reachable:
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
                list(agent_0_start),
                list(agent_1_start),
            ],

            "initial_health": [
                1.0,
                1.0,
            ],
        },
    }


def render_map(
    map_config: dict[str, Any],
    output_path: Path,
) -> None:
    """Render one map as a PNG."""

    environment = map_config["environment"]
    agents = map_config["agents"]

    rows, columns = environment["grid_size"]

    obstacles = {
        tuple(position)
        for position in environment["obstacles"]
    }

    targets = {
        tuple(position)
        for position in environment["target_locations"]
    }

    agent_positions = {
        tuple(position): agent_id
        for agent_id, position
        in enumerate(agents["initial_poses"])
    }

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for row in range(rows):
        for column in range(columns):
            coordinate = (row, column)

            rectangle = Rectangle(
                (column, row),
                1,
                1,
                fill=False,
                linewidth=0.8,
            )

            ax.add_patch(rectangle)

            center_x = column + 0.5
            center_y = row + 0.5

            if coordinate in obstacles:
                ax.text(
                    center_x,
                    center_y,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            elif coordinate in targets:
                ax.text(
                    center_x,
                    center_y,
                    "T",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

            elif coordinate in agent_positions:
                agent_id = agent_positions[coordinate]

                ax.text(
                    center_x,
                    center_y,
                    f"A{agent_id}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xlim(0, columns)
    ax.set_ylim(rows, 0)

    ax.set_xticks(
        [column + 0.5 for column in range(columns)]
    )
    ax.set_xticklabels(range(columns))

    ax.set_yticks(
        [row + 0.5 for row in range(rows)]
    )
    ax.set_yticklabels(range(rows))

    ax.xaxis.tick_top()

    ax.set_xlabel("Column")
    ax.xaxis.set_label_position("top")

    ax.set_ylabel("Row")

    ax.set_aspect("equal")

    ax.set_title(
        map_config.get(
            "map_id",
            "DDID-Bench 16x16 Map",
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_map_collection(
    *,
    count: int,
    size: int,
    obstacle_probability: float,
    seed: int,
    output_directory: Path,
) -> None:
    """Generate and save fixed maps."""

    if count <= 0:
        raise ValueError(
            "count must be positive."
        )

    if size != 16:
        raise ValueError(
            "This script is intended for 16x16 maps."
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

    for map_index in range(
        1,
        count + 1,
    ):
        map_config = generate_map(
            size=size,
            obstacle_probability=obstacle_probability,
            rng=rng,
        )

        map_config["map_id"] = (
            f"map_{map_index:03d}"
        )

        map_config["generation"] = {
            "collection_seed": seed,
            "map_index": map_index,
            "agent_count": 2,
            "target_count": 1,
        }

        yaml_path = (
            output_directory
            / f"map_{map_index:03d}.yaml"
        )

        png_path = (
            output_directory
            / f"map_{map_index:03d}.png"
        )

        if yaml_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing map: "
                f"{yaml_path}"
            )

        if png_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing image: "
                f"{png_path}"
            )

        with yaml_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                map_config,
                file,
                sort_keys=False,
            )

        render_map(
            map_config=map_config,
            output_path=png_path,
        )

        print(f"Created {yaml_path}")
        print(f"Created {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed 16x16 DDID-Bench maps "
            "with two agents and one target."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
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
        default=Path(
            "configs/maps/16x16_two_agents"
        ),
    )

    arguments = parser.parse_args()

    generate_map_collection(
        count=arguments.count,
        size=16,
        obstacle_probability=arguments.obstacle_probability,
        seed=arguments.seed,
        output_directory=arguments.output,
    )


if __name__ == "__main__":
    main()