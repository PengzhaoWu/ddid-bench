"""Grid-world environment orchestration for DDID-Bench.

This module loads a fixed benchmark instance from YAML, initializes the hidden
simulator state, and coordinates the transition, reward, and sensor modules.

GridWorld does not implement transition rules, reward formulas, or sensor
noise directly.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml

from ddid.domain.action import Action
from ddid.domain.observation import Observation
from ddid.domain.state import Coordinate, State


class TransitionModel(Protocol):
    """Required interface for a transition model."""

    def apply(
        self,
        state: State,
        action: Action,
    ) -> State:
        """Apply an action and return the next hidden state."""
        ...


class RewardModel(Protocol):
    """Required interface for a reward model."""

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> float:
        """Compute the reward for one state transition."""
        ...


class SensorModel(Protocol):
    """Required interface for a sensor model."""

    def observe(
        self,
        state: State,
        agent_id: int,
        rng: random.Random | None = None,
    ) -> Observation:
        """Generate an observation for one agent."""
        ...


class GridWorld:
    """DDID-Bench grid-world environment.

    The environment loads one fixed YAML benchmark instance. Calling reset()
    restores the initial state from that instance but does not regenerate or
    modify the map.
    """

    def __init__(
        self,
        *,
        instance_path: str | Path,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        sensor: SensorModel,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self._instance_path = Path(instance_path)
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._sensor = sensor
        self._max_steps = max_steps
        self._rng = random.Random(seed)

        self._config = self._load_yaml(self._instance_path)

        self._state: State | None = None
        self._terminated = False
        self._truncated = False

    @property
    def state(self) -> State:
        """Return the current hidden simulator state.

        This property is intended for the simulator and testing code. Policies
        should never receive the hidden state directly.
        """

        if self._state is None:
            raise RuntimeError(
                "The environment has not been reset. Call reset() first."
            )

        return self._state

    @property
    def instance_path(self) -> Path:
        """Return the benchmark-instance path."""

        return self._instance_path

    @property
    def map_id(self) -> str:
        """Return the map identifier stored in the YAML file."""

        return str(
            self._config.get(
                "map_id",
                self._instance_path.stem,
            )
        )

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return the configured grid size as (rows, columns)."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )

        raw_size = environment.get("grid_size")

        if (
            not isinstance(raw_size, list | tuple)
            or len(raw_size) != 2
        ):
            raise ValueError(
                "environment.grid_size must contain exactly two values."
            )

        rows = int(raw_size[0])
        columns = int(raw_size[1])

        return rows, columns

    @property
    def obstacles(self) -> frozenset[Coordinate]:
        """Return all fixed obstacle coordinates."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )

        raw_obstacles = environment.get("obstacles", [])

        return frozenset(
            self._parse_coordinate(value)
            for value in raw_obstacles
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        agent_id: int = 0,
    ) -> tuple[Observation, Mapping[str, Any]]:
        """Reset the episode using the unchanged YAML benchmark instance."""

        if seed is not None:
            self._rng.seed(seed)

        self._state = self._build_initial_state()
        self._terminated = False
        self._truncated = False

        self._validate_agent_id(
            state=self._state,
            agent_id=agent_id,
        )

        observation = self._sensor.observe(
            state=self._state,
            agent_id=agent_id,
            rng=self._rng,
        )

        info: Mapping[str, Any] = {
            "map_id": self.map_id,
            "timestep": self._state.timestep,
            "agent_id": agent_id,
        }

        return observation, info

    def step(
        self,
        action: Action,
        *,
        agent_id: int = 0,
    ) -> tuple[
        Observation,
        float,
        bool,
        bool,
        Mapping[str, Any],
    ]:
        """Advance the environment by one timestep.

        Returns:
            observation:
                Noisy observation generated from the next hidden state.
            reward:
                Reward associated with the transition.
            terminated:
                Whether the mission termination condition was reached.
            truncated:
                Whether the episode reached the step limit.
            info:
                Simulator metadata that is not policy input.
        """

        current_state = self.state

        if self._terminated or self._truncated:
            raise RuntimeError(
                "The episode has ended. Call reset() before step()."
            )

        self._validate_agent_id(
            state=current_state,
            agent_id=agent_id,
        )

        next_state = self._transition_model.apply(
            state=current_state,
            action=action,
        )

        self._validate_next_state(
            current_state=current_state,
            next_state=next_state,
        )

        reward = self._reward_model.compute(
            state=current_state,
            action=action,
            next_state=next_state,
        )

        observation = self._sensor.observe(
            state=next_state,
            agent_id=agent_id,
            rng=self._rng,
        )

        self._terminated = self._is_terminated(next_state)
        self._truncated = (
            not self._terminated
            and next_state.timestep >= self._max_steps
        )

        self._state = next_state

        info: Mapping[str, Any] = {
            "map_id": self.map_id,
            "timestep": next_state.timestep,
            "agent_id": agent_id,
            "target_found": self._terminated,
            "time_limit_reached": self._truncated,
        }

        return (
            observation,
            float(reward),
            self._terminated,
            self._truncated,
            info,
        )

    def render(self) -> str:
        """Return an ASCII representation of the current map."""

        state = self.state
        rows, columns = self.grid_size

        agent_positions = {
            coordinate: agent_id
            for agent_id, coordinate in enumerate(state.agent_poses)
        }

        target_locations = set(state.target_locations)
        obstacles = self.obstacles

        column_width = max(1, len(str(columns - 1)))

        header = (
            " " * (column_width + 2)
            + " ".join(
                f"{column:>{column_width}}"
                for column in range(columns)
            )
        )

        rendered_rows = [header]

        for row in range(rows):
            symbols: list[str] = []

            for column in range(columns):
                coordinate = (row, column)

                if coordinate in agent_positions:
                    agent_id = agent_positions[coordinate]

                    if len(state.agent_poses) == 1:
                        symbol = "A"
                    else:
                        symbol = str(agent_id)

                elif coordinate in target_locations:
                    symbol = "T"

                elif coordinate in obstacles:
                    symbol = "#"

                else:
                    symbol = "."

                symbols.append(
                    f"{symbol:>{column_width}}"
                )

            rendered_rows.append(
                f"{row:>{column_width}}  "
                + " ".join(symbols)
            )

        return "\n".join(rendered_rows)

    def _build_initial_state(self) -> State:
        """Construct the initial hidden state from the loaded YAML."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )
        agents = self._require_mapping(
            self._config,
            "agents",
        )

        target_locations = tuple(
            self._parse_coordinate(value)
            for value in environment.get(
                "target_locations",
                [],
            )
        )

        if not target_locations:
            raise ValueError(
                "The benchmark instance must contain at least one target."
            )

        agent_poses = tuple(
            self._parse_coordinate(value)
            for value in agents.get(
                "initial_poses",
                [],
            )
        )

        if not agent_poses:
            raise ValueError(
                "The benchmark instance must contain at least one agent."
            )

        agent_health = tuple(
            float(value)
            for value in agents.get(
                "initial_health",
                [1.0] * len(agent_poses),
            )
        )

        if len(agent_health) != len(agent_poses):
            raise ValueError(
                "initial_health must contain one value per agent."
            )

        risk_field = self._parse_risk_field(
            environment.get("risk_field")
        )

        exogenous_state = environment.get(
            "initial_exogenous_state",
            {},
        )

        if not isinstance(exogenous_state, Mapping):
            raise TypeError(
                "initial_exogenous_state must be a mapping."
            )

        return State(
            schema_version=str(
                self._config.get(
                    "schema_version",
                    "1.0",
                )
            ),
            timestep=0,
            target_locations=target_locations,
            agent_poses=agent_poses,
            risk_field=risk_field,
            xi_t=dict(exogenous_state),
            agent_health=agent_health,
            metadata={
                "map_id": self.map_id,
                "instance_path": str(self._instance_path),
                "obstacles": tuple(sorted(self.obstacles)),
                "generation": dict(
                    self._config.get(
                        "generation",
                        {},
                    )
                ),
            },
        )

    def _parse_risk_field(
        self,
        value: Any,
    ) -> tuple[tuple[float, ...], ...]:
        """Parse and validate the static risk field."""

        rows, columns = self.grid_size

        if value is None:
            return tuple(
                tuple(0.0 for _ in range(columns))
                for _ in range(rows)
            )

        if not isinstance(value, list | tuple):
            raise TypeError(
                "environment.risk_field must be a two-dimensional sequence."
            )

        risk_field = tuple(
            tuple(float(cell) for cell in row)
            for row in value
        )

        if len(risk_field) != rows:
            raise ValueError(
                "Risk-field row count must match the grid size."
            )

        if any(len(row) != columns for row in risk_field):
            raise ValueError(
                "Risk-field column count must match the grid size."
            )

        return risk_field

    def _is_terminated(
        self,
        state: State,
    ) -> bool:
        """Return whether an agent has reached a target."""

        target_locations = set(state.target_locations)

        return any(
            agent_position in target_locations
            for agent_position in state.agent_poses
        )

    def _validate_next_state(
        self,
        *,
        current_state: State,
        next_state: State,
    ) -> None:
        """Validate the output produced by the transition model."""

        expected_timestep = current_state.timestep + 1

        if next_state.timestep != expected_timestep:
            raise ValueError(
                "The transition model must increment timestep by one. "
                f"Expected {expected_timestep}, "
                f"received {next_state.timestep}."
            )

        if len(next_state.agent_poses) != len(
            current_state.agent_poses
        ):
            raise ValueError(
                "The transition model cannot change the number of agents."
            )

        for coordinate in next_state.agent_poses:
            if not self._is_inside_map(coordinate):
                raise ValueError(
                    "The transition model produced an out-of-bounds "
                    f"position: {coordinate}."
                )

            if coordinate in self.obstacles:
                raise ValueError(
                    "The transition model placed an agent on an obstacle: "
                    f"{coordinate}."
                )

    def _is_inside_map(
        self,
        coordinate: Coordinate,
    ) -> bool:
        rows, columns = self.grid_size
        row, column = coordinate

        return (
            0 <= row < rows
            and 0 <= column < columns
        )

    @staticmethod
    def _validate_agent_id(
        *,
        state: State,
        agent_id: int,
    ) -> None:
        if not 0 <= agent_id < len(state.agent_poses):
            raise IndexError(
                f"Invalid agent_id: {agent_id}"
            )

    @staticmethod
    def _parse_coordinate(
        value: Any,
    ) -> Coordinate:
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
        ):
            raise ValueError(
                f"Invalid coordinate: {value!r}"
            )

        return int(value[0]), int(value[1])

    @staticmethod
    def _require_mapping(
        config: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        value = config.get(key)

        if not isinstance(value, Mapping):
            raise TypeError(
                f"Configuration section {key!r} must be a mapping."
            )

        return value

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> Mapping[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark instance not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, Mapping):
            raise TypeError(
                "The YAML root must be a mapping."
            )

        return config