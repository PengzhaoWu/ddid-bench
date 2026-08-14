"""Grid-world environment orchestration for DDID-Bench.

This module loads a fixed benchmark instance from YAML, initializes the
hidden simulator state, and coordinates transition and reward models.

GridWorld owns the hidden simulator State. Policies must never receive
the hidden State directly.

Sensing remains external: a Sensor observes the hidden State returned
by GridWorld.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml

from ddid.domain.action import Action
from ddid.domain.state import Coordinate, State


class TransitionModel(Protocol):
    """Required interface for a transition model."""

    def transition(
        self,
        state: State,
        action: Action,
    ) -> State:
        """Apply an action and return the next hidden state."""
        ...


class RewardModel(Protocol):
    """Required interface for a reward model."""

    def reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        duplicate_count: int = 0,
        collision_count: int = 0,
    ) -> float:
        """Compute reward for one state transition."""
        ...


class GridWorld:
    """DDID-Bench grid-world environment.

    The environment loads one fixed YAML benchmark instance.

    Calling ``reset()`` restores the initial state from that instance.
    The map itself is never regenerated or modified during an episode.

    GridWorld coordinates:

        Action
            -> TransitionModel
            -> next State
            -> event detection
            -> RewardModel
            -> termination

    Sensing, belief updates, communication, and policy decisions are
    intentionally outside this class.
    """

    def __init__(
        self,
        *,
        instance_path: str | Path,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        max_steps: int = 100,
    ) -> None:
        """Initialize the grid-world environment."""

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        self._instance_path = Path(instance_path)
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._max_steps = max_steps

        self._config = self._load_yaml(
            self._instance_path
        )

        self._state: State | None = None

        self._terminated = False
        self._truncated = False

        # Simulator-side physical visitation history.
        #
        # This is used only for the current baseline definition of
        # duplicate physical coverage. It is not automatically exposed
        # to policies.
        self._visited_cells: set[Coordinate] = set()

    @property
    def state(self) -> State:
        """Return the current hidden simulator state.

        This property is intended for environment and testing code.
        Policies must never receive the hidden state directly.
        """

        if self._state is None:
            raise RuntimeError(
                "The environment has not been reset. "
                "Call reset() first."
            )

        return self._state

    @property
    def instance_path(self) -> Path:
        """Return the benchmark-instance path."""

        return self._instance_path

    @property
    def map_id(self) -> str:
        """Return the map identifier."""

        return str(
            self._config.get(
                "map_id",
                self._instance_path.stem,
            )
        )

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return grid size as (rows, columns)."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )

        raw_size = environment.get(
            "grid_size"
        )

        if (
            not isinstance(
                raw_size,
                (list, tuple),
            )
            or len(raw_size) != 2
        ):
            raise ValueError(
                "environment.grid_size must contain "
                "exactly two values."
            )

        rows = int(raw_size[0])
        columns = int(raw_size[1])

        if rows <= 0 or columns <= 0:
            raise ValueError(
                "Grid dimensions must be positive."
            )

        return rows, columns

    @property
    def obstacles(self) -> frozenset[Coordinate]:
        """Return all fixed obstacle coordinates."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )

        raw_obstacles = environment.get(
            "obstacles",
            [],
        )

        return frozenset(
            self._parse_coordinate(value)
            for value in raw_obstacles
        )

    def reset(
        self,
    ) -> tuple[State, Mapping[str, Any]]:
        """Reset the episode to the fixed initial state.

        Returns:
            state:
                Initial hidden simulator state.

            info:
                Simulator-side metadata. This is not policy input.
        """

        self._state = self._build_initial_state()

        self._terminated = False
        self._truncated = False

        self._visited_cells = set(
            self._state.agent_positions
        )

        info: Mapping[str, Any] = {
            "map_id": self.map_id,
            "timestep": self._state.timestep,
            "target_found": self._is_terminated(
                self._state
            ),
        }

        return self._state, info

    def step(
        self,
        action: Action,
    ) -> tuple[
        State,
        float,
        bool,
        bool,
        Mapping[str, Any],
    ]:
        """Advance the hidden simulator state by one timestep.

        GridWorld performs:

        1. Action validation.
        2. State transition.
        3. Transition-output validation.
        4. Duplicate/collision event detection.
        5. Reward computation.
        6. Termination/truncation checks.

        Sensing is performed separately by the Sensor.

        Returns:
            next_state:
                Hidden simulator state after the action.

            reward:
                Scalar reward associated with the transition.

            terminated:
                Whether the mission termination condition was reached.

            truncated:
                Whether the episode horizon was reached.

            info:
                Simulator metadata that is not policy input.
        """

        current_state = self.state

        if self._terminated or self._truncated:
            raise RuntimeError(
                "The episode has ended. "
                "Call reset() before step()."
            )

        self._validate_action(
            current_state,
            action,
        )

        next_state = self._transition_model.transition(
            state=current_state,
            action=action,
        )

        self._validate_next_state(
            current_state=current_state,
            next_state=next_state,
        )

        duplicate_count = self._duplicate_count(
            action=action,
            next_state=next_state,
        )

        collision_count = self._collision_count(
            next_state
        )

        reward = self._reward_model.reward(
            state=current_state,
            action=action,
            next_state=next_state,
            duplicate_count=duplicate_count,
            collision_count=collision_count,
        )

        self._update_visited_cells(
            next_state
        )

        self._terminated = self._is_terminated(
            next_state
        )

        self._truncated = (
            not self._terminated
            and next_state.timestep >= self._max_steps
        )

        self._state = next_state

        info: Mapping[str, Any] = {
            "map_id": self.map_id,
            "timestep": next_state.timestep,
            "target_found": self._terminated,
            "time_limit_reached": self._truncated,
            "duplicate_count": duplicate_count,
            "collision_count": collision_count,
        }

        return (
            next_state,
            float(reward),
            self._terminated,
            self._truncated,
            info,
        )

    def render(self) -> str:
        """Return an evaluator/debug ASCII representation.

        This rendering exposes the hidden target location and therefore
        must never be used as policy input.
        """

        state = self.state

        rows, columns = self.grid_size

        agent_positions = {
            position: agent_id
            for agent_id, position in enumerate(
                state.agent_positions
            )
        }

        obstacles = self.obstacles

        column_width = max(
            1,
            len(str(columns - 1)),
        )

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
                coordinate = (
                    column,
                    row,
                )

                if coordinate in agent_positions:
                    agent_id = agent_positions[
                        coordinate
                    ]

                    if len(state.agent_positions) == 1:
                        symbol = "A"
                    else:
                        symbol = str(agent_id)

                elif coordinate == state.target_location:
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
        """Construct the initial hidden State from YAML."""

        environment = self._require_mapping(
            self._config,
            "environment",
        )

        agents = self._require_mapping(
            self._config,
            "agents",
        )

        target_location = self._parse_target_location(
            environment
        )

        agent_positions = tuple(
            self._parse_coordinate(value)
            for value in agents.get(
                "initial_poses",
                [],
            )
        )

        if not agent_positions:
            raise ValueError(
                "The benchmark instance must contain "
                "at least one agent."
            )

        risk_field = self._parse_risk_field(
            environment.get(
                "risk_field"
            )
        )

        environment_conditions = environment.get(
            "initial_exogenous_state",
            {},
        )

        if not isinstance(
            environment_conditions,
            Mapping,
        ):
            raise TypeError(
                "initial_exogenous_state "
                "must be a mapping."
            )

        raw_health = agents.get(
            "initial_health",
            [],
        )

        platform_health = self._parse_platform_health(
            raw_health,
            agent_count=len(agent_positions),
        )

        return State(
            schema_version=str(
                self._config.get(
                    "schema_version",
                    "1.0",
                )
            ),
            timestep=0,
            target_location=target_location,
            agent_positions=agent_positions,
            risk_field=risk_field,
            environment_conditions=dict(
                environment_conditions
            ),
            platform_health=platform_health,
            provenance={
                "map_id": self.map_id,
                "instance_path": str(
                    self._instance_path
                ),
            },
        )

    def _parse_target_location(
        self,
        environment: Mapping[str, Any],
    ) -> Coordinate:
        """Parse the single hidden target location.

        Existing benchmark YAML files may use ``target_locations``.
        The current State contract contains one ``target_location``,
        so exactly one target is required.
        """

        if "target_location" in environment:
            return self._parse_coordinate(
                environment["target_location"]
            )

        raw_targets = environment.get(
            "target_locations",
            [],
        )

        if not isinstance(
            raw_targets,
            (list, tuple),
        ):
            raise TypeError(
                "environment.target_locations "
                "must be a sequence."
            )

        if len(raw_targets) != 1:
            raise ValueError(
                "The current State contract supports "
                "exactly one target location."
            )

        return self._parse_coordinate(
            raw_targets[0]
        )

    def _parse_platform_health(
        self,
        value: Any,
        *,
        agent_count: int,
    ) -> tuple[Mapping[str, Any], ...]:
        """Convert YAML health data to State.platform_health."""

        if value in (None, []):
            return ()

        if not isinstance(
            value,
            (list, tuple),
        ):
            raise TypeError(
                "agents.initial_health must be a sequence."
            )

        if len(value) != agent_count:
            raise ValueError(
                "initial_health must contain one "
                "value per agent."
            )

        return tuple(
            {
                "health": float(health)
            }
            for health in value
        )

    def _parse_risk_field(
        self,
        value: Any,
    ) -> tuple[tuple[float, ...], ...]:
        """Parse and validate the static risk field."""

        rows, columns = self.grid_size

        if value is None:
            return tuple(
                tuple(
                    0.0
                    for _ in range(columns)
                )
                for _ in range(rows)
            )

        if not isinstance(
            value,
            (list, tuple),
        ):
            raise TypeError(
                "environment.risk_field must be "
                "a two-dimensional sequence."
            )

        risk_field = tuple(
            tuple(
                float(cell)
                for cell in row
            )
            for row in value
        )

        if len(risk_field) != rows:
            raise ValueError(
                "Risk-field row count must match "
                "the grid size."
            )

        if any(
            len(row) != columns
            for row in risk_field
        ):
            raise ValueError(
                "Risk-field column count must match "
                "the grid size."
            )

        return risk_field

    @staticmethod
    def _validate_action(
        state: State,
        action: Action,
    ) -> None:
        """Validate an Action against the current State."""

        if not 0 <= action.agent_id < len(
            state.agent_positions
        ):
            raise IndexError(
                f"Invalid agent_id: {action.agent_id}"
            )

        if action.timestep != state.timestep:
            raise ValueError(
                "Action timestep does not match "
                "the current state: "
                f"action={action.timestep}, "
                f"state={state.timestep}"
            )

    def _validate_next_state(
        self,
        *,
        current_state: State,
        next_state: State,
    ) -> None:
        """Validate output produced by the transition model."""

        expected_timestep = (
            current_state.timestep + 1
        )

        if next_state.timestep != expected_timestep:
            raise ValueError(
                "The transition model must increment "
                "timestep by one. "
                f"Expected {expected_timestep}, "
                f"received {next_state.timestep}."
            )

        if len(next_state.agent_positions) != len(
            current_state.agent_positions
        ):
            raise ValueError(
                "The transition model cannot change "
                "the number of agents."
            )

        if (
            next_state.target_location
            != current_state.target_location
        ):
            raise ValueError(
                "The transition model changed the "
                "static target location."
            )

        if next_state.risk_field != current_state.risk_field:
            raise ValueError(
                "The transition model changed the "
                "static risk field."
            )

        for coordinate in next_state.agent_positions:
            if not self._is_inside_map(
                coordinate
            ):
                raise ValueError(
                    "The transition model produced "
                    "an out-of-bounds position: "
                    f"{coordinate}."
                )

            if coordinate in self.obstacles:
                raise ValueError(
                    "The transition model placed "
                    "an agent on an obstacle: "
                    f"{coordinate}."
                )

    def _duplicate_count(
        self,
        *,
        action: Action,
        next_state: State,
    ) -> int:
        """Return duplicate physical-coverage count for this step.

        Current baseline definition:

        A duplicate event occurs when the acting agent ends the step
        on a cell that had already been physically visited earlier in
        the episode.

        This helper can later be replaced if the canonical definition
        uses sensor-footprint coverage rather than physical visitation.
        """

        position = next_state.agent_positions[
            action.agent_id
        ]

        return int(
            position in self._visited_cells
        )

    @staticmethod
    def _collision_count(
        state: State,
    ) -> int:
        """Return the number of agent-position collisions.

        A collision is detected when two or more agents occupy the
        same cell in the resulting state.
        """

        positions = state.agent_positions

        return len(positions) - len(
            set(positions)
        )

    def _update_visited_cells(
        self,
        state: State,
    ) -> None:
        """Add current agent positions to visitation history."""

        self._visited_cells.update(
            state.agent_positions
        )

    def _is_terminated(
        self,
        state: State,
    ) -> bool:
        """Return whether the target is physically found."""

        return (
            state.target_location
            in state.agent_positions
        )

    def _is_inside_map(
        self,
        coordinate: Coordinate,
    ) -> bool:
        """Return whether a coordinate lies inside the grid.

        Coordinate convention is ``(x, y)``.
        """

        rows, columns = self.grid_size

        x, y = coordinate

        return (
            0 <= x < columns
            and 0 <= y < rows
        )

    @staticmethod
    def _parse_coordinate(
        value: Any,
    ) -> Coordinate:
        """Parse one ``(x, y)`` coordinate."""

        if (
            not isinstance(
                value,
                (list, tuple),
            )
            or len(value) != 2
        ):
            raise ValueError(
                f"Invalid coordinate: {value!r}"
            )

        return (
            int(value[0]),
            int(value[1]),
        )

    @staticmethod
    def _require_mapping(
        config: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        """Return a required YAML mapping section."""

        value = config.get(key)

        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"Configuration section "
                f"{key!r} must be a mapping."
            )

        return value

    @staticmethod
    def _load_yaml(
        path: Path,
    ) -> Mapping[str, Any]:
        """Load a benchmark YAML configuration."""

        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark instance not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(
                file
            )

        if not isinstance(
            config,
            Mapping,
        ):
            raise TypeError(
                "The YAML root must be a mapping."
            )

        return config