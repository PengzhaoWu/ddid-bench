"""Grid-world transition model for DDID-Bench.

This module defines how the simulator-owned hidden State evolves after
an agent executes a physical motion action.

The transition model operates on the complete hidden State but does not
expose that state to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from ddid.domain.action import Action, Motion
from ddid.domain.state import State

Coordinate = tuple[int, int]


_MOTION_DELTAS: dict[Motion, Coordinate] = {
    Motion.UP: (0, -1),
    Motion.DOWN: (0, 1),
    Motion.LEFT: (-1, 0),
    Motion.RIGHT: (1, 0),
    Motion.STAY: (0, 0),
}


@dataclass(frozen=True, slots=True)
class GridTransitionModel:
    """Deterministic physical transition model for the grid world.

    Attributes:
        obstacles:
            Grid cells that agents may not enter.

    Notes:
        Grid dimensions are inferred from ``State.risk_field``.

        Invalid moves, including moves outside the grid or into an
        obstacle, are treated as no-op transitions.

        Communication requests contained in ``Action`` are not handled
        here. They belong to the communication subsystem.
    """

    obstacles: frozenset[Coordinate] = frozenset()

    def transition(
        self,
        state: State,
        action: Action,
    ) -> State:
        """Apply one action and return the next hidden state.

        Args:
            state:
                Current simulator-owned hidden state.

            action:
                Action selected by an agent.

        Returns:
            Hidden simulator state at the next timestep.
        """

        self._validate_action(state, action)

        current_position = state.agent_positions[action.agent_id]

        candidate_position = self._candidate_position(
            current_position,
            action.motion,
        )

        if self._is_valid_position(
            candidate_position,
            state,
        ):
            next_position = candidate_position
        else:
            next_position = current_position

        next_agent_positions = list(state.agent_positions)
        next_agent_positions[action.agent_id] = next_position

        return State(
            schema_version=state.schema_version,
            timestep=state.timestep + 1,
            target_location=state.target_location,
            agent_positions=tuple(next_agent_positions),
            risk_field=state.risk_field,
            environment_conditions=state.environment_conditions,
            platform_health=state.platform_health,
            provenance=state.provenance,
        )

    @staticmethod
    def _validate_action(
        state: State,
        action: Action,
    ) -> None:
        """Validate that an action is consistent with the current state."""

        if not 0 <= action.agent_id < len(state.agent_positions):
            raise IndexError(
                f"Invalid agent_id: {action.agent_id}"
            )

        if action.timestep != state.timestep:
            raise ValueError(
                "action timestep must match state timestep: "
                f"action={action.timestep}, "
                f"state={state.timestep}"
            )

    @staticmethod
    def _candidate_position(
        position: Coordinate,
        motion: Motion,
    ) -> Coordinate:
        """Return the position requested by a motion command."""

        x, y = position
        dx, dy = _MOTION_DELTAS[motion]

        return (
            x + dx,
            y + dy,
        )

    def _is_valid_position(
        self,
        position: Coordinate,
        state: State,
    ) -> bool:
        """Return whether an agent may occupy a position."""

        return (
            self._in_bounds(position, state)
            and position not in self.obstacles
        )

    @staticmethod
    def _in_bounds(
        position: Coordinate,
        state: State,
    ) -> bool:
        """Return whether a position lies inside the grid."""

        x, y = position

        height = len(state.risk_field)
        width = len(state.risk_field[0])

        return (
            0 <= x < width
            and 0 <= y < height
        )