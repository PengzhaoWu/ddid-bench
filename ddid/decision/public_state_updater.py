"""Public-state update behavior for DDID-Bench.

This module constructs the policy-visible PublicState from the current
simulator State and previously accumulated public information.

The updater may access simulator State, but it must expose only fields
defined as public. Hidden mission variables such as target locations must
never be copied into PublicState.
"""

from __future__ import annotations

from dataclasses import dataclass

from ddid.domain.public_state import PublicState
from ddid.domain.state import State

Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PublicStateUpdater:
    """Update policy-visible public mission information."""

    grid_width: int
    grid_height: int
    schema_version: str = "1.0"

    def initialize(
        self,
        state: State,
    ) -> PublicState:
        """Construct the initial public state."""

        visited_map = [False] * (
            self.grid_width * self.grid_height
        )

        self._mark_visited(
            visited_map,
            state.agent_poses,
        )

        return PublicState(
            schema_version=self.schema_version,
            timestep=state.timestep,
            agent_poses=state.agent_poses,
            risk_field=state.risk_field,
            health_states=state.health_states,
            visited_map=tuple(visited_map),
        )

    def update(
        self,
        previous: PublicState,
        state: State,
    ) -> PublicState:
        """Construct the next public state."""

        visited_map = list(previous.visited_map)

        self._mark_visited(
            visited_map,
            state.agent_poses,
        )

        return PublicState(
            schema_version=previous.schema_version,
            timestep=state.timestep,
            agent_poses=state.agent_poses,
            risk_field=state.risk_field,
            health_states=state.health_states,
            visited_map=tuple(visited_map),
        )

    def _mark_visited(
        self,
        visited_map: list[bool],
        agent_poses: tuple[Coordinate, ...],
    ) -> None:
        """Mark all current agent positions as visited."""

        for pose in agent_poses:
            region_id = self._pose_to_region_id(pose)
            visited_map[region_id] = True

    def _pose_to_region_id(
        self,
        pose: Coordinate,
    ) -> int:
        """Convert a grid coordinate to a flat region index."""

        x, y = pose

        if not 0 <= x < self.grid_width:
            raise ValueError(
                f"x coordinate {x} is outside the grid"
            )

        if not 0 <= y < self.grid_height:
            raise ValueError(
                f"y coordinate {y} is outside the grid"
            )

        return y * self.grid_width + x