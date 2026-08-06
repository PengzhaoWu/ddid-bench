"""Binary footprint sensor for DDID-Bench.

The sensor converts the hidden simulator state into a noisy binary
observation for one agent.

Policies never access the hidden simulator state or sensor directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ddid.domain.observation import Observation
from ddid.domain.state import State

Coordinate: TypeAlias = tuple[int, int]


class RandomSource(Protocol):
    """Protocol for random number generators."""

    def random(self) -> float:
        """Return a random value in the interval [0.0, 1.0)."""
        ...


def _manhattan_distance(
    first: Coordinate,
    second: Coordinate,
) -> int:
    """Return the Manhattan distance between two grid coordinates."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


@dataclass(frozen=True, slots=True)
class BinaryFootprintSensor:
    """Noisy binary footprint sensor."""

    footprint_radius: int
    p_detection: float
    p_false_alarm: float

    def __post_init__(self) -> None:
        """Validate sensor parameters."""
        if self.footprint_radius < 0:
            raise ValueError("footprint_radius must be non-negative")

        if not 0.0 <= self.p_detection <= 1.0:
            raise ValueError("p_detection must be between 0 and 1")

        if not 0.0 <= self.p_false_alarm <= 1.0:
            raise ValueError("p_false_alarm must be between 0 and 1")

    def observe(
        self,
        state: State,
        agent_id: int,
        rng: RandomSource | None = None,
    ) -> Observation:
        """Generate one noisy binary observation."""

        if not 0 <= agent_id < len(state.agent_poses):
            raise IndexError(f"Invalid agent_id: {agent_id}")

        random_source = rng if rng is not None else random

        agent_position = state.agent_poses[agent_id]

        target_in_footprint = (
            _manhattan_distance(
                agent_position,
                state.target_location,
            )
            <= self.footprint_radius
        )

        probability = (
            self.p_detection
            if target_in_footprint
            else self.p_false_alarm
        )

        detected = (
            random_source.random()
            < probability
        )

        return Observation(
            schema_version="0.1",
            timestep=state.timestep,
            agent_id=agent_id,
            sensor_type="binary_footprint",
            sensor_position=agent_position,
            detected=detected,
        )