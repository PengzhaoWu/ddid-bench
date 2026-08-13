"""Public-state data contract for DDID-Bench.

A PublicState stores policy-visible mission information derived from the
simulator State together with accumulated public history such as visited
regions.

PublicState must never expose hidden mission variables such as target
locations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

Coordinate: TypeAlias = tuple[int, int]
RiskField: TypeAlias = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class PublicState:
    """Immutable policy-visible public mission state.

    Attributes:
        schema_version:
            Version of the public-state data contract.

        timestep:
            Discrete environment timestep associated with this state.

        agent_poses:
            Current publicly known positions of all agents.

        risk_field:
            Publicly known environmental risk field.

        health_states:
            Publicly known health values associated with the agents.

        visited_map:
            Flattened boolean map indicating which grid regions have been
            visited by at least one agent.
    """

    schema_version: str
    timestep: int

    agent_poses: tuple[Coordinate, ...]
    risk_field: RiskField
    health_states: tuple[float, ...]

    visited_map: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate fields and freeze caller-owned containers."""

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must be a non-empty string"
            )

        if self.timestep < 0:
            raise ValueError(
                "timestep must be nonnegative"
            )

        object.__setattr__(
            self,
            "agent_poses",
            tuple(tuple(pose) for pose in self.agent_poses),
        )

        object.__setattr__(
            self,
            "risk_field",
            tuple(tuple(row) for row in self.risk_field),
        )

        object.__setattr__(
            self,
            "health_states",
            tuple(self.health_states),
        )

        object.__setattr__(
            self,
            "visited_map",
            tuple(self.visited_map),
        )

        for pose in self.agent_poses:
            self._validate_coordinate(pose)

        if len(self.health_states) != len(self.agent_poses):
            raise ValueError(
                "health_states and agent_poses must have the same length"
            )

        for value in self.health_states:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "health_states must contain real numbers"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "health_states must contain finite values"
                )

        if not self.risk_field:
            raise ValueError(
                "risk_field must contain at least one row"
            )

        row_width = len(self.risk_field[0])

        if row_width == 0:
            raise ValueError(
                "risk_field rows must not be empty"
            )

        if any(
            len(row) != row_width
            for row in self.risk_field
        ):
            raise ValueError(
                "risk_field must be rectangular"
            )

        for row in self.risk_field:
            for value in row:
                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    raise TypeError(
                        "risk_field must contain real numbers"
                    )

                if not math.isfinite(float(value)):
                    raise ValueError(
                        "risk_field must contain finite values"
                    )

        expected_regions = len(self.risk_field) * row_width

        if len(self.visited_map) != expected_regions:
            raise ValueError(
                "visited_map length must match the number of grid regions"
            )

        if any(
            not isinstance(value, bool)
            for value in self.visited_map
        ):
            raise TypeError(
                "visited_map must contain only boolean values"
            )

    @staticmethod
    def _validate_coordinate(
        coordinate: Coordinate,
    ) -> None:
        """Validate a 2-D integer coordinate."""

        if len(coordinate) != 2:
            raise ValueError(
                "agent poses must contain exactly two integers"
            )

        x, y = coordinate

        if (
            isinstance(x, bool)
            or not isinstance(x, int)
            or isinstance(y, bool)
            or not isinstance(y, int)
        ):
            raise TypeError(
                "agent poses must contain integer coordinates"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly dictionary."""

        return {
            "schema_version": self.schema_version,
            "timestep": self.timestep,
            "agent_poses": [
                list(pose)
                for pose in self.agent_poses
            ],
            "risk_field": [
                list(row)
                for row in self.risk_field
            ],
            "health_states": list(self.health_states),
            "visited_map": list(self.visited_map),
        }