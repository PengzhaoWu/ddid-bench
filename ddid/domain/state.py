"""Hidden mission-state contract for DDID-Bench.

This module implements the simulator-owned ground-truth state described in
Section 4.2 of the DDID-Bench specification:

    s_t = (x_star, p_1:N,t, rho, xi_t, h_1:N,t^health)

The hidden state is separate from observations, beliefs, rewards, and
policy-visible decision states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

Coordinate = tuple[int, int]
AgentPositions = tuple[Coordinate, ...]
RiskField = tuple[tuple[float, ...], ...]
Metadata = Mapping[str, Any]


def _freeze_mapping(value: Mapping[str, Any]) -> Metadata:
    """Return an immutable shallow copy of a metadata mapping."""
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class State:
    """Simulator-owned hidden mission state.

    The fields ``target_location``, ``agent_positions``, ``risk_field``,
    ``environment_conditions``, and ``platform_health`` implement the
    mathematical state in Section 4.2.

    ``schema_version`` and ``provenance`` are software-contract metadata and
    are not part of the mathematical state tuple.
    """

    schema_version: str
    timestep: int
    target_location: Coordinate
    agent_positions: AgentPositions
    risk_field: RiskField
    environment_conditions: Metadata = field(default_factory=dict)
    platform_health: tuple[Metadata, ...] = ()
    provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants and freeze mapping-valued fields."""
        if not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")

        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")

        if not self.agent_positions:
            raise ValueError("agent_positions must contain at least one agent")

        if not self.risk_field or not self.risk_field[0]:
            raise ValueError("risk_field must be a non-empty rectangular grid")

        width = len(self.risk_field[0])
        height = len(self.risk_field)

        for row in self.risk_field:
            if len(row) != width:
                raise ValueError("risk_field must be rectangular")
            if any(not 0.0 <= risk <= 1.0 for risk in row):
                raise ValueError("risk values must lie in [0, 1]")

        self._validate_coordinate(
            self.target_location,
            width=width,
            height=height,
            field_name="target_location",
        )

        for index, position in enumerate(self.agent_positions):
            self._validate_coordinate(
                position,
                width=width,
                height=height,
                field_name=f"agent_positions[{index}]",
            )

        if self.platform_health and len(self.platform_health) != len(
            self.agent_positions
        ):
            raise ValueError(
                "platform_health must be empty or contain one entry per agent"
            )

        object.__setattr__(
            self,
            "environment_conditions",
            _freeze_mapping(self.environment_conditions),
        )
        object.__setattr__(
            self,
            "platform_health",
            tuple(_freeze_mapping(item) for item in self.platform_health),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_mapping(self.provenance),
        )

    @staticmethod
    def _validate_coordinate(
        coordinate: Coordinate,
        *,
        width: int,
        height: int,
        field_name: str,
    ) -> None:
        """Validate that a coordinate lies inside the risk-field grid."""
        x, y = coordinate

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"{field_name}={coordinate!r} is outside grid bounds "
                f"width={width}, height={height}"
            )
