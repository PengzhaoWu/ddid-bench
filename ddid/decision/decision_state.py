"""Decision-state data contract for DDID-Bench.

A DecisionState is the only policy-visible state representation. It is the
output of the information-distillation module and contains exactly the
information required by a policy to select an action.

Policies must never receive the hidden simulator state or a direct reference
to the environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

Coordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DecisionState:
    """Immutable policy-visible decision state."""

    schema_version: str
    agent_id: str
    timestep: int

    target_belief: tuple[float, ...]
    risk_map: tuple[float, ...]
    visited_map: tuple[bool, ...]

    own_pose: Coordinate
    teammate_poses: tuple[Coordinate, ...]

    remaining_horizon: int
    mission_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate fields and freeze caller-owned containers."""
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")

        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")

        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative")

        if self.remaining_horizon < 0:
            raise ValueError("remaining_horizon must be nonnegative")

        object.__setattr__(
            self,
            "target_belief",
            tuple(self.target_belief),
        )

        object.__setattr__(
            self,
            "risk_map",
            tuple(self.risk_map),
        )

        object.__setattr__(
            self,
            "visited_map",
            tuple(self.visited_map),
        )

        object.__setattr__(
            self,
            "teammate_poses",
            tuple(tuple(p) for p in self.teammate_poses),
        )

        object.__setattr__(
            self,
            "mission_weights",
            MappingProxyType(dict(self.mission_weights)),
        )

        self._validate_probability_tuple(
            self.target_belief,
            "target_belief",
        )

        self._validate_probability_tuple(
            self.risk_map,
            "risk_map",
        )

        if len(self.visited_map) != len(self.risk_map):
            raise ValueError(
                "visited_map and risk_map must have the same length"
            )

        self._validate_coordinate(
            self.own_pose,
            "own_pose",
        )

        for pose in self.teammate_poses:
            self._validate_coordinate(
                pose,
                "teammate_poses",
            )

        for key, value in self.mission_weights.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "mission_weights keys must be non-empty strings"
                )

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "mission_weights values must be real numbers"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "mission_weights values must be finite"
                )

    @staticmethod
    def _validate_probability_tuple(
        values: tuple[float, ...],
        field_name: str,
    ) -> None:
        """Validate a tuple of probabilities."""
        for value in values:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{field_name} must contain real numbers"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{field_name} must contain finite values"
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} values must lie in [0, 1]"
                )

    @staticmethod
    def _validate_coordinate(
        coordinate: Coordinate,
        field_name: str,
    ) -> None:
        """Validate a 2-D integer coordinate."""
        if len(coordinate) != 2:
            raise ValueError(
                f"{field_name} must contain exactly two integers"
            )

        x, y = coordinate

        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or not isinstance(y, int)
            or isinstance(y, bool)
        ):
            raise TypeError(
                f"{field_name} must contain integer coordinates"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly dictionary."""
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "timestep": self.timestep,
            "target_belief": list(self.target_belief),
            "risk_map": list(self.risk_map),
            "visited_map": list(self.visited_map),
            "own_pose": list(self.own_pose),
            "teammate_poses": [
                list(pose) for pose in self.teammate_poses
            ],
            "remaining_horizon": self.remaining_horizon,
            "mission_weights": dict(self.mission_weights),
        }