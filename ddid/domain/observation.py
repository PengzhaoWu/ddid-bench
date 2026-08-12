"""Observation contract for DDID-Bench.

An Observation contains raw sensor output generated for one agent
at one environment timestep.

Different sensor types may produce different observation values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

Coordinate: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Observation:
    """Raw observation produced by one environment sensor."""

    schema_version: str
    timestep: int
    agent_id: int
    sensor_type: str
    sensor_position: Coordinate
    value: Any

    def __post_init__(self) -> None:
        """Validate common observation fields."""

        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")

        if self.agent_id < 0:
            raise ValueError("agent_id must be non-negative")

        if not self.sensor_type:
            raise ValueError("sensor_type must not be empty")

        if len(self.sensor_position) != 2:
            raise ValueError(
                "sensor_position must contain exactly two coordinates"
            )