"""Action data contract for DDID-Bench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """A policy-selected action for one decision step."""

    motion: str
    goal_region: int | None = None
    sensor_mode: str | None = None
    communication_request: str | None = None
    urgency: str = "normal"