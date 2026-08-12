"""Action data contract for DDID-Bench.

This module defines the policy-selected action issued at one decision step.
An action may include both physical motion and an optional communication
request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Motion(str, Enum):
    """Supported grid-world motion commands."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


@dataclass(frozen=True, slots=True)
class Action:
    """Policy-selected action for one decision step.

    Attributes:
        motion:
            Requested physical movement.

        communication_request:
            Optional request for communication or information exchange.
            ``None`` means no communication is requested.
    """

    motion: Motion
    communication_request: str | None = None

    def __post_init__(self) -> None:
        """Validate action fields."""

        if (
            self.communication_request is not None
            and not self.communication_request.strip()
        ):
            raise ValueError(
                "communication_request must be non-empty or None"
            )