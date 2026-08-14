"""Action data contract for DDID-Bench.

This module defines the policy-selected action issued by one agent at one
decision step. An action may include both physical motion and an optional
communication request.
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
    """Policy-selected action for one agent at one decision step.

    Attributes:
        schema_version:
            Version of the action data contract.

        agent_id:
            Identifier of the agent executing the action.

        timestep:
            Environment timestep for which the action is issued.

        motion:
            Requested physical movement.

        communication_request:
            Optional request for communication or information exchange.
            ``None`` means no communication is requested.
    """

    schema_version: str
    agent_id: int
    timestep: int
    motion: Motion
    communication_request: str | None = None

    def __post_init__(self) -> None:
        """Validate action fields."""

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must be a non-empty string"
            )

        if self.agent_id < 0:
            raise ValueError(
                "agent_id must be non-negative"
            )

        if self.timestep < 0:
            raise ValueError(
                "timestep must be non-negative"
            )

        if (
            self.communication_request is not None
            and not self.communication_request.strip()
        ):
            raise ValueError(
                "communication_request must be non-empty or None"
            )