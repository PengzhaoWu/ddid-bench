"""Communication-token data contract for DDID-Bench.

A CommunicationToken represents the transmission of one ObservationToken
from a sender agent to a receiver agent.

The original ObservationToken is preserved unchanged. Communication-specific
metadata is stored separately in this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ddid.domain.observation_token import ObservationToken


@dataclass(frozen=True, slots=True)
class CommunicationToken:
    """One communicated observation token.

    Attributes:
        schema_version:
            Version of the communication-token contract.

        sender_id:
            Identifier of the agent sending the token.

        receiver_id:
            Identifier of the agent receiving the token.

        timestep:
            Environment timestep at which the communication occurs.

        observation_token:
            Original ObservationToken being transmitted.
    """

    schema_version: str
    sender_id: str
    receiver_id: str
    timestep: int
    observation_token: ObservationToken

    def __post_init__(self) -> None:
        """Validate communication-token fields."""

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must be a non-empty string"
            )

        if not self.sender_id.strip():
            raise ValueError(
                "sender_id must be a non-empty string"
            )

        if not self.receiver_id.strip():
            raise ValueError(
                "receiver_id must be a non-empty string"
            )

        if self.sender_id == self.receiver_id:
            raise ValueError(
                "sender_id and receiver_id must be different"
            )

        if self.timestep < 0:
            raise ValueError(
                "timestep must be non-negative"
            )

        if self.sender_id != self.observation_token.agent_id:
            raise ValueError(
                "sender_id must match observation_token.agent_id"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly dictionary representation."""

        return {
            "schema_version": self.schema_version,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestep": self.timestep,
            "observation_token": self.observation_token.to_dict(),
        }