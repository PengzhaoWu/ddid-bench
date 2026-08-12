"""Communication graph for DDID-Bench.

This module defines the communication topology between agents.

The topology is loaded from a DDID-Bench YAML configuration. The graph
determines whether an ObservationToken may be transmitted from one agent
to another and creates CommunicationToken objects for valid transmissions.

The communication layer must never access the hidden simulator State.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ddid.communication.communication_token import CommunicationToken
from ddid.domain.observation_token import ObservationToken


@dataclass(frozen=True, slots=True)
class CommunicationGraph:
    """Static directed communication topology between agents.

    Each edge ``(sender_id, receiver_id)`` means that the sender may
    transmit information directly to the receiver.
    """

    edges: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        """Validate communication edges."""

        for sender_id, receiver_id in self.edges:
            if not sender_id.strip():
                raise ValueError(
                    "sender_id must be a non-empty string"
                )

            if not receiver_id.strip():
                raise ValueError(
                    "receiver_id must be a non-empty string"
                )

            if sender_id == receiver_id:
                raise ValueError(
                    "communication edges cannot connect "
                    "an agent to itself"
                )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
    ) -> CommunicationGraph:
        """Create a communication graph from a YAML configuration."""

        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        raw_edges = config["communication"]["edges"]

        edges = frozenset(
            (str(sender_id), str(receiver_id))
            for sender_id, receiver_id in raw_edges
        )

        return cls(edges=edges)

    def can_communicate(
        self,
        sender_id: str,
        receiver_id: str,
    ) -> bool:
        """Return whether direct communication is allowed."""

        return (
            sender_id,
            receiver_id,
        ) in self.edges

    def neighbors(
        self,
        sender_id: str,
    ) -> tuple[str, ...]:
        """Return agents that can directly receive from sender."""

        return tuple(
            receiver_id
            for edge_sender, receiver_id in sorted(self.edges)
            if edge_sender == sender_id
        )

    def transmit(
        self,
        token: ObservationToken,
        receiver_id: str,
        *,
        timestep: int | None = None,
    ) -> CommunicationToken:
        """Create a communication token for a valid transmission."""

        sender_id = token.agent_id

        if not self.can_communicate(
            sender_id,
            receiver_id,
        ):
            raise ValueError(
                f"Communication from agent {sender_id!r} "
                f"to agent {receiver_id!r} is not allowed"
            )

        communication_timestep = (
            token.timestep
            if timestep is None
            else timestep
        )

        return CommunicationToken(
            schema_version="0.1",
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestep=communication_timestep,
            observation_token=token,
        )

    def broadcast(
        self,
        token: ObservationToken,
        *,
        timestep: int | None = None,
    ) -> tuple[CommunicationToken, ...]:
        """Transmit a token to every directly reachable neighbor."""

        return tuple(
            self.transmit(
                token,
                receiver_id,
                timestep=timestep,
            )
            for receiver_id in self.neighbors(token.agent_id)
        )