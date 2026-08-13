"""Information-distillation behavior for DDID-Bench.

The distiller consumes locally generated ObservationToken objects and
CommunicationToken objects received from other agents and produces
DistilledToken objects for downstream belief updating.

This module defines behavior only. The DistilledToken data contract is
defined separately in distilled_token.py.

The distillation layer must never access the hidden simulator State.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

from ddid.information.distilled_token import DistilledToken

from ddid.communication.communication_token import CommunicationToken
from ddid.domain.observation_token import ObservationToken


@dataclass(frozen=True, slots=True)
class Distiller:
    """Baseline information distiller.

    The baseline implementation preserves all available information.

    Local observation tokens are converted directly into distilled tokens.
    Information carried by communication tokens is also converted into
    distilled tokens while preserving source-agent and source-token
    provenance.

    More advanced DDID strategies can later replace this baseline behavior
    with filtering, fusion, compression, relevance scoring, or
    decision-driven selection.
    """

    schema_version: str = "1.0"

    def distill(
        self,
        observation_tokens: Iterable[ObservationToken],
        communication_tokens: Iterable[CommunicationToken],
    ) -> tuple[DistilledToken, ...]:
        """Distill all information currently available to an agent.

        Args:
            observation_tokens:
                Locally generated observation tokens.

            communication_tokens:
                Information received through the communication layer.

        Returns:
            Immutable sequence of distilled tokens.
        """
        distilled_tokens: list[DistilledToken] = []

        for token in observation_tokens:
            distilled_tokens.append(
                self._from_observation_token(token)
            )

        for token in communication_tokens:
            distilled_tokens.append(
                self._from_communication_token(token)
            )

        return tuple(distilled_tokens)

    def _from_observation_token(
        self,
        token: ObservationToken,
    ) -> DistilledToken:
        """Convert a local observation token into distilled information."""
        return DistilledToken(
            schema_version=self.schema_version,
            token_id=self._make_token_id(),
            source_token_ids=(token.token_id,),
            source_agent_ids=(token.agent_id,),
            timestep=token.timestep,
            modality=token.modality,
            region_ids=token.region_ids,
            value=token.value,
            likelihood_id=token.likelihood_id,
            bit_cost=token.bit_cost,
            compute_cost=token.compute_cost,
            confidence=token.confidence,
            provenance={
                **dict(token.provenance),
                "distillation_source": "observation",
            },
        )

    def _from_communication_token(
        self,
        token: CommunicationToken,
    ) -> DistilledToken:
        """Convert communicated information into distilled information."""
        observation_token = token.observation_token

        return DistilledToken(
            schema_version=self.schema_version,
            token_id=self._make_token_id(),
            source_token_ids=(observation_token.token_id,),
            source_agent_ids=(observation_token.agent_id,),
            timestep=observation_token.timestep,
            modality=observation_token.modality,
            region_ids=observation_token.region_ids,
            value=observation_token.value,
            likelihood_id=observation_token.likelihood_id,
            bit_cost=observation_token.bit_cost,
            compute_cost=observation_token.compute_cost,
            confidence=observation_token.confidence,
            provenance={
                **dict(observation_token.provenance),
                "distillation_source": "communication",
                "sender_id": token.sender_id,
                "receiver_id": token.receiver_id,
            },
        )

    @staticmethod
    def _make_token_id() -> str:
        """Generate a unique distilled-token identifier."""
        return f"distilled-{uuid4().hex}"