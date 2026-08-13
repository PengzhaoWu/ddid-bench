"""Distilled-token data contract for DDID-Bench.

A DistilledToken is a standardized unit of information produced by the
information-distillation layer.

Distilled tokens are derived from local ObservationToken objects and/or
received CommunicationToken objects and are consumed by downstream
components such as belief updating, decision-state assembly, experiment
logging, and regression testing.

This module defines the data contract only. Distillation behavior belongs
in distiller.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

Metadata = Mapping[str, Any]


def _freeze_mapping(value: Mapping[str, Any]) -> Metadata:
    """Return an immutable shallow copy of a mapping."""
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class DistilledToken:
    """Standardized information token produced by distillation.

    Attributes:
        schema_version:
            Version of the distilled-token data contract.

        token_id:
            Unique identifier for this distilled token.

        source_token_ids:
            Identifiers of the input tokens from which this token was
            produced.

        source_agent_ids:
            Identifiers of agents that originally generated the information.

        timestep:
            Environment timestep associated with the distilled information.

        modality:
            Type of information carried by the token.

        region_ids:
            Spatial regions associated with the information.

        value:
            Distilled information value.

        likelihood_id:
            Identifier of the likelihood model used by belief updating.

        bit_cost:
            Communication cost associated with the distilled information.

        compute_cost:
            Computational cost associated with producing or processing
            the distilled information.

        confidence:
            Confidence associated with the distilled information.

        provenance:
            Metadata describing the origin and transformation history
            of the token.
    """

    schema_version: str
    token_id: str

    source_token_ids: tuple[str, ...]
    source_agent_ids: tuple[str, ...]

    timestep: int
    modality: str
    region_ids: tuple[int, ...]
    value: Any

    likelihood_id: str = "default"
    bit_cost: int = 0
    compute_cost: float = 0.0
    confidence: float = 1.0
    provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate token invariants and freeze metadata."""

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must be a non-empty string"
            )

        if not self.token_id.strip():
            raise ValueError(
                "token_id must be a non-empty string"
            )

        if not self.source_token_ids:
            raise ValueError(
                "source_token_ids must contain at least one token identifier"
            )

        if any(
            not token_id.strip()
            for token_id in self.source_token_ids
        ):
            raise ValueError(
                "source_token_ids must contain non-empty strings"
            )

        if not self.source_agent_ids:
            raise ValueError(
                "source_agent_ids must contain at least one agent identifier"
            )

        if any(
            not agent_id.strip()
            for agent_id in self.source_agent_ids
        ):
            raise ValueError(
                "source_agent_ids must contain non-empty strings"
            )

        if self.timestep < 0:
            raise ValueError(
                "timestep must be non-negative"
            )

        if not self.modality.strip():
            raise ValueError(
                "modality must be a non-empty string"
            )

        if any(region_id < 0 for region_id in self.region_ids):
            raise ValueError(
                "region_ids must contain non-negative integers"
            )

        if not self.likelihood_id.strip():
            raise ValueError(
                "likelihood_id must be a non-empty string"
            )

        if self.bit_cost < 0:
            raise ValueError(
                "bit_cost must be non-negative"
            )

        if self.compute_cost < 0.0:
            raise ValueError(
                "compute_cost must be non-negative"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must lie in [0, 1]"
            )

        object.__setattr__(
            self,
            "provenance",
            _freeze_mapping(self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly dictionary representation."""

        return {
            "schema_version": self.schema_version,
            "token_id": self.token_id,
            "source_token_ids": list(self.source_token_ids),
            "source_agent_ids": list(self.source_agent_ids),
            "timestep": self.timestep,
            "modality": self.modality,
            "region_ids": list(self.region_ids),
            "value": self.value,
            "likelihood_id": self.likelihood_id,
            "bit_cost": self.bit_cost,
            "compute_cost": self.compute_cost,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
        }