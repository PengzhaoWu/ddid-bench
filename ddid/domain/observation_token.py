"""Observation-token data contract for DDID-Bench.

An ObservationToken is a standardized unit of information derived from
a raw Observation.

Tokens are consumed by downstream components such as communication,
information distillation, belief updating, experiment logging, and
regression testing.
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
class ObservationToken:
    """Standardized information token derived from an observation.

    Attributes:
        schema_version:
            Version of the token data contract.

        token_id:
            Unique identifier for this token.

        agent_id:
            Identifier of the agent that generated the observation.

        source_id:
            Identifier of the information source, such as a sensor.

        timestep:
            Environment timestep at which the information was generated.

        modality:
            Type of information carried by the token.

        region_ids:
            Spatial regions associated with the information.

        value:
            Actual observation value carried by the token.

        likelihood_id:
            Identifier of the likelihood model used by belief updating.
            The baseline uses "default" until explicit likelihood models
            are implemented.

        bit_cost:
            Communication cost associated with the token.
            The baseline uses 0 until a communication-cost model
            is implemented.

        compute_cost:
            Computational cost associated with processing the token.
            The baseline uses 0.0 until a compute-cost model
            is implemented.

        confidence:
            Confidence associated with the information.
            The baseline uses 1.0 as a placeholder until a confidence
            model is implemented.

        provenance:
            Metadata describing the origin of the token.
    """

    schema_version: str
    token_id: str
    agent_id: str
    source_id: str
    timestep: int
    modality: str
    region_ids: tuple[int, ...]
    value: Any

    # Baseline placeholders. These can be replaced by explicit
    # models/configuration in later DDID-Bench stages.
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

        if not self.agent_id.strip():
            raise ValueError(
                "agent_id must be a non-empty string"
            )

        if not self.source_id.strip():
            raise ValueError(
                "source_id must be a non-empty string"
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
            "agent_id": self.agent_id,
            "source_id": self.source_id,
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