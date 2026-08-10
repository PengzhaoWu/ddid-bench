"""Observation-token data contract for DDID-Bench.

An ObservationToken is the immutable, versioned representation of one
tokenized sensor observation. Raw environment observations should be
converted into this contract before being routed to belief-update modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservationToken:
    """Immutable tokenized observation produced by a sensor source.

    Attributes:
        schema_version:
            Version of the observation-token contract.
        token_id:
            Identifier that must be unique within an episode or manifest.
        agent_id:
            Identifier of the agent that received the observation.
        source_id:
            Identifier of the sensor or source that produced the observation.
        timestep:
            Discrete timestep at which the observation was generated.
        modality:
            Observation modality, such as ``"binary"``, ``"camera"``,
            ``"rf"``, or ``"thermal"``.
        region_ids:
            Region identifiers affected or described by the observation.
        value:
            Observation payload. Its expected type is determined by
            ``modality`` and ``schema_version``.
        likelihood_id:
            Identifier of the likelihood model used for belief updating.
        bit_cost:
            Communication cost of the token in bits.
        compute_cost:
            Computational cost associated with processing the token.
        confidence:
            Confidence value in the closed interval ``[0.0, 1.0]``.
        provenance:
            Immutable metadata describing how the token was produced.
    """

    schema_version: str
    token_id: str
    agent_id: str
    source_id: str
    timestep: int
    modality: str
    region_ids: tuple[int, ...]
    value: Any
    likelihood_id: str
    bit_cost: int
    compute_cost: float
    confidence: float
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate fields and freeze mutable input containers."""
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")

        if not self.token_id:
            raise ValueError("token_id must be a non-empty string")

        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")

        if not self.source_id:
            raise ValueError("source_id must be a non-empty string")

        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative")

        if not self.modality:
            raise ValueError("modality must be a non-empty string")

        if any(
            not isinstance(region_id, int) or isinstance(region_id, bool)
            for region_id in self.region_ids
        ):
            raise TypeError("every region_id must be an integer")

        if any(region_id < 0 for region_id in self.region_ids):
            raise ValueError("region_ids must be nonnegative")

        if len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError("region_ids must not contain duplicates")

        if not self.likelihood_id:
            raise ValueError("likelihood_id must be a non-empty string")

        if (
            not isinstance(self.bit_cost, int)
            or isinstance(self.bit_cost, bool)
        ):
            raise TypeError("bit_cost must be an integer")

        if self.bit_cost < 0:
            raise ValueError("bit_cost must be nonnegative")

        if self.compute_cost < 0.0:
            raise ValueError("compute_cost must be nonnegative")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Convert caller-owned containers into immutable representations.
        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
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