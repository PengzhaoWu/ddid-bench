"""Belief-state data contract for DDID-Bench.

A BeliefState stores one agent's probabilistic estimate of hidden mission
variables after incorporating the available observation tokens.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Immutable probabilistic belief maintained by one agent.

    Attributes:
        schema_version:
            Version of the belief-state contract.
        agent_id:
            Identifier of the agent that owns this belief.
        timestep:
            Discrete timestep associated with the belief.
        target_probs:
            Probability distribution over candidate target regions.
        risk_probs:
            Estimated risk probabilities over the environment regions.
        peer_trust:
            Trust values associated with peer agents.
        environment_params:
            Estimated scalar environment parameters.
        evidence_token_ids:
            Observation-token identifiers used to construct this belief.
        normalization_error:
            Absolute numerical error remaining after normalization.
    """

    schema_version: str
    agent_id: str
    timestep: int
    target_probs: tuple[float, ...]
    risk_probs: tuple[float, ...]
    peer_trust: tuple[float, ...]
    environment_params: Mapping[str, float]
    evidence_token_ids: tuple[str, ...]
    normalization_error: float

    def __post_init__(self) -> None:
        """Validate values and freeze caller-owned containers."""
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")

        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")

        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative")

        object.__setattr__(self, "target_probs", tuple(self.target_probs))
        object.__setattr__(self, "risk_probs", tuple(self.risk_probs))
        object.__setattr__(self, "peer_trust", tuple(self.peer_trust))
        object.__setattr__(
            self,
            "environment_params",
            MappingProxyType(dict(self.environment_params)),
        )
        object.__setattr__(
            self,
            "evidence_token_ids",
            tuple(self.evidence_token_ids),
        )

        self._validate_probability_tuple(
            self.target_probs,
            field_name="target_probs",
        )
        self._validate_probability_tuple(
            self.risk_probs,
            field_name="risk_probs",
        )
        self._validate_probability_tuple(
            self.peer_trust,
            field_name="peer_trust",
        )

        for name, value in self.environment_params.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "environment_params keys must be non-empty strings"
                )

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "environment_params values must be real numbers"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "environment_params values must be finite"
                )

        if any(
            not isinstance(token_id, str) or not token_id
            for token_id in self.evidence_token_ids
        ):
            raise ValueError(
                "evidence_token_ids must contain non-empty strings"
            )

        if len(set(self.evidence_token_ids)) != len(
            self.evidence_token_ids
        ):
            raise ValueError(
                "evidence_token_ids must not contain duplicates"
            )

        if isinstance(self.normalization_error, bool) or not isinstance(
            self.normalization_error,
            (int, float),
        ):
            raise TypeError("normalization_error must be a real number")

        if not math.isfinite(float(self.normalization_error)):
            raise ValueError("normalization_error must be finite")

        if self.normalization_error < 0.0:
            raise ValueError("normalization_error must be nonnegative")

    @staticmethod
    def _validate_probability_tuple(
        values: tuple[float, ...],
        *,
        field_name: str,
    ) -> None:
        """Validate a tuple whose entries represent probabilities."""
        for value in values:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{field_name} must contain only real numbers"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{field_name} must contain only finite values"
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} values must be between 0.0 and 1.0"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "timestep": self.timestep,
            "target_probs": list(self.target_probs),
            "risk_probs": list(self.risk_probs),
            "peer_trust": list(self.peer_trust),
            "environment_params": dict(self.environment_params),
            "evidence_token_ids": list(self.evidence_token_ids),
            "normalization_error": self.normalization_error,
        }