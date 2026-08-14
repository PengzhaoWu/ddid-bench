"""Reward model for DDID-Bench.

This module implements the canonical reference reward defined in the
DDID-Bench specification.

The reward penalizes elapsed time, environmental risk exposure,
duplicate coverage, and collisions, while rewarding physical discovery
of the target.
"""

from __future__ import annotations

from dataclasses import dataclass

from ddid.domain.action import Action
from ddid.domain.state import State


@dataclass(frozen=True, slots=True)
class RewardModel:
    """Canonical DDID-Bench reward model.

    The reward is

        r_t =
            - step_cost
            - risk_cost * sum_i rho(p_i,t)
            - duplicate_cost * duplicate_count
            - collision_cost * collision_count
            + find_reward * target_found

    where ``target_found`` equals 1 when any agent physically occupies
    the hidden target location, and 0 otherwise.

    Attributes:
        step_cost:
            Cost incurred for every environment timestep.

        risk_cost:
            Weight applied to total environmental risk exposure across
            all agents.

        duplicate_cost:
            Weight applied to duplicate-coverage events.

        collision_cost:
            Weight applied to agent-collision events.

        find_reward:
            Positive reward for physically finding the target.
    """

    step_cost: float = 1.0
    risk_cost: float = 2.0
    duplicate_cost: float = 0.0
    collision_cost: float = 0.0
    find_reward: float = 100.0

    def __post_init__(self) -> None:
        """Validate reward-model parameters."""

        if self.step_cost < 0.0:
            raise ValueError(
                "step_cost must be non-negative"
            )

        if self.risk_cost < 0.0:
            raise ValueError(
                "risk_cost must be non-negative"
            )

        if self.duplicate_cost < 0.0:
            raise ValueError(
                "duplicate_cost must be non-negative"
            )

        if self.collision_cost < 0.0:
            raise ValueError(
                "collision_cost must be non-negative"
            )

        if self.find_reward < 0.0:
            raise ValueError(
                "find_reward must be non-negative"
            )

    def reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        duplicate_count: int = 0,
        collision_count: int = 0,
    ) -> float:
        """Compute reward for one simulator transition.

        Args:
            state:
                Hidden simulator state before the action.

            action:
                Action executed during this transition.

            next_state:
                Hidden simulator state after the action.

            duplicate_count:
                Number of duplicate-coverage events occurring during
                this timestep.

            collision_count:
                Number of collision events occurring during this
                timestep.

        Returns:
            Scalar reward associated with the transition.
        """

        self._validate_transition(
            state=state,
            action=action,
            next_state=next_state,
            duplicate_count=duplicate_count,
            collision_count=collision_count,
        )

        total_risk = self._total_risk(
            next_state
        )

        target_found = self._target_physically_found(
            next_state
        )

        reward = (
            -self.step_cost
            -self.risk_cost * total_risk
            -self.duplicate_cost * duplicate_count
            -self.collision_cost * collision_count
        )

        if target_found:
            reward += self.find_reward

        return float(reward)

    @staticmethod
    def _total_risk(
        state: State,
    ) -> float:
        """Return total risk exposure across all agents.

        The State coordinate convention is ``(x, y)``, while the
        risk field is indexed as ``risk_field[y][x]``.
        """

        total_risk = 0.0

        for x, y in state.agent_positions:
            total_risk += state.risk_field[y][x]

        return float(total_risk)

    @staticmethod
    def _target_physically_found(
        state: State,
    ) -> bool:
        """Return whether any agent physically occupies the target."""

        return (
            state.target_location
            in state.agent_positions
        )

    @staticmethod
    def _validate_transition(
        *,
        state: State,
        action: Action,
        next_state: State,
        duplicate_count: int,
        collision_count: int,
    ) -> None:
        """Validate reward-model inputs."""

        if not 0 <= action.agent_id < len(
            state.agent_positions
        ):
            raise IndexError(
                f"Invalid agent_id: {action.agent_id}"
            )

        if action.timestep != state.timestep:
            raise ValueError(
                "Action timestep must match state timestep: "
                f"action={action.timestep}, "
                f"state={state.timestep}"
            )

        if next_state.timestep != state.timestep + 1:
            raise ValueError(
                "next_state timestep must equal "
                "state timestep + 1"
            )

        if len(next_state.agent_positions) != len(
            state.agent_positions
        ):
            raise ValueError(
                "The number of agents cannot change "
                "during reward computation."
            )

        if duplicate_count < 0:
            raise ValueError(
                "duplicate_count must be non-negative"
            )

        if collision_count < 0:
            raise ValueError(
                "collision_count must be non-negative"
            )