"""
Random Policy Baseline for Taxi City Navigation.
Selects actions uniformly at random from the discrete action space.
"""

from typing import Tuple, Any
import numpy as np
from config import Action


class RandomPolicyAgent:
    """Agent that chooses actions uniformly at random."""

    def __init__(self, num_actions: int = 4, seed: int = 42):
        self.num_actions = num_actions
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: Tuple[int, ...], explore: bool = False) -> int:
        """Select action randomly."""
        return int(self.rng.randint(0, self.num_actions))

    def __call__(self, env: Any, state: Tuple[int, ...]) -> int:
        return self.select_action(state)
