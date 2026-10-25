"""
Tabular Q-Learning Agent implemented from scratch using NumPy.

Update Rule:
Q(s, a) <- Q(s, a) + alpha * [ Reward + gamma * max_a'( Q(s', a') ) - Q(s, a) ]
"""

from typing import Tuple, Dict, Any, Optional
import pickle
import numpy as np
from config import Action, QLearningConfig


class QLearningAgent:
    """
    Model-free Temporal Difference (TD) Q-Learning Agent.
    Maintains a tabular state-action value function Q(s, a).
    """

    def __init__(self, num_actions: int = 4, config: Optional[QLearningConfig] = None, seed: int = 42):
        self.config = config or QLearningConfig()
        self.num_actions = num_actions
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.rng = np.random.RandomState(seed)

        # Tabular Q-map: state_tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _encode_state(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        State Abstraction:
        - If passenger not picked (state[6] == 0), the destination does not affect navigation to passenger.
        - If passenger is picked (state[6] == 1), original passenger position does not affect navigation to destination.
        This collapses redundant states, enabling rapid convergence and superior generalization.
        """
        if len(state) >= 7:
            if state[6] == 0:
                # Active target: Passenger at (state[2], state[3])
                return (state[0], state[1], state[2], state[3], -1, -1, 0)
            else:
                # Active target: Destination at (state[4], state[5])
                return (state[0], state[1], -1, -1, state[4], state[5], 1)
        return state

    def get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Retrieve or lazily initialize Q-values for a given discrete state."""
        key = self._encode_state(state)
        if key not in self.q_table:
            # Initialize with zeros
            self.q_table[key] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[key]

    def select_action(self, state: Tuple[int, ...], explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        - With probability epsilon (if explore=True): choose random action.
        - Otherwise: choose action with highest Q-value, breaking ties randomly.
        """
        if explore and self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, self.num_actions))

        q_vals = self.get_q_values(state)
        max_q = np.max(q_vals)
        # Find all actions matching the maximum Q-value to break ties randomly
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(self, state: Tuple[int, ...], action: int, reward: float,
               next_state: Tuple[int, ...], done: bool) -> float:
        """
        Perform standard Bellman optimality Q-value update:
        Q(s, a) <- Q(s, a) + alpha * [ R + gamma * max_a'(Q(s', a')) - Q(s, a) ]

        Returns:
            td_error: The Temporal Difference error [Target - Q(s, a)]
        """
        key = self._encode_state(state)
        current_q = self.get_q_values(state)[action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * next_max_q

        td_error = target - current_q
        self.q_table[key][action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay epsilon after an episode towards epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str):
        """Save learned Q-table and hyperparameters to disk."""
        payload = {
            "q_table": self.q_table,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "num_actions": self.num_actions,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def load(self, filepath: str):
        """Load Q-table and state from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.q_table = payload["q_table"]
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.num_actions = payload.get("num_actions", self.num_actions)

    @property
    def table_size(self) -> int:
        """Return the number of explored discrete states in the Q-table."""
        return len(self.q_table)
