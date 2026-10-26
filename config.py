"""
Configuration parameters for the Taxi Route Optimization RL environment and agent.
All magic numbers and hyperparameter settings are centralized here.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class Action(IntEnum):
    """Discrete action space for the Taxi Agent."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # Optional WAIT action (stays in place, useful when dynamic traffic fluctuates)
    WAIT = 4


class TrafficLevel(IntEnum):
    """Traffic congestion levels and their corresponding penalty weights."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class RewardConfig:
    """Configurable reward structure for the environment."""
    step_penalty: float = -1.0           # Baseline cost per step to encourage shortest path
    boundary_penalty: float = -5.0       # Penalty for attempting to move off the grid
    obstacle_penalty: float = -5.0       # Penalty for attempting to hit a wall/obstacle
    traffic_low_penalty: float = 0.0     # Extra penalty for moving through LOW traffic
    traffic_med_penalty: float = -3.0    # Extra penalty for moving through MEDIUM traffic (total step cost = -4.0)
    traffic_high_penalty: float = -8.0   # Extra penalty for moving through HIGH traffic (total step cost = -9.0)
    pickup_reward: float = 20.0          # Reward upon picking up passenger
    delivery_reward: float = 100.0       # Reward upon successfully reaching destination with passenger
    timeout_penalty: float = -10.0       # Penalty when episode hits max steps without delivery


@dataclass
class EnvConfig:
    """Environment dimensions and behavioral settings."""
    grid_size: Tuple[int, int] = (10, 10) # (rows, cols)
    max_steps: int = 100                 # Maximum steps per episode before truncation
    obstacle_density: float = 0.15       # 15% of cells are obstacles
    traffic_density: float = 0.25        # 25% of open cells have traffic variations
    dynamic_traffic: bool = False        # If True, traffic shifts dynamically during episode
    traffic_change_prob: float = 0.05    # Probability of traffic shifting per step
    use_stations: bool = True            # Use designated city transit hubs for manageable state space
    num_stations: int = 4                # Number of transit hubs (e.g., Airport, Downtown, Station, Suburb)
    seed: int = 42                       # Reproducibility seed


@dataclass
class QLearningConfig:
    """Hyperparameters for tabular Q-Learning."""
    alpha: float = 0.2                   # Learning rate (step size)
    gamma: float = 0.98                  # Discount factor for future rewards
    epsilon: float = 1.0                 # Initial exploration rate
    epsilon_min: float = 0.05            # Minimum exploration rate
    epsilon_decay: float = 0.997         # Multiplicative decay per episode
    num_episodes: int = 1500             # Default training episodes
