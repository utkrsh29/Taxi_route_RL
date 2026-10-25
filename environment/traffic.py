"""
Traffic management module for the dynamic grid city.
Handles traffic intensity grids and dynamic traffic condition shifts.
"""

import numpy as np
from typing import Tuple
from config import TrafficLevel, RewardConfig


class TrafficManager:
    """Manages traffic conditions across the city grid."""

    def __init__(self, shape: Tuple[int, int], traffic_density: float = 0.25, seed: int = 42):
        self.rows, self.cols = shape
        self.traffic_density = traffic_density
        self.rng = np.random.RandomState(seed)
        # 2D array storing TrafficLevel values (0=LOW, 1=MEDIUM, 2=HIGH)
        self.traffic_grid = np.zeros((self.rows, self.cols), dtype=np.int32)

    def reset(self, obstacle_mask: np.ndarray = None):
        """Randomly generate traffic levels on non-obstacle road cells."""
        self.traffic_grid.fill(TrafficLevel.LOW)
        for r in range(self.rows):
            for c in range(self.cols):
                if obstacle_mask is not None and obstacle_mask[r, c]:
                    continue
                # Assign traffic levels based on traffic density
                roll = self.rng.rand()
                if roll < self.traffic_density * 0.4:
                    self.traffic_grid[r, c] = TrafficLevel.HIGH
                elif roll < self.traffic_density:
                    self.traffic_grid[r, c] = TrafficLevel.MEDIUM
                else:
                    self.traffic_grid[r, c] = TrafficLevel.LOW

    def step_dynamic(self, obstacle_mask: np.ndarray = None, change_prob: float = 0.05):
        """Simulate dynamic traffic flow shifts over time."""
        for r in range(self.rows):
            for c in range(self.cols):
                if obstacle_mask is not None and obstacle_mask[r, c]:
                    continue
                if self.rng.rand() < change_prob:
                    # Randomly fluctuate traffic level by +/- 1
                    current = self.traffic_grid[r, c]
                    delta = self.rng.choice([-1, 1])
                    new_val = int(np.clip(current + delta, TrafficLevel.LOW, TrafficLevel.HIGH))
                    self.traffic_grid[r, c] = new_val

    def get_traffic_level(self, row: int, col: int) -> int:
        """Return the traffic level at given coordinates."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.traffic_grid[row, col]
        return TrafficLevel.LOW

    def get_traffic_penalty(self, row: int, col: int, reward_config: RewardConfig) -> float:
        """Compute the additional penalty associated with entering a traffic zone."""
        lvl = self.get_traffic_level(row, col)
        if lvl == TrafficLevel.HIGH:
            return reward_config.traffic_high_penalty
        elif lvl == TrafficLevel.MEDIUM:
            return reward_config.traffic_med_penalty
        return reward_config.traffic_low_penalty
