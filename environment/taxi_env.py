"""
Custom Taxi City Environment built from scratch.
Implements Markov Decision Process (MDP) interfaces: reset(), step(), render().
Handles movement, passenger pickup, destination delivery, traffic penalties, and rewards.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from config import Action, EnvConfig, RewardConfig, TrafficLevel
from environment.grid import CityGrid
from environment.traffic import TrafficManager


class TaxiCityEnv:
    """
    Custom MDP Environment for Taxi Route Optimization.
    State representation: (taxi_r, taxi_c, pass_r, pass_c, dest_r, dest_c, passenger_picked)
    Action space: UP (0), DOWN (1), LEFT (2), RIGHT (3), WAIT (4)
    """

    def __init__(self, env_config: Optional[EnvConfig] = None, reward_config: Optional[RewardConfig] = None):
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()

        self.grid = CityGrid(
            rows=self.env_config.grid_size[0],
            cols=self.env_config.grid_size[1],
            obstacle_density=self.env_config.obstacle_density,
            seed=self.env_config.seed,
            num_stations=getattr(self.env_config, "num_stations", 4)
        )

        self.traffic_mgr = TrafficManager(
            shape=self.env_config.grid_size,
            traffic_density=self.env_config.traffic_density,
            seed=self.env_config.seed
        )

        # Agent & Task State
        self.taxi_pos: Tuple[int, int] = (0, 0)
        self.passenger_pos: Tuple[int, int] = (0, 0)
        self.dest_pos: Tuple[int, int] = (0, 0)
        self.passenger_picked: int = 0  # 0: not picked, 1: picked / in taxi
        self.steps_taken: int = 0
        self.total_episode_reward: float = 0.0

        # Action deltas: (dr, dc)
        self.action_deltas = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
            Action.WAIT: (0, 0),
        }

    def reset(self, fixed_endpoints: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
              reset_traffic: bool = True) -> Tuple[int, ...]:
        """
        Reset the environment for a new episode.
        Optionally accepts fixed (taxi_start, passenger, destination) for deterministic evaluation.
        If reset_traffic is False, existing traffic grid is strictly preserved.
        """
        self.steps_taken = 0
        self.total_episode_reward = 0.0
        self.passenger_picked = 0

        # Reset traffic only if requested
        if reset_traffic:
            self.traffic_mgr.reset(self.grid.obstacles)

        if fixed_endpoints is not None:
            self.taxi_pos, self.passenger_pos, self.dest_pos = fixed_endpoints
        else:
            use_stations = getattr(self.env_config, "use_stations", False)
            self.taxi_pos, self.passenger_pos, self.dest_pos = self.grid.sample_reachable_endpoints(use_stations=use_stations)

        return self.get_state()

    def get_state(self) -> Tuple[int, ...]:
        """
        Return discrete state tuple:
        (taxi_row, taxi_col, pass_row, pass_col, dest_row, dest_col, passenger_picked)
        """
        return (
            self.taxi_pos[0],
            self.taxi_pos[1],
            self.passenger_pos[0],
            self.passenger_pos[1],
            self.dest_pos[0],
            self.dest_pos[1],
            self.passenger_picked
        )

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.
        Returns:
            next_state (tuple): S'
            reward (float): R
            terminated (bool): True if destination reached
            truncated (bool): True if step limit reached
            info (dict): Auxiliary diagnostic information
        """
        self.steps_taken += 1
        act = Action(action)
        dr, dc = self.action_deltas[act]

        next_r = self.taxi_pos[0] + dr
        next_c = self.taxi_pos[1] + dc

        reward = 0.0
        terminated = False
        truncated = False
        event = "normal_move"

        # Check 1: Boundary Violation
        if not self.grid.is_valid_coord(next_r, next_c):
            reward += self.reward_config.boundary_penalty
            event = "hit_boundary"
            # Taxi stays at current position

        # Check 2: Obstacle Collision
        elif self.grid.obstacles[next_r, next_c]:
            reward += self.reward_config.obstacle_penalty
            event = "hit_obstacle"
            # Taxi stays at current position

        # Valid Road Transition or WAIT
        else:
            self.taxi_pos = (next_r, next_c)
            reward += self.reward_config.step_penalty

            # Traffic penalty
            traffic_cost = self.traffic_mgr.get_traffic_penalty(next_r, next_c, self.reward_config)
            reward += traffic_cost

            # Check 3: Passenger Pickup (if at passenger location and not yet picked)
            if not self.passenger_picked and self.taxi_pos == self.passenger_pos:
                self.passenger_picked = 1
                reward += self.reward_config.pickup_reward
                event = "passenger_pickup"

            # Check 4: Destination Dropoff (if at destination and passenger is on board)
            if self.passenger_picked and self.taxi_pos == self.dest_pos:
                terminated = True
                reward += self.reward_config.delivery_reward
                event = "delivery_success"

        # Check Step Truncation (Timeout)
        if self.steps_taken >= self.env_config.max_steps and not terminated:
            truncated = True
            reward += self.reward_config.timeout_penalty
            event = "timeout"

        # Dynamic traffic evolution if enabled
        if self.env_config.dynamic_traffic and not (terminated or truncated):
            self.traffic_mgr.step_dynamic(
                obstacle_mask=self.grid.obstacles,
                change_prob=self.env_config.traffic_change_prob
            )

        self.total_episode_reward += reward

        info = {
            "steps": self.steps_taken,
            "event": event,
            "taxi_pos": self.taxi_pos,
            "passenger_picked": self.passenger_picked,
            "traffic_level": self.traffic_mgr.get_traffic_level(*self.taxi_pos)
        }

        return self.get_state(), reward, terminated, truncated, info

    def render(self) -> str:
        """Render the city grid with current state."""
        return self.grid.render_ascii(
            taxi=self.taxi_pos,
            passenger=self.passenger_pos,
            destination=self.dest_pos,
            passenger_picked=bool(self.passenger_picked),
            traffic_grid=self.traffic_mgr.traffic_grid
        )
