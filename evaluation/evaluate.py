"""
Evaluation module: Benchmark trained Q-Learning agent against baseline policies
(Random Policy and Shortest-Path BFS baseline) across identical test scenarios.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config import EnvConfig, RewardConfig
from environment.taxi_env import TaxiCityEnv
from baselines.random_policy import RandomPolicyAgent
from baselines.shortest_path import ShortestPathAgent
from agent.q_learning import QLearningAgent


class Evaluator:
    """Runs rigorous multi-episode comparative benchmarks across agents."""

    def __init__(self, env: Optional[TaxiCityEnv] = None, env_config: Optional[EnvConfig] = None):
        self.env_config = env_config or EnvConfig()
        self.env = env or TaxiCityEnv(env_config=self.env_config)

    def generate_test_scenarios(self, num_scenarios: int = 50, seed: int = 999) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Generate deterministic evaluation scenarios (taxi, passenger, destination)."""
        rng = np.random.RandomState(seed)
        scenarios = []
        for _ in range(num_scenarios):
            endpoints = self.env.grid.sample_reachable_endpoints(use_stations=getattr(self.env_config, "use_stations", False))
            scenarios.append(endpoints)
        return scenarios

    def evaluate_policy(self, agent: Any, policy_name: str,
                        scenarios: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                        max_steps: int = 100) -> Dict[str, Any]:
        """
        Evaluate a single agent/policy on fixed scenarios.
        No exploration (greedy execution for RL).
        """
        rewards: List[float] = []
        steps_list: List[int] = []
        success_list: List[int] = []

        for endpoints in scenarios:
            state = self.env.reset(fixed_endpoints=endpoints)
            total_r = 0.0
            steps = 0
            done = False

            while not done and steps < max_steps:
                if hasattr(agent, "select_action"):
                    # Check if method supports env parameter
                    try:
                        action = agent.select_action(state, explore=False, env=self.env)
                    except TypeError:
                        action = agent.select_action(state, explore=False)
                elif callable(agent):
                    action = agent(self.env, state)
                else:
                    action = 0

                next_state, reward, term, trunc, info = self.env.step(action)
                done = term or trunc
                total_r += reward
                steps += 1
                state = next_state

            delivered = 1 if info.get("event") == "delivery_success" else 0
            rewards.append(total_r)
            steps_list.append(steps)
            success_list.append(delivered)

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_steps = float(np.mean(steps_list))
        std_steps = float(np.std(steps_list))
        success_rate = float(np.mean(success_list) * 100.0)

        return {
            "policy": policy_name,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_steps": mean_steps,
            "std_steps": std_steps,
            "success_rate": success_rate,
            "episodes_tested": len(scenarios)
        }

    def benchmark_all(self, q_agent: QLearningAgent,
                      num_scenarios: int = 50,
                      seed: int = 999) -> pd.DataFrame:
        """
        Benchmark:
        1. Random Policy
        2. Shortest-Path Baseline
        3. Q-Learning Agent
        on the exact same test scenarios.
        """
        scenarios = self.generate_test_scenarios(num_scenarios=num_scenarios, seed=seed)

        random_agent = RandomPolicyAgent(seed=seed)
        shortest_path_agent = ShortestPathAgent()

        print(f"\nEvaluating policies on {len(scenarios)} identical test scenarios...")

        results = []
        results.append(self.evaluate_policy(random_agent, "Random Policy", scenarios))
        results.append(self.evaluate_policy(shortest_path_agent, "Shortest-Path Baseline", scenarios))
        results.append(self.evaluate_policy(q_agent, "Q-Learning (Ours)", scenarios))

        df_comparison = pd.DataFrame(results)
        return df_comparison
