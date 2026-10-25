"""
Training pipeline for Tabular Q-Learning in the Taxi Route Optimization Environment.
Executes training loops, logs performance metrics, computes moving averages,
and periodically evaluates greedy policy performance.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config import EnvConfig, RewardConfig, QLearningConfig
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent


class Trainer:
    """Orchestrates agent training, metrics collection, and evaluation checkpoints."""

    def __init__(self,
                 env: Optional[TaxiCityEnv] = None,
                 agent: Optional[QLearningAgent] = None,
                 env_config: Optional[EnvConfig] = None,
                 reward_config: Optional[RewardConfig] = None,
                 agent_config: Optional[QLearningConfig] = None):
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.agent_config = agent_config or QLearningConfig()

        self.env = env or TaxiCityEnv(env_config=self.env_config, reward_config=self.reward_config)
        self.agent = agent or QLearningAgent(num_actions=4, config=self.agent_config, seed=self.env_config.seed)

        # Historical metrics tracker
        self.metrics: Dict[str, List[Any]] = {
            "episode": [],
            "reward": [],
            "steps": [],
            "delivered": [],
            "epsilon": [],
            "avg_td_error": [],
            "q_table_size": [],
        }

    def train(self, num_episodes: Optional[int] = None,
              eval_interval: int = 100,
              verbose: bool = True) -> pd.DataFrame:
        """
        Execute full Q-learning training loop across specified episodes.
        """
        total_episodes = num_episodes or self.agent_config.num_episodes
        start_time = time.time()

        if verbose:
            print("==================================================")
            print(f"STARTING TRAINING: {total_episodes} Episodes")
            print(f"Grid: {self.env_config.grid_size[0]}x{self.env_config.grid_size[1]} | "
                  f"Alpha: {self.agent.alpha} | Gamma: {self.agent.gamma} | "
                  f"Dynamic Traffic: {self.env_config.dynamic_traffic}")
            print("==================================================")

        for ep in range(1, total_episodes + 1):
            state = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            td_errors = []
            done = False

            while not done:
                # Epsilon-greedy action selection
                action = self.agent.select_action(state, explore=True)

                # Environment transition
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Bellman Q-Learning Update
                td_error = self.agent.update(state, action, reward, next_state, done)
                td_errors.append(abs(td_error))

                # Advance state
                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Episode completed: decay epsilon
            current_eps = self.agent.decay_epsilon()
            delivered = 1 if info.get("event") == "delivery_success" else 0
            avg_td = float(np.mean(td_errors)) if td_errors else 0.0

            # Log metrics
            self.metrics["episode"].append(ep)
            self.metrics["reward"].append(episode_reward)
            self.metrics["steps"].append(episode_steps)
            self.metrics["delivered"].append(delivered)
            self.metrics["epsilon"].append(current_eps)
            self.metrics["avg_td_error"].append(avg_td)
            self.metrics["q_table_size"].append(self.agent.table_size)

            # Periodic log output
            if verbose and (ep % eval_interval == 0 or ep == total_episodes):
                recent_rewards = self.metrics["reward"][-eval_interval:]
                recent_steps = self.metrics["steps"][-eval_interval:]
                recent_success = self.metrics["delivered"][-eval_interval:]
                mean_r = np.mean(recent_rewards)
                mean_s = np.mean(recent_steps)
                success_pct = np.mean(recent_success) * 100.0

                elapsed = time.time() - start_time
                print(f"Episode {ep:4d}/{total_episodes} | "
                      f"Avg Reward: {mean_r:6.1f} | "
                      f"Avg Steps: {mean_s:4.1f} | "
                      f"Success Rate: {success_pct:5.1f}% | "
                      f"Epsilon: {current_eps:.3f} | "
                      f"Q-states: {self.agent.table_size:4d} | "
                      f"Elapsed: {elapsed:.1f}s")

        df_metrics = pd.DataFrame(self.metrics)
        # Compute rolling averages
        df_metrics["rolling_reward_50"] = df_metrics["reward"].rolling(window=50, min_periods=1).mean()
        df_metrics["rolling_steps_50"] = df_metrics["steps"].rolling(window=50, min_periods=1).mean()
        df_metrics["rolling_success_50"] = df_metrics["delivered"].rolling(window=50, min_periods=1).mean() * 100.0

        if verbose:
            print("==================================================")
            print(f"TRAINING COMPLETE in {time.time() - start_time:.2f}s")
            final_succ = df_metrics['rolling_success_50'].iloc[-1]
            print(f"Final 50-Episode Rolling Success Rate: {final_succ:.1f}%")
            print("==================================================")

        return df_metrics


def train_agent(num_episodes: int = 2000,
                env_config: Optional[EnvConfig] = None,
                reward_config: Optional[RewardConfig] = None,
                agent_config: Optional[QLearningConfig] = None,
                save_dir: str = "results",
                verbose: bool = True) -> Tuple[QLearningAgent, TaxiCityEnv, pd.DataFrame]:
    """
    Convenience function to initialize, train, and save Q-agent and metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    trainer = Trainer(env_config=env_config, reward_config=reward_config, agent_config=agent_config)
    df_metrics = trainer.train(num_episodes=num_episodes, verbose=verbose)

    # Save model and CSV metrics
    model_path = os.path.join(save_dir, "q_table_trained.pkl")
    trainer.agent.save(model_path)
    csv_path = os.path.join(save_dir, "training_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)

    return trainer.agent, trainer.env, df_metrics
