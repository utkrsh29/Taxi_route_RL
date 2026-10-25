"""
Experiments Suite for Taxi Route Optimization in Dynamic Grid City.
Executes academic experiments:
- Exp 1: Learning Rates Comparison (alpha)
- Exp 2: Discount Factors Comparison (gamma)
- Exp 3: Epsilon Decay Rates Comparison
- Exp 4: Static Traffic vs Dynamic Traffic
- Exp 5: Small Grid (6x6) vs Large Grid (10x10)
- Exp 6: Q-Learning vs Baseline Policies Benchmark
"""

import os
import sys
import argparse
from typing import Dict
import pandas as pd

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import EnvConfig, RewardConfig, QLearningConfig
from training.train import Trainer
from evaluation.evaluate import Evaluator
from visualization.plots import (
    plot_training_curves,
    plot_baseline_comparison,
    plot_hyperparameter_experiment
)


def run_experiment_1_alpha(results_dir: str = "results", episodes: int = 1000):
    """Experiment 1: Learning Rate Comparison."""
    print("\n>>> Running Experiment 1: Learning Rates (alpha)...")
    alphas = [0.05, 0.1, 0.3]
    results: Dict[str, pd.DataFrame] = {}

    for alpha in alphas:
        label = f"alpha={alpha}"
        print(f"  Training with {label}...")
        q_cfg = QLearningConfig(alpha=alpha, gamma=0.99, epsilon_decay=0.995, num_episodes=episodes)
        trainer = Trainer(agent_config=q_cfg)
        df = trainer.train(num_episodes=episodes, verbose=False)
        results[label] = df

    plot_hyperparameter_experiment(
        results, param_name="Learning Rate (Alpha)",
        metric_col="rolling_reward_50",
        ylabel="50-Ep Rolling Reward",
        save_dir=results_dir,
        filename="exp1_learning_rates.png"
    )
    print("  Exp 1 completed & saved to exp1_learning_rates.png")


def run_experiment_2_gamma(results_dir: str = "results", episodes: int = 1000):
    """Experiment 2: Discount Factor Comparison."""
    print("\n>>> Running Experiment 2: Discount Factor (gamma)...")
    gammas = [0.80, 0.95, 0.99]
    results: Dict[str, pd.DataFrame] = {}

    for gamma in gammas:
        label = f"gamma={gamma}"
        print(f"  Training with {label}...")
        q_cfg = QLearningConfig(alpha=0.1, gamma=gamma, epsilon_decay=0.995, num_episodes=episodes)
        trainer = Trainer(agent_config=q_cfg)
        df = trainer.train(num_episodes=episodes, verbose=False)
        results[label] = df

    plot_hyperparameter_experiment(
        results, param_name="Discount Factor (Gamma)",
        metric_col="rolling_reward_50",
        ylabel="50-Ep Rolling Reward",
        save_dir=results_dir,
        filename="exp2_discount_factors.png"
    )
    print("  Exp 2 completed & saved to exp2_discount_factors.png")


def run_experiment_3_epsilon_decay(results_dir: str = "results", episodes: int = 1000):
    """Experiment 3: Epsilon Decay Rate Comparison."""
    print("\n>>> Running Experiment 3: Epsilon Decay Rates...")
    decays = [0.990, 0.995, 0.998]
    results: Dict[str, pd.DataFrame] = {}

    for decay in decays:
        label = f"decay={decay}"
        print(f"  Training with {label}...")
        q_cfg = QLearningConfig(alpha=0.1, gamma=0.99, epsilon_decay=decay, num_episodes=episodes)
        trainer = Trainer(agent_config=q_cfg)
        df = trainer.train(num_episodes=episodes, verbose=False)
        results[label] = df

    plot_hyperparameter_experiment(
        results, param_name="Epsilon Decay Rate",
        metric_col="rolling_success_50",
        ylabel="50-Ep Rolling Success Rate (%)",
        save_dir=results_dir,
        filename="exp3_epsilon_decays.png"
    )
    print("  Exp 3 completed & saved to exp3_epsilon_decays.png")


def run_experiment_4_traffic(results_dir: str = "results", episodes: int = 1000):
    """Experiment 4: Static Traffic vs Dynamic Traffic."""
    print("\n>>> Running Experiment 4: Static vs Dynamic Traffic...")
    results: Dict[str, pd.DataFrame] = {}

    for dynamic in [False, True]:
        label = "Dynamic Traffic" if dynamic else "Static Traffic"
        print(f"  Training under {label}...")
        env_cfg = EnvConfig(dynamic_traffic=dynamic, traffic_change_prob=0.08)
        q_cfg = QLearningConfig(num_episodes=episodes)
        trainer = Trainer(env_config=env_cfg, agent_config=q_cfg)
        df = trainer.train(num_episodes=episodes, verbose=False)
        results[label] = df

    plot_hyperparameter_experiment(
        results, param_name="Static vs Dynamic Traffic",
        metric_col="rolling_reward_50",
        ylabel="50-Ep Rolling Reward",
        save_dir=results_dir,
        filename="exp4_traffic_comparison.png"
    )
    print("  Exp 4 completed & saved to exp4_traffic_comparison.png")


def run_experiment_5_grid_size(results_dir: str = "results", episodes: int = 1000):
    """Experiment 5: Small Grid (6x6) vs Large Grid (10x10)."""
    print("\n>>> Running Experiment 5: Small Grid (6x6) vs Large Grid (10x10)...")
    results: Dict[str, pd.DataFrame] = {}

    sizes = [(6, 6), (10, 10)]
    for size in sizes:
        label = f"Grid {size[0]}x{size[1]}"
        print(f"  Training on {label}...")
        env_cfg = EnvConfig(grid_size=size, max_steps=100)
        q_cfg = QLearningConfig(num_episodes=episodes)
        trainer = Trainer(env_config=env_cfg, agent_config=q_cfg)
        df = trainer.train(num_episodes=episodes, verbose=False)
        results[label] = df

    plot_hyperparameter_experiment(
        results, param_name="Grid Size Complexity",
        metric_col="rolling_steps_50",
        ylabel="50-Ep Rolling Steps to Delivery",
        save_dir=results_dir,
        filename="exp5_grid_sizes.png"
    )
    print("  Exp 5 completed & saved to exp5_grid_sizes.png")


def run_all_experiments(results_dir: str = "results", episodes_per_run: int = 800):
    """Run full academic experiment suite sequentially."""
    os.makedirs(results_dir, exist_ok=True)
    print("==================================================")
    print("STARTING COMPLETE RL EXPERIMENTAL SUITE")
    print(f"Results Directory: {results_dir}")
    print("==================================================")

    run_experiment_1_alpha(results_dir=results_dir, episodes=episodes_per_run)
    run_experiment_2_gamma(results_dir=results_dir, episodes=episodes_per_run)
    run_experiment_3_epsilon_decay(results_dir=results_dir, episodes=episodes_per_run)
    run_experiment_4_traffic(results_dir=results_dir, episodes=episodes_per_run)
    run_experiment_5_grid_size(results_dir=results_dir, episodes=episodes_per_run)

    print("\n==================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! All graphs saved in results/")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Taxi RL Experiments")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["1", "2", "3", "4", "5", "all"],
                        help="Experiment number to run (1-5 or all)")
    parser.add_argument("--episodes", type=int, default=800,
                        help="Number of episodes per experimental configuration")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to store figures and data")
    args = parser.parse_args()

    if args.exp == "1":
        run_experiment_1_alpha(args.save_dir, args.episodes)
    elif args.exp == "2":
        run_experiment_2_gamma(args.save_dir, args.episodes)
    elif args.exp == "3":
        run_experiment_3_epsilon_decay(args.save_dir, args.episodes)
    elif args.exp == "4":
        run_experiment_4_traffic(args.save_dir, args.episodes)
    elif args.exp == "5":
        run_experiment_5_grid_size(args.save_dir, args.episodes)
    else:
        run_all_experiments(args.save_dir, args.episodes)
