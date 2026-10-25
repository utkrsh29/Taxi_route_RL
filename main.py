"""
Main Driver Pipeline for "RL-Based Taxi Route Optimization in a Dynamic Grid City".
College Continuous Assessment RL Project.

Executes:
1. Environment Creation and Topology Display
2. Full Q-Learning Agent Training from scratch
3. Training Curves Visualization & Plot Generation
4. Academic Baseline Benchmarking (Q-Learning vs Random Policy vs Shortest Path)
5. Learned Optimal Route Trajectory Visualization (ASCII & Matplotlib Figure)
6. Optional Hyperparameter Experiments Suite
"""

import os
import sys
import argparse
import pandas as pd

from config import EnvConfig, RewardConfig, QLearningConfig
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent
from training.train import Trainer
from evaluation.evaluate import Evaluator
from visualization.plots import plot_training_curves, plot_baseline_comparison
from visualization.route_visualizer import RouteVisualizer
from experiments.run_experiments import run_all_experiments


def print_banner():
    banner = """
========================================================================
    RL-BASED TAXI ROUTE OPTIMIZATION IN A DYNAMIC GRID CITY
    Course Continuous Assessment - Reinforcement Learning Project
========================================================================
    """
    print(banner)


def run_pipeline(episodes: int = 1500,
                 grid_size: int = 8,
                 dynamic_traffic: bool = False,
                 run_exp: bool = False,
                 save_dir: str = "results"):
    """Execute end-to-end RL pipeline."""
    print_banner()
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Environment Creation
    # ------------------------------------------------------------------
    print("\n[STEP 1/5] Initializing Dynamic City Grid Environment...")
    env_cfg = EnvConfig(
        grid_size=(grid_size, grid_size),
        max_steps=100,
        obstacle_density=0.15,
        traffic_density=0.25,
        dynamic_traffic=dynamic_traffic,
        use_stations=True,
        num_stations=4,
        seed=42
    )
    reward_cfg = RewardConfig()
    q_cfg = QLearningConfig(
        alpha=0.2,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        num_episodes=episodes
    )

    env = TaxiCityEnv(env_config=env_cfg, reward_config=reward_cfg)
    env.reset()

    print(f"City Size: {grid_size}x{grid_size} cells")
    print(f"Obstacles: {int(grid_size * grid_size * env_cfg.obstacle_density)} blocked cells")
    print(f"Transit Stations: {len(env.grid.stations)} designated hubs")
    print(f"Dynamic Traffic Mode: {'ENABLED' if dynamic_traffic else 'STATIC'}")
    print("\nInitial Environment ASCII Snapshot:")
    print(env.render())

    # ------------------------------------------------------------------
    # Step 2: Agent Training
    # ------------------------------------------------------------------
    print(f"\n[STEP 2/5] Training Tabular Q-Learning Agent ({episodes} Episodes)...")
    trainer = Trainer(env=env, env_config=env_cfg, reward_config=reward_cfg, agent_config=q_cfg)
    df_metrics = trainer.train(num_episodes=episodes, eval_interval=200, verbose=True)

    # Save trained model and metrics
    model_path = os.path.join(save_dir, "q_table_trained.pkl")
    trainer.agent.save(model_path)
    metrics_path = os.path.join(save_dir, "training_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Trained Q-Table saved to -> {model_path}")
    print(f"Training metrics saved to -> {metrics_path}")

    # ------------------------------------------------------------------
    # Step 3: Visualization of Training Metrics
    # ------------------------------------------------------------------
    print("\n[STEP 3/5] Generating Publication-Quality Training Visualizations...")
    curves_path = plot_training_curves(df_metrics, save_dir=save_dir, filename="training_curves.png")
    print(f"Saved: {curves_path}")

    # ------------------------------------------------------------------
    # Step 4: Baseline Benchmarking & Academic Evaluation
    # ------------------------------------------------------------------
    print("\n[STEP 4/5] Academic Benchmarking vs Baseline Policies...")
    evaluator = Evaluator(env=env, env_config=env_cfg)
    df_benchmark = evaluator.benchmark_all(trainer.agent, num_scenarios=50, seed=123)

    print("\n" + "=" * 72)
    print("           ACADEMIC PERFORMANCE COMPARISON (50 Test Scenarios)")
    print("=" * 72)
    print(df_benchmark.to_string(index=False))
    print("=" * 72)

    # Save benchmark table and plot
    benchmark_csv = os.path.join(save_dir, "benchmark_comparison.csv")
    df_benchmark.to_csv(benchmark_csv, index=False)
    comp_plot_path = plot_baseline_comparison(df_benchmark, save_dir=save_dir, filename="baseline_comparison.png")
    print(f"Saved: {comp_plot_path}")

    # ------------------------------------------------------------------
    # Step 5: Learned Optimal Route Demonstration
    # ------------------------------------------------------------------
    print("\n[STEP 5/5] Visualizing Learned Route Demonstration...")
    visualizer = RouteVisualizer(env=env)

    # Test sample scenario
    sample_endpoints = None
    if len(env.grid.stations) >= 2:
        sample_endpoints = (env.grid.stations[0], env.grid.stations[1], env.grid.stations[-1])

    rollout = visualizer.rollout_policy(trainer.agent, fixed_endpoints=sample_endpoints)
    ascii_trace = visualizer.render_ascii_route(rollout)
    print("\nLearned Policy Mission Trace:")
    print(ascii_trace)

    route_img = os.path.join(save_dir, "learned_route.png")
    visualizer.plot_route_map(rollout, title="Learned Taxi Route (Greedy Q-Policy)", save_path=route_img)
    print(f"Graphical Route Map saved to -> {route_img}")

    # ------------------------------------------------------------------
    # Optional Step 6: Full Experimental Suite
    # ------------------------------------------------------------------
    if run_exp:
        print("\n[OPTIONAL STEP] Running Academic Experiments Suite...")
        run_all_experiments(results_dir=save_dir, episodes_per_run=600)

    print("\n========================================================================")
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  All results, CSVs, models, and graphs are saved in '{save_dir}/'")
    print("========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Based Taxi Route Optimization")
    parser.add_argument("--episodes", type=int, default=1500, help="Training episodes")
    parser.add_argument("--grid_size", type=int, default=8, help="City grid dimension (NxN)")
    parser.add_argument("--dynamic_traffic", action="store_true", help="Enable dynamic traffic")
    parser.add_argument("--run_experiments", action="store_true", help="Run full experiments suite")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory for artifacts")

    args = parser.parse_args()
    run_pipeline(
        episodes=args.episodes,
        grid_size=args.grid_size,
        dynamic_traffic=args.dynamic_traffic,
        run_exp=args.run_experiments,
        save_dir=args.save_dir
    )
