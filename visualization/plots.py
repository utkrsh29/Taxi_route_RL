"""
Plotting and Visualization module for Taxi Reinforcement Learning.
Generates publication-quality training curves and baseline comparison charts using Matplotlib.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set modern aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})


def plot_training_curves(df_metrics: pd.DataFrame, save_dir: str = "results", filename: str = "training_curves.png"):
    """
    Generate comprehensive 4-panel training dashboard:
    1. Episode Reward & 50-Ep Moving Average
    2. Steps per Episode & Moving Average
    3. Success Rate (%) Progression
    4. Epsilon Decay & TD-Error
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = df_metrics["episode"]

    # 1. Reward Curve
    ax = axes[0, 0]
    ax.plot(episodes, df_metrics["reward"], color="#6baed6", alpha=0.35, label="Raw Episode Reward")
    ax.plot(episodes, df_metrics["rolling_reward_50"], color="#08519c", linewidth=2.0, label="50-Ep Moving Avg")
    ax.set_title("Reward vs Episode", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Steps per Episode
    ax = axes[0, 1]
    ax.plot(episodes, df_metrics["steps"], color="#fc9272", alpha=0.35, label="Raw Steps")
    ax.plot(episodes, df_metrics["rolling_steps_50"], color="#cb181d", linewidth=2.0, label="50-Ep Moving Avg")
    ax.set_title("Steps per Episode (Travel Time)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps Taken")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Success Rate
    ax = axes[1, 0]
    ax.plot(episodes, df_metrics["rolling_success_50"], color="#238b45", linewidth=2.2, label="Success Rate (50-ep avg)")
    ax.set_title("Passenger Delivery Success Rate (%)", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.axhline(100, color="grey", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Epsilon & Exploration
    ax = axes[1, 1]
    color1 = "#756bb1"
    ax.plot(episodes, df_metrics["epsilon"], color=color1, linewidth=2.0, label="Epsilon (Exploration Rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)

    # Secondary axis for TD Error
    if "avg_td_error" in df_metrics.columns:
        ax2 = ax.twinx()
        rolling_td = df_metrics["avg_td_error"].rolling(window=50, min_periods=1).mean()
        color2 = "#fe9929"
        ax2.plot(episodes, rolling_td, color=color2, linewidth=1.8, linestyle="--", label="TD Error (50-ep avg)")
        ax2.set_ylabel("Avg TD Error", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

    ax.set_title("Exploration Decay & Bellman TD-Error Convergence", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_baseline_comparison(comparison_df: pd.DataFrame, save_dir: str = "results", filename: str = "baseline_comparison.png"):
    """
    Generate grouped comparison bar charts for:
    - Average Reward
    - Average Steps
    - Delivery Success Rate (%)
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    policies = comparison_df["policy"].tolist()
    colors = ["#969696", "#3182bd", "#31a354"] if len(policies) == 3 else ["#3182bd", "#31a354"]

    # 1. Average Reward
    ax = axes[0]
    bars = ax.bar(policies, comparison_df["mean_reward"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_title("Average Cumulative Reward\n(Higher is Better)", fontweight="bold")
    ax.set_ylabel("Mean Reward")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -14), textcoords="offset points",
                    ha="center", va="bottom" if h >= 0 else "top", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # 2. Average Steps
    ax = axes[1]
    bars = ax.bar(policies, comparison_df["mean_steps"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_title("Average Steps to Delivery\n(Lower is Better)", fontweight="bold")
    ax.set_ylabel("Mean Steps")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # 3. Success Rate
    ax = axes[2]
    bars = ax.bar(policies, comparison_df["success_rate"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_title("Delivery Success Rate (%)\n(Higher is Better)", fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_hyperparameter_experiment(results_dict: Dict[str, pd.DataFrame], param_name: str,
                                   metric_col: str = "rolling_reward_50",
                                   ylabel: str = "50-Ep Rolling Reward",
                                   save_dir: str = "results",
                                   filename: Optional[str] = None):
    """Plot multi-curve comparison for hyperparameter sweeps."""
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"experiment_{param_name.lower().replace(' ', '_')}.png"

    plt.figure(figsize=(9, 5.5))
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for idx, (label, df) in enumerate(results_dict.items()):
        color = palette[idx % len(palette)]
        plt.plot(df["episode"], df[metric_col], label=label, linewidth=2.0, color=color)

    plt.title(f"Hyperparameter Study: Effect of {param_name}", fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
