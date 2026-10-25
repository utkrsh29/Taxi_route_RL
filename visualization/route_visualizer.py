"""
Route Visualizer for Taxi Reinforcement Learning Agent.
Provides clear ASCII visualization with path arrows and graphical Matplotlib city route maps.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from config import Action, TrafficLevel
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent


class RouteVisualizer:
    """Visualizes trajectory steps, obstacles, traffic levels, and pickup/dropoff on the grid."""

    def __init__(self, env: TaxiCityEnv):
        self.env = env
        self.grid = env.grid

    def rollout_policy(self, agent: Any,
                       fixed_endpoints: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
                       max_steps: int = 100) -> Dict[str, Any]:
        """
        Execute deterministic evaluation rollout using greedy policy.
        Returns detailed trajectory information.
        """
        state = self.env.reset(fixed_endpoints=fixed_endpoints)
        start_taxi = self.env.taxi_pos
        passenger_pos = self.env.passenger_pos
        dest_pos = self.env.dest_pos

        trajectory: List[Tuple[int, int]] = [start_taxi]
        actions_taken: List[int] = []
        rewards: List[float] = []
        events: List[str] = []

        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < max_steps:
            # Greedy action
            if hasattr(agent, "select_action"):
                action = agent.select_action(state, explore=False)
            elif callable(agent):
                action = agent(self.env, state)
            else:
                action = 0

            next_state, reward, term, trunc, info = self.env.step(action)
            done = term or trunc

            actions_taken.append(action)
            rewards.append(reward)
            events.append(info.get("event", "move"))
            trajectory.append(self.env.taxi_pos)

            total_reward += reward
            step_count += 1
            state = next_state

        delivered = (events[-1] == "delivery_success") if events else False

        return {
            "start_taxi": start_taxi,
            "passenger_pos": passenger_pos,
            "dest_pos": dest_pos,
            "trajectory": trajectory,
            "actions": actions_taken,
            "rewards": rewards,
            "events": events,
            "total_reward": total_reward,
            "step_count": step_count,
            "delivered": delivered,
            "traffic_grid": self.env.traffic_mgr.traffic_grid.copy()
        }

    def render_ascii_route(self, rollout_info: Dict[str, Any]) -> str:
        """Generate clear terminal ASCII map showing route steps with directional arrows."""
        rows, cols = self.grid.rows, self.grid.cols
        start_pos = rollout_info["start_taxi"]
        pass_pos = rollout_info["passenger_pos"]
        dest_pos = rollout_info["dest_pos"]
        traj = rollout_info["trajectory"]
        actions = rollout_info["actions"]

        # Map movement arrows
        action_arrows = {
            Action.UP: "^",
            Action.DOWN: "v",
            Action.LEFT: "<",
            Action.RIGHT: ">",
            Action.WAIT: "w"
        }

        # Build grid cell annotations
        char_grid = [["." for _ in range(cols)] for _ in range(rows)]

        # Place obstacles
        for r in range(rows):
            for c in range(cols):
                if self.grid.obstacles[r, c]:
                    char_grid[r][c] = "#"
                elif rollout_info["traffic_grid"][r, c] == TrafficLevel.MEDIUM:
                    char_grid[r][c] = "T"
                elif rollout_info["traffic_grid"][r, c] == TrafficLevel.HIGH:
                    char_grid[r][c] = "H"

        # Overlay path with arrows
        for i in range(len(actions)):
            r, c = traj[i]
            act = Action(actions[i])
            char_grid[r][c] = action_arrows.get(act, "*")

        # Mark key locations
        sr, sc = start_pos
        pr, pc = pass_pos
        dr, dc = dest_pos

        char_grid[sr][sc] = "S"
        char_grid[pr][pc] = "P"
        char_grid[dr][dc] = "D"

        lines = [
            f"Learned Route Trace ({rows}x{cols} City Grid)",
            f"Total Reward: {rollout_info['total_reward']:.1f} | Steps: {rollout_info['step_count']} | Delivered: {rollout_info['delivered']}",
            "Legend: S=Start, P=Passenger, D=Destination, #=Obstacle, T/H=Traffic, ^v<>=Trajectory",
            "-" * (cols * 2 + 3)
        ]

        for r in range(rows):
            lines.append("  " + " ".join(char_grid[r]))
        lines.append("-" * (cols * 2 + 3))

        return "\n".join(lines)

    def plot_route_map(self, rollout_info: Dict[str, Any],
                       title: str = "Learned Taxi Route Optimization",
                       save_path: Optional[str] = "results/learned_route.png"):
        """Generate high-resolution Matplotlib graphical route visualization."""
        rows, cols = self.grid.rows, self.grid.cols
        start_pos = rollout_info["start_taxi"]
        pass_pos = rollout_info["passenger_pos"]
        dest_pos = rollout_info["dest_pos"]
        traj = rollout_info["trajectory"]
        traffic_grid = rollout_info["traffic_grid"]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Base grid background
        for r in range(rows):
            for c in range(cols):
                y = rows - 1 - r
                x = c
                # Check obstacles
                if self.grid.obstacles[r, c]:
                    rect = patches.Rectangle((x, y), 1, 1, facecolor="#374151", edgecolor="#1f2937", linewidth=1.5)
                    ax.add_patch(rect)
                elif traffic_grid[r, c] == TrafficLevel.HIGH:
                    rect = patches.Rectangle((x, y), 1, 1, facecolor="#fecaca", edgecolor="#cbd5e1", linewidth=0.8)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, "HIGH\nTRAFFIC", ha="center", va="center", fontsize=7, color="#dc2626", fontweight="bold")
                elif traffic_grid[r, c] == TrafficLevel.MEDIUM:
                    rect = patches.Rectangle((x, y), 1, 1, facecolor="#fed7aa", edgecolor="#cbd5e1", linewidth=0.8)
                    ax.add_patch(rect)
                    ax.text(x + 0.5, y + 0.5, "MED\nTRAFFIC", ha="center", va="center", fontsize=7, color="#d97706", fontweight="bold")
                else:
                    rect = patches.Rectangle((x, y), 1, 1, facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=0.8)
                    ax.add_patch(rect)

        # Plot trajectory arrows
        path_x = [c + 0.5 for (r, c) in traj]
        path_y = [rows - 1 - r + 0.5 for (r, c) in traj]

        # Draw connecting line
        ax.plot(path_x, path_y, color="#2563eb", linewidth=3.0, alpha=0.8, zorder=3, linestyle="-")

        # Draw directional arrows along trajectory
        for i in range(len(path_x) - 1):
            dx = (path_x[i+1] - path_x[i]) * 0.4
            dy = (path_y[i+1] - path_y[i]) * 0.4
            ax.annotate("", xy=(path_x[i] + dx, path_y[i] + dy), xytext=(path_x[i], path_y[i]),
                        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2.2, mutation_scale=12),
                        zorder=4)

        # Highlight key landmarks
        # Start
        sy = rows - 1 - start_pos[0] + 0.5
        sx = start_pos[1] + 0.5
        ax.scatter(sx, sy, s=400, color="#f59e0b", edgecolors="black", linewidth=2.0, zorder=6, label="Taxi Start (S)")
        ax.text(sx, sy, "S", ha="center", va="center", color="black", fontweight="bold", fontsize=11, zorder=7)

        # Passenger
        py = rows - 1 - pass_pos[0] + 0.5
        px = pass_pos[1] + 0.5
        ax.scatter(px, py, s=400, color="#10b981", edgecolors="black", linewidth=2.0, zorder=6, label="Passenger (P)")
        ax.text(px, py, "P", ha="center", va="center", color="white", fontweight="bold", fontsize=11, zorder=7)

        # Destination
        dy = rows - 1 - dest_pos[0] + 0.5
        dx = dest_pos[1] + 0.5
        ax.scatter(dx, dy, s=400, color="#ef4444", edgecolors="black", linewidth=2.0, zorder=6, label="Destination (D)")
        ax.text(dx, dy, "D", ha="center", va="center", color="white", fontweight="bold", fontsize=11, zorder=7)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(cols + 1))
        ax.set_yticks(np.arange(rows + 1))
        ax.set_aspect("equal")
        ax.grid(color="#94a3b8", linestyle=":", linewidth=0.5)

        status_text = f"Steps: {rollout_info['step_count']} | Reward: {rollout_info['total_reward']:.1f} | Success: {'Delivered' if rollout_info['delivered'] else 'Failed'}"
        ax.set_title(f"{title}\n{status_text}", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True, fontsize=10)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=160)
            plt.close()
            return save_path

        plt.show()
        return None
