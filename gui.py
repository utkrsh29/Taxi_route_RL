"""
Interactive Desktop GUI for "RL-Based Taxi Route Optimization in a Dynamic Grid City".
Built with Python's native Tkinter (Zero external dependencies).

Features:
- Real-time animated grid simulation (Taxi, Passenger, Destination, Obstacles, Traffic)
- Policy Switcher: Compare Q-Learning vs Shortest-Path vs Random Policy live
- Step-by-Step and Auto-Play modes with adjustable speed
- Preset Station Mission Selector (Airport, Downtown, Station, Suburb)
- Live metrics telemetry (Current State, Cumulative Reward, Steps, Target)
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Tuple, Optional, List, Dict, Any

from config import EnvConfig, RewardConfig, QLearningConfig, Action, TrafficLevel
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent
from baselines.random_policy import RandomPolicyAgent
from baselines.shortest_path import ShortestPathAgent
from training.train import Trainer


class TaxiRLApp:
    """Tkinter Desktop Frontend for Taxi Route Optimization."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RL-Based Taxi Route Optimization - Interactive City")
        self.root.geometry("1100x750")
        self.root.minsize(950, 680)

        # Environment & RL setup
        self.grid_dim = 8
        self.env_cfg = EnvConfig(grid_size=(self.grid_dim, self.grid_dim), num_stations=4, seed=42)
        self.env = TaxiCityEnv(env_config=self.env_cfg)
        self.agent = QLearningAgent(num_actions=4, config=QLearningConfig(alpha=0.2, gamma=0.98))

        # Baselines
        self.random_agent = RandomPolicyAgent(seed=42)
        self.shortest_agent = ShortestPathAgent()

        # Load trained model if exists, else quick train
        model_file = os.path.join("results", "q_table_trained.pkl")
        if os.path.exists(model_file):
            self.agent.load(model_file)
        else:
            trainer = Trainer(env=self.env, agent=self.agent, env_config=self.env_cfg)
            trainer.train(num_episodes=1000, verbose=False)

        # Simulation state
        self.is_playing = False
        self.sim_speed_ms = 220
        self.current_state = self.env.reset()
        self.initial_endpoints = (self.env.taxi_pos, self.env.passenger_pos, self.env.dest_pos)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.current_reward = 0.0
        self.step_history: List[Tuple[int, int]] = [self.env.taxi_pos]
        self.actions_history: List[int] = []

        self._build_ui()
        self.render_grid()

    def _build_ui(self):
        """Construct GUI components."""
        # Main layout: Left = Canvas, Right = Control Panel
        self.root.configure(bg="#0f172a")

        header = tk.Frame(self.root, bg="#1e293b", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        title_lbl = tk.Label(header, text="🚖 RL Taxi Route Optimization | Dynamic Grid City",
                             font=("Segoe UI", 16, "bold"), fg="#f8fafc", bg="#1e293b", pady=10)
        title_lbl.pack(side=tk.LEFT, padx=20)

        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left: City Grid Canvas Frame
        canvas_card = tk.Frame(main_frame, bg="#1e293b", bd=2, relief=tk.FLAT)
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        canvas_header = tk.Label(canvas_card, text="City Map & Live Vehicle Telemetry",
                                 font=("Segoe UI", 12, "bold"), fg="#94a3b8", bg="#1e293b", pady=8)
        canvas_header.pack(side=tk.TOP)

        self.canvas_size = 540
        self.canvas = tk.Canvas(canvas_card, width=self.canvas_size, height=self.canvas_size,
                               bg="#0f172a", highlightthickness=0)
        self.canvas.pack(padx=15, pady=10)

        # Legend
        legend_frame = tk.Frame(canvas_card, bg="#1e293b")
        legend_frame.pack(fill=tk.X, pady=(0, 10), padx=15)
        legends = [
            ("🚕 Taxi (S)", "#f59e0b"),
            ("👤 Passenger (P)", "#10b981"),
            ("🏁 Dest (D)", "#ef4444"),
            ("⬛ Roadblock (#)", "#334155"),
            ("🟧 Med Traffic", "#f97316"),
            ("🟥 High Traffic", "#dc2626")
        ]
        for name, col in legends:
            lbl = tk.Label(legend_frame, text=name, fg=col, bg="#1e293b", font=("Segoe UI", 9, "bold"))
            lbl.pack(side=tk.LEFT, padx=6)

        # Right: Dashboard & Control Panel
        ctrl_card = tk.Frame(main_frame, bg="#1e293b", width=380, bd=2, relief=tk.FLAT)
        ctrl_card.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        ctrl_card.pack_propagate(False)

        # Telemetry Stats Frame
        stats_frame = tk.LabelFrame(ctrl_card, text=" Live Mission Status ", font=("Segoe UI", 11, "bold"),
                                   fg="#38bdf8", bg="#1e293b", bd=1, relief=tk.SOLID, padx=12, pady=10)
        stats_frame.pack(fill=tk.X, padx=15, pady=12)

        self.lbl_policy = tk.Label(stats_frame, text="Active Agent: Q-Learning (Greedy)",
                                   font=("Segoe UI", 10, "bold"), fg="#a855f7", bg="#1e293b", anchor="w")
        self.lbl_policy.pack(fill=tk.X, pady=2)

        self.lbl_steps = tk.Label(stats_frame, text="Steps Taken: 0 / 100",
                                  font=("Segoe UI", 10), fg="#f8fafc", bg="#1e293b", anchor="w")
        self.lbl_steps.pack(fill=tk.X, pady=2)

        self.lbl_reward = tk.Label(stats_frame, text="Cumulative Reward: 0.0",
                                   font=("Segoe UI", 10, "bold"), fg="#22c55e", bg="#1e293b", anchor="w")
        self.lbl_reward.pack(fill=tk.X, pady=2)

        self.lbl_status = tk.Label(stats_frame, text="Mission: Navigating to Passenger",
                                   font=("Segoe UI", 10), fg="#facc15", bg="#1e293b", anchor="w")
        self.lbl_status.pack(fill=tk.X, pady=2)

        self.lbl_last_action = tk.Label(stats_frame, text="Last Action: None",
                                        font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b", anchor="w")
        self.lbl_last_action.pack(fill=tk.X, pady=2)

        # Control Panel: Agent Selector
        agent_frame = tk.LabelFrame(ctrl_card, text=" Agent / Policy Selection ", font=("Segoe UI", 10, "bold"),
                                   fg="#cbd5e1", bg="#1e293b", padx=10, pady=8)
        agent_frame.pack(fill=tk.X, padx=15, pady=6)

        self.selected_agent = tk.StringVar(value="Q-Learning (Trained RL)")
        agent_menu = ttk.Combobox(agent_frame, textvariable=self.selected_agent, state="readonly",
                                  values=["Q-Learning (Trained RL)", "Shortest-Path Baseline (BFS)", "Random Policy"])
        agent_menu.pack(fill=tk.X, pady=4)
        agent_menu.bind("<<ComboboxSelected>>", self._on_agent_changed)

        # Station Mission Selector
        mission_frame = tk.LabelFrame(ctrl_card, text=" Transit Mission Preset ", font=("Segoe UI", 10, "bold"),
                                     fg="#cbd5e1", bg="#1e293b", padx=10, pady=8)
        mission_frame.pack(fill=tk.X, padx=15, pady=6)

        self.mission_options = [
            "Station 0 -> Station 1",
            "Station 1 -> Station 2",
            "Station 2 -> Station 3",
            "Station 3 -> Station 0",
            "Station 0 -> Station 2",
            "Station 1 -> Station 3"
        ]
        self.selected_mission = tk.StringVar(value=self.mission_options[0])
        mission_menu = ttk.Combobox(mission_frame, textvariable=self.selected_mission, state="readonly",
                                    values=self.mission_options)
        mission_menu.pack(fill=tk.X, pady=4)

        btn_apply_mission = tk.Button(mission_frame, text="Load Mission Route", command=self._apply_preset_mission,
                                      bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT)
        btn_apply_mission.pack(fill=tk.X, pady=3)

        # Simulation Controls
        sim_ctrl_frame = tk.Frame(ctrl_card, bg="#1e293b")
        sim_ctrl_frame.pack(fill=tk.X, padx=15, pady=10)

        self.btn_play = tk.Button(sim_ctrl_frame, text="▶ Auto Play", command=self.toggle_play,
                                  bg="#16a34a", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, height=1)
        self.btn_play.pack(fill=tk.X, pady=3)

        btn_step = tk.Button(sim_ctrl_frame, text="⏩ Single Step", command=self.step_agent,
                             bg="#2563eb", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
        btn_step.pack(fill=tk.X, pady=3)

        btn_restart = tk.Button(sim_ctrl_frame, text="↺ Reset Taxi to Start (Same Mission)", command=self.restart_same_mission,
                                bg="#0284c7", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
        btn_restart.pack(fill=tk.X, pady=3)

        btn_reset = tk.Button(sim_ctrl_frame, text="🎲 New Random Mission", command=self.reset_env,
                              bg="#475569", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
        btn_reset.pack(fill=tk.X, pady=3)

        # Speed slider
        speed_frame = tk.Frame(ctrl_card, bg="#1e293b")
        speed_frame.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(speed_frame, text="Playback Delay (Speed):", fg="#94a3b8", bg="#1e293b",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.speed_slider = tk.Scale(speed_frame, from_=50, to=500, orient=tk.HORIZONTAL,
                                     bg="#1e293b", fg="#f8fafc", highlightthickness=0, command=self._on_speed_change)
        self.speed_slider.set(self.sim_speed_ms)
        self.speed_slider.pack(fill=tk.X)

        # Retrain Button
        btn_train = tk.Button(ctrl_card, text="🧠 Retrain Agent (1000 Ep)", command=self._retrain_dialog,
                              bg="#9333ea", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT)
        btn_train.pack(fill=tk.X, padx=15, pady=(10, 5))

    def _on_speed_change(self, val):
        self.sim_speed_ms = int(val)

    def _on_agent_changed(self, event=None):
        name = self.selected_agent.get()
        self.lbl_policy.config(text=f"Active Agent: {name}")

    def _apply_preset_mission(self):
        """Set endpoints based on dropdown station selection."""
        self.is_playing = False
        self.btn_play.config(text="▶ Auto Play", bg="#16a34a")
        st = self.env.grid.stations
        mission = self.selected_mission.get()

        idx1 = int(mission.split("->")[0].strip().split()[-1])
        idx2 = int(mission.split("->")[1].strip().split()[-1])

        start_taxi = st[idx1]
        passenger = st[idx1]
        dest = st[idx2]

        self.current_state = self.env.reset(fixed_endpoints=(start_taxi, passenger, dest), reset_traffic=True)
        self.initial_endpoints = (start_taxi, passenger, dest)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.current_reward = 0.0
        self.step_history = [self.env.taxi_pos]
        self.actions_history = []
        self._update_telemetry()
        self.render_grid()

    def restart_same_mission(self):
        """Reset taxi to original starting position while keeping passenger, destination, AND traffic identical."""
        self.is_playing = False
        self.btn_play.config(text="▶ Auto Play", bg="#16a34a")
        self.current_state = self.env.reset(fixed_endpoints=self.initial_endpoints, reset_traffic=False)
        self.env.traffic_mgr.traffic_grid = self.initial_traffic.copy()
        self.current_reward = 0.0
        self.step_history = [self.env.taxi_pos]
        self.actions_history = []
        self._update_telemetry()
        self.render_grid()

    def reset_env(self):
        """Randomize mission endpoints and traffic."""
        self.is_playing = False
        self.btn_play.config(text="▶ Auto Play", bg="#16a34a")
        self.current_state = self.env.reset(reset_traffic=True)
        self.initial_endpoints = (self.env.taxi_pos, self.env.passenger_pos, self.env.dest_pos)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.current_reward = 0.0
        self.step_history = [self.env.taxi_pos]
        self.actions_history = []
        self._update_telemetry()
        self.render_grid()

    def _retrain_dialog(self):
        """Trigger rapid background training with status prompt."""
        self.is_playing = False
        self.btn_play.config(text="▶ Auto Play", bg="#16a34a")
        self.lbl_status.config(text="Status: Retraining Agent...", fg="#a855f7")
        self.root.update()

        trainer = Trainer(env=self.env, agent=self.agent, env_config=self.env_cfg)
        df = trainer.train(num_episodes=1000, verbose=False)
        self.agent.save(os.path.join("results", "q_table_trained.pkl"))

        messagebox.showinfo("Training Complete", "Agent retraining finished successfully!\nModel saved to results/q_table_trained.pkl")
        self.reset_env()

    def toggle_play(self):
        """Toggle automatic simulation playback."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause", bg="#dc2626")
            self._playback_loop()
        else:
            self.btn_play.config(text="▶ Auto Play", bg="#16a34a")

    def _playback_loop(self):
        """Recursive timer callback for auto-play animation."""
        if not self.is_playing:
            return

        done = self.step_agent()
        if not done:
            self.root.after(self.sim_speed_ms, self._playback_loop)
        else:
            self.is_playing = False
            self.btn_play.config(text="▶ Auto Play", bg="#16a34a")

    def step_agent(self) -> bool:
        """Execute a single step using the selected agent policy."""
        agent_type = self.selected_agent.get()
        if "Q-Learning" in agent_type:
            action = self.agent.select_action(self.current_state, explore=False)
        elif "Shortest-Path" in agent_type:
            action = self.shortest_agent.select_action(self.current_state, env=self.env)
        else:
            action = self.random_agent.select_action(self.current_state)

        next_state, reward, term, trunc, info = self.env.step(action)
        self.current_state = next_state
        self.current_reward += reward
        self.step_history.append(self.env.taxi_pos)
        self.actions_history.append(action)

        self._update_telemetry(action_taken=action, event=info.get("event"))
        self.render_grid()

        if term:
            self.lbl_status.config(text="MISSION SUCCESS! Passenger Delivered 🎉", fg="#22c55e")
            return True
        elif trunc:
            self.lbl_status.config(text="TIMEOUT: Exceeded Maximum Steps (100)", fg="#ef4444")
            return True
        return False

    def _update_telemetry(self, action_taken: Optional[int] = None, event: Optional[str] = None):
        """Refresh status metrics in UI."""
        act_names = ["UP ↑", "DOWN ↓", "LEFT ←", "RIGHT →", "WAIT ⏸"]
        act_str = act_names[action_taken] if action_taken is not None else "None"

        self.lbl_steps.config(text=f"Steps Taken: {self.env.steps_taken} / {self.env_cfg.max_steps}")
        color = "#22c55e" if self.current_reward >= 0 else "#f87171"
        self.lbl_reward.config(text=f"Cumulative Reward: {self.current_reward:.1f}", fg=color)
        self.lbl_last_action.config(text=f"Last Action: {act_str} ({event or 'ready'})")

        if self.env.passenger_picked:
            self.lbl_status.config(text="Status: Passenger On-board 🚕 -> Routing to Dest", fg="#38bdf8")
        else:
            self.lbl_status.config(text="Status: Navigating to Passenger 👤", fg="#facc15")

    def render_grid(self):
        """Draw dynamic city canvas with obstacles, traffic, and vehicle path."""
        self.canvas.delete("all")
        rows, cols = self.env.grid.rows, self.env.grid.cols
        cell_size = self.canvas_size / cols
        traffic_grid = self.env.traffic_mgr.traffic_grid

        # 1. Base Grid, Obstacles & Traffic Cells
        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Check obstacle
                if self.env.grid.obstacles[r, c]:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#334155", outline="#1e293b", width=1.5)
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text="#", fill="#64748b", font=("Segoe UI", 12, "bold"))
                elif traffic_grid[r, c] == TrafficLevel.HIGH:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#7f1d1d", outline="#450a0a", width=1)
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text="HIGH", fill="#fca5a5", font=("Segoe UI", 8, "bold"))
                elif traffic_grid[r, c] == TrafficLevel.MEDIUM:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#7c2d12", outline="#431407", width=1)
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text="MED", fill="#fed7aa", font=("Segoe UI", 8, "bold"))
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1e293b", outline="#334155", width=1)

        # 2. Draw Trajectory Path History
        if len(self.step_history) >= 2:
            pts = []
            for (r, c) in self.step_history:
                pts.append(c * cell_size + cell_size / 2)
                pts.append(r * cell_size + cell_size / 2)
            self.canvas.create_line(pts, fill="#38bdf8", width=3, smooth=False, arrow=tk.LAST)

        # 3. Passenger Location (if not picked)
        if not self.env.passenger_picked:
            pr, pc = self.env.passenger_pos
            px = pc * cell_size + cell_size / 2
            py = pr * cell_size + cell_size / 2
            self.canvas.create_oval(px - 18, py - 18, px + 18, py + 18, fill="#10b981", outline="#059669", width=2)
            self.canvas.create_text(px, py, text="P", fill="white", font=("Segoe UI", 11, "bold"))

        # 4. Destination Location
        dr, dc = self.env.dest_pos
        dx = dc * cell_size + cell_size / 2
        dy = dr * cell_size + cell_size / 2
        self.canvas.create_rectangle(dx - 18, dy - 18, dx + 18, dy + 18, fill="#ef4444", outline="#b91c1c", width=2)
        self.canvas.create_text(dx, dy, text="D", fill="white", font=("Segoe UI", 11, "bold"))

        # 5. Taxi Vehicle
        tr, tc = self.env.taxi_pos
        tx = tc * cell_size + cell_size / 2
        ty = tr * cell_size + cell_size / 2
        self.canvas.create_oval(tx - 20, ty - 20, tx + 20, ty + 20, fill="#f59e0b", outline="#d97706", width=2.5)
        badge = "🚕" if not self.env.passenger_picked else "🚕👤"
        self.canvas.create_text(tx, ty, text=badge, font=("Segoe UI", 11))


def main():
    root = tk.Tk()
    app = TaxiRLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
