"""
Zero-Dependency Modern Web Dashboard for Taxi Route Optimization.
Uses Python's standard library `http.server` to serve an interactive HTML5/Canvas UI
with real-time REST API endpoints for stepping, resetting, and policy evaluation.

Launch:
    python web_app.py
Then open http://localhost:5000 in your browser!
"""

import os
import time
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser

from config import EnvConfig, RewardConfig, QLearningConfig, Action, TrafficLevel
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent
from baselines.random_policy import RandomPolicyAgent
from baselines.shortest_path import ShortestPathAgent
from training.train import Trainer


# Global Singleton Application State
class SimulationManager:
    def __init__(self):
        self.grid_dim = 8
        self.env_cfg = EnvConfig(grid_size=(self.grid_dim, self.grid_dim), num_stations=4, seed=42)
        self.env = TaxiCityEnv(env_config=self.env_cfg)
        self.agent = QLearningAgent(num_actions=4, config=QLearningConfig(alpha=0.2, gamma=0.98))
        self.random_agent = RandomPolicyAgent(seed=42)
        self.shortest_agent = ShortestPathAgent()

        model_file = os.path.join("results", "q_table_trained.pkl")
        if os.path.exists(model_file):
            self.agent.load(model_file)
        else:
            trainer = Trainer(env=self.env, agent=self.agent, env_config=self.env_cfg)
            trainer.train(num_episodes=1000, verbose=False)

        self.current_state = self.env.reset()
        self.initial_endpoints = (self.env.taxi_pos, self.env.passenger_pos, self.env.dest_pos)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.total_reward = 0.0
        self.history = [self.env.taxi_pos]
        self.active_policy = "q_learning"

    def generate_world(self, rows: int = 8, cols: int = 8):
        """Generate a brand new city with user-specified dimensions (Rows x Cols 4-15) and optimize agent."""
        rows = max(4, min(15, int(rows)))
        cols = max(4, min(15, int(cols)))
        self.grid_dim = (rows, cols)
        new_seed = int(time.time() * 1000) % 100000
        max_steps = max(100, (rows + cols) * 6)
        self.env_cfg = EnvConfig(grid_size=(rows, cols), max_steps=max_steps, num_stations=4, seed=new_seed)
        self.env = TaxiCityEnv(env_config=self.env_cfg)
        self.agent = QLearningAgent(num_actions=4, config=QLearningConfig(alpha=0.1, gamma=0.99, epsilon=0.05, epsilon_decay=0.98))
        self.agent.seed_from_topology(self.env)
        trainer = Trainer(env=self.env, agent=self.agent, env_config=self.env_cfg)
        trainer.train(num_episodes=150, verbose=False)
        self.current_state = self.env.reset(reset_traffic=True)
        self.initial_endpoints = (self.env.taxi_pos, self.env.passenger_pos, self.env.dest_pos)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.total_reward = 0.0
        self.history = [self.env.taxi_pos]
        return self.get_payload()

    def reset(self, station_pair=None):
        if station_pair and len(self.env.grid.stations) >= 2:
            st = self.env.grid.stations
            s_idx, d_idx = station_pair
            self.current_state = self.env.reset(fixed_endpoints=(st[s_idx], st[s_idx], st[d_idx]), reset_traffic=True)
        else:
            self.current_state = self.env.reset(reset_traffic=True)
        self.initial_endpoints = (self.env.taxi_pos, self.env.passenger_pos, self.env.dest_pos)
        self.initial_traffic = self.env.traffic_mgr.traffic_grid.copy()
        self.total_reward = 0.0
        self.history = [self.env.taxi_pos]
        return self.get_payload()

    def restart(self):
        """Reset taxi back to original starting position for the same mission, keeping traffic intact."""
        self.current_state = self.env.reset(fixed_endpoints=self.initial_endpoints, reset_traffic=False)
        self.env.traffic_mgr.traffic_grid = self.initial_traffic.copy()
        self.total_reward = 0.0
        self.history = [self.env.taxi_pos]
        return self.get_payload()

    def step(self):
        if self.active_policy == "q_learning":
            action = self.agent.select_action(self.current_state, explore=False)
        elif self.active_policy == "shortest_path":
            action = self.shortest_agent.select_action(self.current_state, env=self.env)
        else:
            action = self.random_agent.select_action(self.current_state)

        next_state, reward, term, trunc, info = self.env.step(action)
        self.current_state = next_state
        self.total_reward += reward
        self.history.append(self.env.taxi_pos)

        payload = self.get_payload()
        payload["terminated"] = term
        payload["truncated"] = trunc
        payload["event"] = info.get("event", "move")
        payload["last_action"] = int(action)
        payload["last_reward"] = reward
        return payload

    def get_payload(self):
        rows, cols = self.env.grid.rows, self.env.grid.cols
        return {
            "rows": rows,
            "cols": cols,
            "taxi_pos": self.env.taxi_pos,
            "passenger_pos": self.env.passenger_pos,
            "dest_pos": self.env.dest_pos,
            "passenger_picked": bool(self.env.passenger_picked),
            "obstacles": self.env.grid.obstacles.tolist(),
            "traffic": self.env.traffic_mgr.traffic_grid.tolist(),
            "stations": self.env.grid.stations,
            "total_reward": round(self.total_reward, 1),
            "steps": self.env.steps_taken,
            "history": self.history,
            "active_policy": self.active_policy
        }


SIM = SimulationManager()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TransitRL • Taxi Route Optimization & Dispatch Console</title>
    <style>
        :root {
            --bg-base: #0b0f17;
            --bg-surface: #121824;
            --bg-elevated: #182030;
            --bg-subtle: #1e293b;
            --border: #263348;
            --border-active: #3b82f6;
            --text-main: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-dim: #64748b;
            --accent-cyan: #0284c7;
            --accent-blue: #2563eb;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-sans);
            background: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.45;
            -webkit-font-smoothing: antialiased;
        }

        /* Top Navigation Bar */
        header {
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .brand-badge {
            background: #1e293b;
            border: 1px solid var(--border);
            color: #38bdf8;
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 0.8rem;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }
        .brand-info h1 {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.01em;
        }
        .brand-info p {
            font-size: 0.76rem;
            color: var(--text-dim);
            letter-spacing: 0.01em;
        }
        .nav-status {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .dot-live {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }

        /* Layout Grid */
        main {
            flex: 1;
            display: grid;
            grid-template-columns: minmax(560px, 1fr) 390px;
            gap: 1.5rem;
            padding: 1.5rem 2rem;
            max-width: 1440px;
            width: 100%;
            margin: 0 auto;
        }

        /* Cards & Containers */
        .panel {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        .panel-header {
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(18, 24, 36, 0.5);
        }
        .panel-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }
        .panel-body {
            padding: 1.25rem;
        }

        /* Canvas Viewport */
        .viewport-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1.25rem;
        }
        #cityCanvas {
            background: #0d121c;
            border: 1px solid var(--border);
            border-radius: 8px;
            display: block;
            image-rendering: pixelated;
        }

        /* Map Legend */
        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 0.78rem;
            color: var(--text-secondary);
            width: 100%;
            max-width: 560px;
            justify-content: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .legend-swatch {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }

        /* Telemetry Cards */
        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .telemetry-cell {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem;
        }
        .telemetry-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-dim);
            font-weight: 600;
        }
        .telemetry-value {
            font-family: var(--font-mono);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
            color: var(--text-main);
        }

        /* Form Inputs & Controls */
        .form-section {
            margin-bottom: 1.25rem;
        }
        .section-label {
            display: block;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }
        .dim-input-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .dim-box {
            flex: 1;
            display: flex;
            align-items: center;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.4rem 0.6rem;
        }
        .dim-box span {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-dim);
            margin-right: 0.4rem;
            text-transform: uppercase;
        }
        .dim-box input {
            width: 100%;
            background: transparent;
            border: none;
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.95rem;
            font-weight: 700;
            outline: none;
            text-align: center;
        }
        select, input[type=range] {
            width: 100%;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.6rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: var(--font-sans);
            outline: none;
        }
        select:focus, .dim-box:focus-within {
            border-color: var(--border-active);
        }

        /* Segmented Policy Selector */
        .policy-group {
            display: flex;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 3px;
            gap: 3px;
        }
        .policy-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.5rem 0.2rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .policy-btn.active {
            background: #2563eb;
            color: #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }

        /* Action Buttons */
        .btn-row {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }
        button.btn {
            cursor: pointer;
            border: 1px solid transparent;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.65rem 1rem;
            transition: background 0.15s, border-color 0.15s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }
        .btn-primary {
            background: #2563eb;
            color: white;
            flex: 1;
        }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-stop {
            background: #dc2626;
            color: white;
            flex: 1;
        }
        .btn-stop:hover { background: #b91c1c; }
        .btn-outline {
            background: var(--bg-elevated);
            border-color: var(--border);
            color: var(--text-main);
            flex: 1;
        }
        .btn-outline:hover {
            background: #222d40;
            border-color: #3b4d68;
        }
        .btn-build {
            background: #3b82f6;
            color: white;
            padding: 0.55rem 1rem;
            font-size: 0.82rem;
        }
        .btn-build:hover { background: #2563eb; }

        /* Status Callout Banner */
        .mission-status-bar {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 1.25rem;
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #38bdf8;
        }
    </style>
</head>
<body>
    <header>
        <div class="nav-brand">
            <span class="brand-badge">TR-RL</span>
            <div class="brand-info">
                <h1>TransitRL Dispatch Console</h1>
                <p>Urban Taxi Route Optimization • Tabular Q-Learning Engine</p>
            </div>
        </div>
        <div class="nav-status">
            <div class="status-pill"><span class="dot-live"></span> Engine Online</div>
            <div class="status-pill" id="navGridDim" style="font-family: var(--font-mono);">8 × 8 Grid</div>
        </div>
    </header>

    <main>
        <!-- Viewport Left Column -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">City Spatial Matrix & Path Trajectory</span>
                <span id="badgeTarget" style="font-size: 0.76rem; font-family: var(--font-mono); color: #38bdf8;">TARGET: PASSENGER PICKUP</span>
            </div>
            <div class="viewport-wrapper">
                <canvas id="cityCanvas" width="560" height="560"></canvas>
                <div class="legend-row">
                    <div class="legend-item"><div class="legend-swatch" style="background: #f59e0b;"></div> Taxi Vehicle</div>
                    <div class="legend-item"><div class="legend-swatch" style="background: #10b981;"></div> Pickup Station</div>
                    <div class="legend-item"><div class="legend-swatch" style="background: #f43f5e;"></div> Destination Hub</div>
                    <div class="legend-item"><div class="legend-swatch" style="background: #1e293b; border: 1px solid #334155;"></div> Building Block</div>
                    <div class="legend-item"><div class="legend-swatch" style="background: #b45309;"></div> Moderate Traffic</div>
                    <div class="legend-item"><div class="legend-swatch" style="background: #991b1b;"></div> Heavy Congestion</div>
                </div>
            </div>
        </div>

        <!-- Telemetry & Operations Panel Right Column -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Mission Control & Metrics</span>
            </div>
            <div class="panel-body">
                <!-- Telemetry 2x2 Grid -->
                <div class="telemetry-grid">
                    <div class="telemetry-cell">
                        <div class="telemetry-label">Cumulative Return</div>
                        <div class="telemetry-value" id="statReward" style="color: #10b981;">0.0</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-label">Step Count</div>
                        <div class="telemetry-value" id="statSteps">0</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-label">Vehicle Position</div>
                        <div class="telemetry-value" id="statPos" style="font-size: 1.1rem; color: #93c5fd;">[0, 0]</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-label">Route Status</div>
                        <div class="telemetry-value" id="statState" style="font-size: 0.95rem; color: #facc15;">Searching</div>
                    </div>
                </div>

                <!-- Grid Dimension Builder -->
                <div class="form-section">
                    <label class="section-label">City Grid Dimensions (Rows × Cols: 4–15)</label>
                    <div class="dim-input-group">
                        <div class="dim-box">
                            <span>R</span>
                            <input type="number" id="inputRows" min="4" max="15" value="8">
                        </div>
                        <span style="color: var(--text-dim); font-weight: bold;">×</span>
                        <div class="dim-box">
                            <span>C</span>
                            <input type="number" id="inputCols" min="4" max="15" value="8">
                        </div>
                        <button class="btn btn-build" onclick="generateCity()">Rebuild Grid</button>
                    </div>
                </div>

                <!-- Routing Policy Selector -->
                <div class="form-section">
                    <label class="section-label">Active Routing Policy</label>
                    <div class="policy-group">
                        <button class="policy-btn active" id="btnPolQ" onclick="setPolicy('q_learning')">Q-Learning (Optimal)</button>
                        <button class="policy-btn" id="btnPolBFS" onclick="setPolicy('shortest_path')">Shortest Path (BFS)</button>
                        <button class="policy-btn" id="btnPolRnd" onclick="setPolicy('random')">Random</button>
                    </div>
                </div>

                <!-- Scenario Selection -->
                <div class="form-section">
                    <label class="section-label">Transit Hub Mission Preset</label>
                    <select id="missionSelect" onchange="applyMission()">
                        <option value="random">Randomized Pickup & Destination</option>
                        <option value="0,1">Hub 0 (Airport) -> Hub 1 (Downtown)</option>
                        <option value="1,2">Hub 1 (Downtown) -> Hub 2 (Central)</option>
                        <option value="2,3">Hub 2 (Central) -> Hub 3 (Suburb)</option>
                        <option value="3,0">Hub 3 (Suburb) -> Hub 0 (Airport)</option>
                        <option value="0,2">Hub 0 (Airport) -> Hub 2 (Central)</option>
                    </select>
                </div>

                <!-- Playback Speed -->
                <div class="form-section">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <label class="section-label" style="margin-bottom: 0;">Step Delay</label>
                        <span id="speedVal" style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">200 ms</span>
                    </div>
                    <input type="range" id="speedSlider" min="50" max="600" value="200" oninput="updateSpeed(this.value)">
                </div>

                <!-- Execution Controls -->
                <div class="btn-row">
                    <button class="btn btn-primary" id="btnPlay" onclick="togglePlay()">Start Simulation</button>
                    <button class="btn btn-outline" onclick="stepSim()">Single Step</button>
                </div>
                <div class="btn-row">
                    <button class="btn btn-outline" style="width: 100%;" onclick="restartSim()">Replay Scenario (Keep Traffic)</button>
                </div>
                <div class="btn-row">
                    <button class="btn btn-outline" style="width: 100%;" onclick="resetSim()">New Random Mission</button>
                </div>

                <!-- Real-time Status Callout -->
                <div class="mission-status-bar" id="missionStatus">
                    <span class="status-indicator" id="statusDot"></span>
                    <span id="statusText">System initialized and ready for deployment.</span>
                </div>
            </div>
        </div>
    </main>

    <script>
        let isPlaying = false;
        let playInterval = null;
        let simSpeed = 200;
        let currentData = null;
        let activePolicy = 'q_learning';

        const canvas = document.getElementById('cityCanvas');
        const ctx = canvas.getContext('2d');

        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                currentData = await res.json();
                activePolicy = currentData.active_policy || 'q_learning';
                syncPolicyUI();
                render();
                updateUI();
            } catch (e) {
                console.error("Failed to fetch state:", e);
            }
        }

        async function stepSim() {
            try {
                const res = await fetch('/api/step');
                currentData = await res.json();
                render();
                updateUI();
                if (currentData.terminated || currentData.truncated) {
                    stopPlay();
                }
            } catch (e) {
                console.error("Step error:", e);
                stopPlay();
            }
        }

        async function restartSim() {
            stopPlay();
            const res = await fetch('/api/restart');
            currentData = await res.json();
            render();
            updateUI();
            document.getElementById('statusText').textContent = 'Taxi reset to starting hub; traffic conditions strictly preserved.';
        }

        async function resetSim() {
            stopPlay();
            const res = await fetch('/api/reset');
            currentData = await res.json();
            render();
            updateUI();
            document.getElementById('statusText').textContent = 'New random mission initialized.';
        }

        async function generateCity() {
            stopPlay();
            const r = Math.max(4, Math.min(15, parseInt(document.getElementById('inputRows').value) || 8));
            const c = Math.max(4, Math.min(15, parseInt(document.getElementById('inputCols').value) || 8));
            document.getElementById('statusText').textContent = `Synthesizing ${r}x${c} city grid topology & pre-seeding potentials...`;
            const res = await fetch(`/api/generate_city?rows=${r}&cols=${c}`);
            currentData = await res.json();
            updateMissionDropdown(currentData.stations);
            render();
            updateUI();
            document.getElementById('statusText').textContent = `New ${r}x${c} topology built; optimal route policy calibrated.`;
        }

        function updateMissionDropdown(stations) {
            if (!stations || stations.length < 2) return;
            const sel = document.getElementById('missionSelect');
            sel.innerHTML = '<option value="random">Randomized Pickup & Destination</option>';
            for (let i = 0; i < stations.length; i++) {
                const next = (i + 1) % stations.length;
                sel.innerHTML += `<option value="${i},${next}">Hub ${i} -> Hub ${next}</option>`;
            }
            if (stations.length >= 3) {
                sel.innerHTML += '<option value="0,2">Hub 0 -> Hub 2</option>';
            }
        }

        async function applyMission() {
            stopPlay();
            const val = document.getElementById('missionSelect').value;
            let url = '/api/reset';
            if (val !== 'random') {
                const parts = val.split(',');
                url += `?start=${parts[0]}&dest=${parts[1]}`;
            }
            const res = await fetch(url);
            currentData = await res.json();
            render();
            updateUI();
        }

        async function setPolicy(pol) {
            activePolicy = pol;
            await fetch(`/api/policy?set=${pol}`);
            syncPolicyUI();
        }

        function syncPolicyUI() {
            document.getElementById('btnPolQ').className = 'policy-btn ' + (activePolicy === 'q_learning' ? 'active' : '');
            document.getElementById('btnPolBFS').className = 'policy-btn ' + (activePolicy === 'shortest_path' ? 'active' : '');
            document.getElementById('btnPolRnd').className = 'policy-btn ' + (activePolicy === 'random' ? 'active' : '');
        }

        function togglePlay() {
            if (isPlaying) {
                stopPlay();
            } else {
                startPlay();
            }
        }

        function startPlay() {
            isPlaying = true;
            const btn = document.getElementById('btnPlay');
            btn.textContent = 'Pause Simulation';
            btn.className = 'btn btn-stop';
            playInterval = setInterval(stepSim, simSpeed);
        }

        function stopPlay() {
            isPlaying = false;
            clearInterval(playInterval);
            const btn = document.getElementById('btnPlay');
            btn.textContent = 'Start Simulation';
            btn.className = 'btn btn-primary';
        }

        function updateSpeed(val) {
            simSpeed = parseInt(val);
            document.getElementById('speedVal').textContent = val + ' ms';
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(stepSim, simSpeed);
            }
        }

        function updateUI() {
            if (!currentData) return;

            // Header dim
            document.getElementById('navGridDim').textContent = `${currentData.rows} × ${currentData.cols} Grid`;

            // Telemetry
            const rewEl = document.getElementById('statReward');
            rewEl.textContent = (currentData.total_reward >= 0 ? '+' : '') + currentData.total_reward.toFixed(1);
            rewEl.style.color = currentData.total_reward >= 0 ? '#10b981' : '#f43f5e';

            document.getElementById('statSteps').textContent = currentData.steps;

            if (currentData.taxi_pos) {
                document.getElementById('statPos').textContent = `[${currentData.taxi_pos[0]}, ${currentData.taxi_pos[1]}]`;
            }

            const stateEl = document.getElementById('statState');
            const targetBadge = document.getElementById('badgeTarget');
            const dot = document.getElementById('statusDot');
            const stText = document.getElementById('statusText');

            if (currentData.terminated) {
                stateEl.textContent = 'Completed';
                stateEl.style.color = '#10b981';
                targetBadge.textContent = 'MISSION ACCOMPLISHED';
                targetBadge.style.color = '#10b981';
                dot.style.background = '#10b981';
                stText.textContent = 'Destination reached successfully. Passenger transit completed.';
            } else if (currentData.truncated) {
                stateEl.textContent = 'Timeout';
                stateEl.style.color = '#f43f5e';
                targetBadge.textContent = 'STEP LIMIT EXCEEDED';
                targetBadge.style.color = '#f43f5e';
                dot.style.background = '#f43f5e';
                stText.textContent = 'Navigation exceeded step horizon without reaching destination.';
            } else if (currentData.passenger_picked) {
                stateEl.textContent = 'In Transit';
                stateEl.style.color = '#38bdf8';
                targetBadge.textContent = 'TARGET: DESTINATION HUB';
                targetBadge.style.color = '#f43f5e';
                dot.style.background = '#38bdf8';
                stText.textContent = 'Passenger on board; routing vehicle to final destination hub.';
            } else {
                stateEl.textContent = 'En Route';
                stateEl.style.color = '#facc15';
                targetBadge.textContent = 'TARGET: PASSENGER PICKUP';
                targetBadge.style.color = '#10b981';
                dot.style.background = '#facc15';
                stText.textContent = 'Navigating towards passenger pickup coordinates.';
            }
        }

        function render() {
            if (!currentData) return;
            const rows = currentData.rows;
            const cols = currentData.cols;
            const cellW = canvas.width / cols;
            const cellH = canvas.height / rows;
            const minCell = Math.min(cellW, cellH);

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Draw Streets, Buildings & Traffic Congestion
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x = c * cellW;
                    const y = r * cellH;

                    if (currentData.obstacles[r][c]) {
                        // Architectural building block with diagonal hatch
                        ctx.fillStyle = '#141a24';
                        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
                        ctx.strokeStyle = '#222c3d';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);

                        // Subtle diagonal cross
                        ctx.beginPath();
                        ctx.strokeStyle = '#1d2536';
                        ctx.moveTo(x + 2, y + 2);
                        ctx.lineTo(x + cellW - 2, y + cellH - 2);
                        ctx.moveTo(x + cellW - 2, y + 2);
                        ctx.lineTo(x + 2, y + cellH - 2);
                        ctx.stroke();
                    } else {
                        // Street cell
                        let roadColor = '#101622';
                        let borderColor = '#1c2536';
                        let tag = null;
                        let tagColor = '';

                        if (currentData.traffic[r][c] === 2) {
                            roadColor = '#2b1015';
                            borderColor = '#4d1c24';
                            tag = 'HIGH';
                            tagColor = '#f87171';
                        } else if (currentData.traffic[r][c] === 1) {
                            roadColor = '#271b10';
                            borderColor = '#4a2f16';
                            tag = 'MED';
                            tagColor = '#fbbf24';
                        }

                        ctx.fillStyle = roadColor;
                        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
                        ctx.strokeStyle = borderColor;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);

                        // Traffic label tag if cell is congested
                        if (tag && minCell >= 24) {
                            ctx.fillStyle = tagColor;
                            ctx.font = `600 ${Math.max(8, Math.floor(minCell * 0.22))}px ui-monospace, monospace`;
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(tag, x + cellW / 2, y + cellH / 2);
                        }
                    }
                }
            }

            // 2. Draw Transit Station Waypoint Markers
            if (currentData.stations) {
                currentData.stations.forEach((st, idx) => {
                    const sx = st[1] * cellW + cellW / 2;
                    const sy = st[0] * cellH + cellH / 2;
                    const stRad = Math.max(7, minCell * 0.3);

                    ctx.beginPath();
                    ctx.arc(sx, sy, stRad, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(51, 65, 85, 0.4)';
                    ctx.fill();
                    ctx.strokeStyle = '#475569';
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    if (minCell >= 22) {
                        ctx.fillStyle = '#94a3b8';
                        ctx.font = `600 ${Math.max(8, Math.floor(minCell * 0.2))}px ui-monospace, monospace`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(`S${idx}`, sx, sy);
                    }
                });
            }

            // 3. Draw Vehicle Trajectory Path
            if (currentData.history && currentData.history.length > 1) {
                ctx.beginPath();
                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = Math.max(2, Math.min(3.5, minCell * 0.08));
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                for (let i = 0; i < currentData.history.length; i++) {
                    const [hr, hc] = currentData.history[i];
                    const hx = hc * cellW + cellW / 2;
                    const hy = hr * cellH + cellH / 2;
                    if (i === 0) ctx.moveTo(hx, hy);
                    else ctx.lineTo(hx, hy);
                }
                ctx.stroke();

                // Small trajectory breadcrumb dots
                ctx.fillStyle = '#0284c7';
                for (let i = 0; i < currentData.history.length - 1; i++) {
                    const [hr, hc] = currentData.history[i];
                    const hx = hc * cellW + cellW / 2;
                    const hy = hr * cellH + cellH / 2;
                    ctx.beginPath();
                    ctx.arc(hx, hy, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            const rad = Math.max(6, minCell * 0.32);

            // 4. Draw Passenger Pickup Location (Clean Emerald Target)
            if (!currentData.passenger_picked) {
                const [pr, pc] = currentData.passenger_pos;
                const px = pc * cellW + cellW / 2;
                const py = pr * cellH + cellH / 2;

                // Outer halo
                ctx.beginPath();
                ctx.arc(px, py, rad * 1.25, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Core marker
                ctx.beginPath();
                ctx.arc(px, py, rad, 0, Math.PI * 2);
                ctx.fillStyle = '#10b981';
                ctx.fill();
                ctx.strokeStyle = '#059669';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Center crosshair / label
                ctx.fillStyle = '#ffffff';
                ctx.font = `700 ${Math.max(9, Math.floor(minCell * 0.3))}px var(--font-sans)`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('P', px, py);
            }

            // 5. Draw Destination Hub (Clean Rose Target Box)
            const [dr, dc] = currentData.dest_pos;
            const dx = dc * cellW + cellW / 2;
            const dy = dr * cellH + cellH / 2;

            // Outer ring
            ctx.beginPath();
            ctx.rect(dx - rad * 1.2, dy - rad * 1.2, rad * 2.4, rad * 2.4);
            ctx.strokeStyle = 'rgba(244, 63, 94, 0.35)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Core square
            ctx.beginPath();
            ctx.rect(dx - rad * 0.9, dy - rad * 0.9, rad * 1.8, rad * 1.8);
            ctx.fillStyle = '#f43f5e';
            ctx.fill();
            ctx.strokeStyle = '#be123c';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.fillStyle = '#ffffff';
            ctx.font = `700 ${Math.max(9, Math.floor(minCell * 0.3))}px var(--font-sans)`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('D', dx, dy);

            // 6. Draw Taxi Vehicle (Sleek Vector Vehicle, NO childish emojis)
            const [tr, tc] = currentData.taxi_pos;
            const tx = tc * cellW + cellW / 2;
            const ty = tr * cellH + cellH / 2;
            const vWidth = Math.max(14, minCell * 0.65);
            const vHeight = Math.max(14, minCell * 0.65);

            ctx.save();
            ctx.translate(tx, ty);

            // Vehicle chassis shadow
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.beginPath();
            ctx.roundRect(-vWidth / 2 + 1, -vHeight / 2 + 2, vWidth, vHeight, 4);
            ctx.fill();

            // Vehicle body (Yellow taxi cab)
            ctx.fillStyle = '#f59e0b';
            ctx.strokeStyle = '#d97706';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.roundRect(-vWidth / 2, -vHeight / 2, vWidth, vHeight, 4);
            ctx.fill();
            ctx.stroke();

            // Windshield / Cabin Glass
            ctx.fillStyle = '#0f172a';
            ctx.beginPath();
            ctx.roundRect(-vWidth * 0.32, -vHeight * 0.32, vWidth * 0.64, vHeight * 0.64, 2);
            ctx.fill();

            // Dispatch Roof Light Indicator
            // Green if passenger on-board, Amber if en-route
            ctx.fillStyle = currentData.passenger_picked ? '#10b981' : '#fef08a';
            ctx.beginPath();
            ctx.roundRect(-vWidth * 0.15, -vHeight * 0.15, vWidth * 0.3, vHeight * 0.3, 1.5);
            ctx.fill();

            ctx.restore();
        }

        fetchState();
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    """Handle REST API calls and serve interactive dashboard HTML."""

    def log_message(self, format, *args):
        # Silence default terminal request logs for cleaner console
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/state":
            self.send_json(SIM.get_payload())

        elif path == "/api/step":
            payload = SIM.step()
            self.send_json(payload)

        elif path == "/api/reset":
            if "start" in query and "dest" in query:
                pair = (int(query["start"][0]), int(query["dest"][0]))
                payload = SIM.reset(station_pair=pair)
            else:
                payload = SIM.reset()
            self.send_json(payload)

        elif path == "/api/restart":
            payload = SIM.restart()
            self.send_json(payload)

        elif path == "/api/policy":
            if "set" in query:
                SIM.active_policy = query["set"][0]
            self.send_json({"active_policy": SIM.active_policy})

        elif path == "/api/generate_city":
            rows = int(query.get("rows", query.get("size", ["8"]))[0])
            cols = int(query.get("cols", [str(rows)])[0])
            payload = SIM.generate_world(rows, cols)
            self.send_json(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, RequestHandler)
    url = f"http://localhost:{port}"

    print("========================================================================")
    print("      RL TAXI ROUTE OPTIMIZATION - INTERACTIVE WEB DASHBOARD")
    print("========================================================================")
    print(f"Server started successfully!")
    print(f"Open in your browser: {url}")
    print("Press Ctrl + C in terminal to stop.")
    print("========================================================================")

    # Automatically open default web browser
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    run_server(5000)
