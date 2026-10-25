"""
Zero-Dependency Modern Web Dashboard for Taxi Route Optimization.
Uses Python's standard library `http.server` to serve an interactive HTML5/Canvas UI
with real-time REST API endpoints for stepping, resetting, and policy evaluation.

Launch:
    python web_app.py
Then open http://localhost:5000 in your browser!
"""

import os
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
    <title>RL Taxi Route Optimization | Dynamic City</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(22, 30, 49, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top, #131d35 0%, var(--bg-dark) 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            padding: 1.2rem 2.5rem;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo-group { display: flex; align-items: center; gap: 0.8rem; }
        .logo-group h1 { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
        .badge {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.5rem;
            padding: 1.5rem 2.5rem;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        .canvas-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        #cityCanvas {
            background: #0b1120;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        }
        .legend-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .legend-item { display: flex; align-items: center; gap: 0.4rem; }
        .legend-dot { width: 12px; height: 12px; border-radius: 4px; }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .stat-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.85rem;
        }
        .stat-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; }
        .stat-val { font-size: 1.4rem; font-weight: 800; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace; }
        
        .control-group { margin-bottom: 1.25rem; }
        .control-label { font-size: 0.8rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.4rem; display: block; text-transform: uppercase; }
        select, input[type=range] {
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.65rem 0.8rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
        }
        .btn-row { display: flex; gap: 0.6rem; margin-top: 0.8rem; }
        button {
            cursor: pointer;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-weight: 700;
            font-size: 0.9rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }
        .btn-primary { background: var(--accent-blue); color: white; flex: 1; }
        .btn-primary:hover { background: #2563eb; transform: translateY(-1px); }
        .btn-success { background: var(--accent-green); color: white; flex: 1; }
        .btn-success:hover { background: #059669; }
        .btn-danger { background: var(--accent-red); color: white; flex: 1; }
        .btn-secondary { background: #334155; color: white; flex: 1; }
        .btn-secondary:hover { background: #475569; }
        
        .mission-status {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 1rem;
            font-size: 0.88rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <span style="font-size: 1.6rem;">🚖</span>
            <h1>RL Taxi Route Optimizer</h1>
            <span class="badge">Continuous Assessment</span>
        </div>
        <div>
            <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">Custom Q-Learning (No Gym)</span>
        </div>
    </header>

    <main>
        <div class="card canvas-container">
            <canvas id="cityCanvas" width="560" height="560"></canvas>
            <div class="legend-bar">
                <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div> Taxi (S)</div>
                <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div> Passenger (P)</div>
                <div class="legend-item"><div class="legend-dot" style="background: #ef4444;"></div> Destination (D)</div>
                <div class="legend-item"><div class="legend-dot" style="background: #334155;"></div> Obstacle (#)</div>
                <div class="legend-item"><div class="legend-dot" style="background: #c2410c;"></div> Medium Traffic</div>
                <div class="legend-item"><div class="legend-dot" style="background: #991b1b;"></div> High Traffic</div>
            </div>
        </div>

        <div class="card">
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-title">Cumulative Reward</div>
                    <div class="stat-val" id="statReward" style="color: #22c55e;">0.0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-title">Steps Taken</div>
                    <div class="stat-val" id="statSteps">0</div>
                </div>
            </div>

            <div class="control-group">
                <label class="control-label">Active Policy</label>
                <select id="policySelect" onchange="changePolicy()">
                    <option value="q_learning">Q-Learning Agent (Learned Policy)</option>
                    <option value="shortest_path">Shortest-Path Baseline (BFS)</option>
                    <option value="random">Random Policy Baseline</option>
                </select>
            </div>

            <div class="control-group">
                <label class="control-label">Preset Transit Station Mission</label>
                <select id="missionSelect" onchange="applyMission()">
                    <option value="random">🎲 Random Spawn Locations</option>
                    <option value="0,1">Airport (St 0) -> Downtown (St 1)</option>
                    <option value="1,2">Downtown (St 1) -> Central Station (St 2)</option>
                    <option value="2,3">Central Station (St 2) -> Suburb (St 3)</option>
                    <option value="3,0">Suburb (St 3) -> Airport (St 0)</option>
                    <option value="0,2">Airport (St 0) -> Central Station (St 2)</option>
                </select>
            </div>

            <div class="control-group">
                <label class="control-label">Playback Delay: <span id="speedVal">200ms</span></label>
                <input type="range" id="speedSlider" min="50" max="600" value="200" oninput="updateSpeed(this.value)">
            </div>

            <div class="btn-row">
                <button class="btn-primary" id="btnPlay" onclick="togglePlay()">▶ Auto Play</button>
                <button class="btn-secondary" onclick="stepSim()">⏩ Step</button>
            </div>
            <div class="btn-row">
                <button class="btn-primary" style="width: 100%; background: #0284c7;" onclick="restartSim()">↺ Reset Taxi to Start (Same Mission)</button>
            </div>
            <div class="btn-row">
                <button class="btn-secondary" style="width: 100%;" onclick="resetSim()">🎲 New Random Mission</button>
            </div>

            <div class="mission-status" id="missionStatus">
                <span>📍</span> <span id="statusText">Ready to start navigation</span>
            </div>
        </div>
    </main>

    <script>
        let isPlaying = false;
        let playInterval = null;
        let simSpeed = 200;
        let currentData = null;

        const canvas = document.getElementById('cityCanvas');
        const ctx = canvas.getContext('2d');

        async function fetchState() {
            const res = await fetch('/api/state');
            currentData = await res.json();
            render();
            updateUI();
        }

        async function stepSim() {
            const res = await fetch('/api/step');
            currentData = await res.json();
            render();
            updateUI();
            if (currentData.terminated || currentData.truncated) {
                stopPlay();
            }
        }

        async function restartSim() {
            stopPlay();
            const res = await fetch('/api/restart');
            currentData = await res.json();
            render();
            updateUI();
        }

        async function resetSim() {
            stopPlay();
            const res = await fetch('/api/reset');
            currentData = await res.json();
            render();
            updateUI();
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

        async function changePolicy() {
            const pol = document.getElementById('policySelect').value;
            await fetch(`/api/policy?set=${pol}`);
            if (currentData) currentData.active_policy = pol;
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
            document.getElementById('btnPlay').textContent = '⏸ Pause';
            document.getElementById('btnPlay').className = 'btn-danger';
            playInterval = setInterval(stepSim, simSpeed);
        }

        function stopPlay() {
            isPlaying = false;
            clearInterval(playInterval);
            document.getElementById('btnPlay').textContent = '▶ Auto Play';
            document.getElementById('btnPlay').className = 'btn-primary';
        }

        function updateSpeed(val) {
            simSpeed = parseInt(val);
            document.getElementById('speedVal').textContent = val + 'ms';
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(stepSim, simSpeed);
            }
        }

        function updateUI() {
            if (!currentData) return;
            const rewEl = document.getElementById('statReward');
            rewEl.textContent = currentData.total_reward;
            rewEl.style.color = currentData.total_reward >= 0 ? '#22c55e' : '#f87171';
            document.getElementById('statSteps').textContent = currentData.steps;

            const stText = document.getElementById('statusText');
            if (currentData.terminated) {
                stText.textContent = '🎉 Mission Accomplished! Delivered successfully.';
            } else if (currentData.truncated) {
                stText.textContent = '⚠️ Timeout! Hit 100 max steps.';
            } else if (currentData.passenger_picked) {
                stText.textContent = 'Passenger on-board 🚕 -> Navigating to Destination';
            } else {
                stText.textContent = 'Navigating to Passenger Pickup 👤';
            }
        }

        function render() {
            if (!currentData) return;
            const rows = currentData.rows;
            const cols = currentData.cols;
            const cellSize = canvas.width / cols;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Grid & Cells
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x = c * cellSize;
                    const y = r * cellSize;

                    if (currentData.obstacles[r][c]) {
                        ctx.fillStyle = '#334155';
                        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                        ctx.fillStyle = '#64748b';
                        ctx.font = 'bold 16px Inter';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('#', x + cellSize/2, y + cellSize/2);
                    } else if (currentData.traffic[r][c] === 2) {
                        ctx.fillStyle = '#7f1d1d';
                        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                        ctx.fillStyle = '#fca5a5';
                        ctx.font = 'bold 10px Inter';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('HIGH', x + cellSize/2, y + cellSize/2);
                    } else if (currentData.traffic[r][c] === 1) {
                        ctx.fillStyle = '#7c2d12';
                        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                        ctx.fillStyle = '#fed7aa';
                        ctx.font = 'bold 10px Inter';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText('MED', x + cellSize/2, y + cellSize/2);
                    } else {
                        ctx.fillStyle = '#1e293b';
                        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                    }
                }
            }

            // Draw Trajectory line
            if (currentData.history && currentData.history.length > 1) {
                ctx.beginPath();
                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = 3.5;
                ctx.lineCap = 'round';
                for (let i = 0; i < currentData.history.length; i++) {
                    const [hr, hc] = currentData.history[i];
                    const hx = hc * cellSize + cellSize / 2;
                    const hy = hr * cellSize + cellSize / 2;
                    if (i === 0) ctx.moveTo(hx, hy);
                    else ctx.lineTo(hx, hy);
                }
                ctx.stroke();
            }

            // Draw Passenger
            if (!currentData.passenger_picked) {
                const [pr, pc] = currentData.passenger_pos;
                const px = pc * cellSize + cellSize / 2;
                const py = pr * cellSize + cellSize / 2;
                ctx.beginPath();
                ctx.arc(px, py, cellSize * 0.32, 0, Math.PI * 2);
                ctx.fillStyle = '#10b981';
                ctx.fill();
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = '#059669';
                ctx.stroke();
                ctx.fillStyle = 'white';
                ctx.font = 'bold 14px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('P', px, py);
            }

            // Draw Destination
            const [dr, dc] = currentData.dest_pos;
            const dx = dc * cellSize + cellSize / 2;
            const dy = dr * cellSize + cellSize / 2;
            ctx.beginPath();
            ctx.rect(dx - cellSize * 0.3, dy - cellSize * 0.3, cellSize * 0.6, cellSize * 0.6);
            ctx.fillStyle = '#ef4444';
            ctx.fill();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#b91c1c';
            ctx.stroke();
            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('D', dx, dy);

            // Draw Taxi
            const [tr, tc] = currentData.taxi_pos;
            const tx = tc * cellSize + cellSize / 2;
            const ty = tr * cellSize + cellSize / 2;
            ctx.beginPath();
            ctx.arc(tx, ty, cellSize * 0.35, 0, Math.PI * 2);
            ctx.fillStyle = '#f59e0b';
            ctx.fill();
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#d97706';
            ctx.stroke();
            ctx.font = '20px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(currentData.passenger_picked ? '🚕' : '🚖', tx, ty);
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
