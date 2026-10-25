# RL-Based Taxi Route Optimization in a Dynamic Grid City

**Course Continuous Assessment (CA) — Reinforcement Learning**  
*Department of Computer Science & Engineering*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-from--scratch-green.svg)](https://numpy.org/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-Tabular%20Q--Learning-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 1. Project Title & Overview

**"RL-Based Taxi Route Optimization in a Dynamic Grid City"**

This project investigates autonomous navigation and sequential decision-making for a simulated taxi agent operating in a complex urban grid. The taxi must pick up waiting passengers and deliver them to designated destinations while negotiating static obstacles (roadblocks, dividers) and stochastic dynamic traffic congestion zones (`LOW`, `MEDIUM`, `HIGH`).

Unlike standard benchmark setups that rely on Gymnasium's canned `Taxi-v3` or high-level libraries (e.g., Stable-Baselines3), this project implements **both the simulation environment and the tabular Q-Learning algorithm completely from scratch** using pure Python and NumPy.

---

## 2. Problem Statement

Urban dispatch and navigation present significant challenges:
1. **Multi-Stage Goal Progression:** The agent must execute an ordered sequence of sub-goals: first localize and navigate to the passenger, trigger pickup, and subsequently route to the passenger's destination.
2. **Constrained Spatial Topology:** Movement is restricted to viable road networks punctuated by blocked cells (buildings, construction barriers).
3. **Stochastic Traffic Delays:** Certain road segments suffer variable congestion that penalizes travel time. An optimal route is not merely the geographically shortest path, but the path that minimizes expected cumulative time and traffic penalties.

---

## 3. Motivation & Why Reinforcement Learning?

Classical pathfinding algorithms (such as Dijkstra’s Algorithm or A\* Search) compute optimal routes based on a static, deterministic graph with pre-known edge costs. In real-world urban transit:
- **Traffic is Stochastic and Dynamic:** Congestion levels fluctuate unpredictably.
- **Model-Free Adaptation:** An autonomous taxi cannot assume perfect analytical models of future traffic shifts.
- **Sequential Optimization:** Reinforcement Learning formulates navigation as an ongoing trial-and-error interaction where the agent learns a robust, expected-cost policy that balances distance against congestion risk.

---

## 4. Markov Decision Process (MDP) Formulation

The navigation problem is formally cast as a finite, discounted Markov Decision Process defined by the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

```mermaid
graph LR
    subgraph Agent ["Taxi Agent"]
        Q["Q-Table / Policy pi"]
    end
    subgraph Environment ["Dynamic City Grid"]
        State["State: (tx, ty, px, py, dx, dy, picked)"]
        Physics["Obstacles & Traffic Grid"]
    end
    Agent -- "Action (UP, DOWN, LEFT, RIGHT)" --> Environment
    Environment -- "Reward R & Next State S'" --> Agent
```

### A. State Space ($\mathcal{S}$)
The global state is represented as a discrete 7-tuple:
$$\mathcal{S} = (t_x, t_y, p_x, p_y, d_x, d_y, \text{picked})$$
- $(t_x, t_y) \in [0, N-1] \times [0, M-1]$: Current row and column of the Taxi.
- $(p_x, p_y)$: Passenger coordinate location.
- $(d_x, d_y)$: Destination drop-off coordinate location.
- $\text{picked} \in \{0, 1\}$: Binary status indicator ($0$ = passenger waiting at station, $1$ = passenger inside taxi).

#### State Space Abstraction (Tackling the Curse of Dimensionality)
To ensure fast convergence without exponential state explosion:
- When $\text{picked} = 0$, the taxi's objective is strictly to reach $(p_x, p_y)$; the destination $(d_x, d_y)$ has no physical bearing on pickup.
- When $\text{picked} = 1$, the taxi's objective is to reach $(d_x, d_y)$; the historical pickup location is irrelevant.

This canonical state abstraction reduces the effective state space to manageable dimensions ($\sim 400$ to $1{,}200$ reachable states on an $8 \times 8$ or $10 \times 10$ city), ensuring complete tabular convergence in fewer than 1,500 episodes.

### B. Action Space ($\mathcal{A}$)
The agent operates over a discrete action set:
- `0 = UP` (Move North: $\Delta r = -1, \Delta c = 0$)
- `1 = DOWN` (Move South: $\Delta r = +1, \Delta c = 0$)
- `2 = LEFT` (Move West: $\Delta r = 0, \Delta c = -1$)
- `3 = RIGHT` (Move East: $\Delta r = 0, \Delta c = +1$)
- `4 = WAIT` (Optional: remain stationary during transient peak congestion)

The environment strictly enforces physical movement: the agent cannot teleport across barriers or grid edges.

---

## 5. Reward Function Architecture

The reward structure is engineered to reward speed and safety while penalizing risky or illegal transitions:

| Event / State Transition | Reward Value | Rationale |
| :--- | :---: | :--- |
| **Normal Step Move** | `-1.0` | Incurs a step cost to incentivize shortest-path routing. |
| **Grid Boundary Collision** | `-5.0` | Penalizes attempts to drive off the grid boundaries; taxi remains stationary. |
| **Obstacle Collision** | `-5.0` | Penalizes collisions with dividers/buildings; taxi remains in place. |
| **LOW Traffic Cell** | `0.0` | Nominal open road; no additional delay. |
| **MEDIUM Traffic Cell (`T`)** | `-2.0` | Simulates moderate congestion delay (Total step cost: `-3.0`). |
| **HIGH Traffic Cell (`H`)** | `-4.0` | Simulates severe gridlock delay (Total step cost: `-5.0`). |
| **Passenger Pickup (`P`)** | `+20.0` | Intermediate sub-goal reward to provide strong reward shaping toward pickup. |
| **Destination Delivery (`D`)** | `+100.0` | Primary terminal milestone upon successful delivery with passenger on-board. |
| **Timeout (Max Steps Exceeded)**| `-10.0` | Terminal truncation penalty for failing to deliver within the budget. |

---

## 6. Q-Learning Algorithm (Implemented from Scratch)

Q-Learning is a model-free, off-policy Temporal Difference (TD) control algorithm that directly approximates the optimal action-value function $Q^*(s, a)$.

### Mathematical Update Rule
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

Where:
- $\alpha \in (0, 1]$ is the **Learning Rate** (default: `0.10`).
- $\gamma \in [0, 1)$ is the **Discount Factor** (default: `0.99`).
- $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$ is the **TD Target**.
- $\delta_t = \text{Target} - Q(S_t, A_t)$ is the **Temporal Difference Error**.

### Exploration vs. Exploitation ($\epsilon$-Greedy Policy)
To explore uncharted city roads while exploiting high-reward routes:
$$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

Epsilon decays exponentially across episodes:
$$\epsilon_{k+1} = \max(\epsilon_{\min},\, \epsilon_k \times \epsilon_{\text{decay}})$$
With $\epsilon_0 = 1.0$, $\epsilon_{\text{decay}} = 0.996$, and $\epsilon_{\min} = 0.05$.

---

## 7. Baseline Algorithms & Academic Benchmarking

To ensure rigorous academic evaluation, our Q-Learning policy is evaluated against two baseline architectures across 50 identical test scenarios:

1. **Random Policy Baseline:** Selects actions uniformly at random ($\pi(a \mid s) = \frac{1}{|\mathcal{A}|}$). Represents an unguided random walk.
2. **Shortest-Path (BFS) Baseline:** Computes the shortest geometric topological path using Breadth-First Search (BFS) around obstacles. Serves as a standard classical routing benchmark.
3. **Q-Learning Agent (Ours):** Greedy policy derived from the learned optimal Q-table without exploration.

---

## 8. Experimental Results & Visualizations

### A. Training Convergence Curves
Generated from actual training runs:
- **Reward vs Episode:** Clear ascent from large negative penalties to consistent positive returns ($+70$ to $+85$).
- **Steps to Delivery:** Reduction from $\sim 95$ steps down to optimal $\sim 15\text{--}20$ steps.
- **Success Rate:** Stabilizes at **$98\%\text{--}100\%$** delivery reliability.
- **Bellman TD-Error:** Smooth convergence toward 0 as Q-values stabilize.

![Training Curves](results/training_curves.png)

### B. Learned Optimal Route Trajectory
The route visualizer plots the learned taxi trajectory from pickup to drop-off:

```
Learned Route Trace (8x8 City Grid)
Total Reward: +85.0 | Steps: 16 | Delivered: True
Legend: S=Start, P=Passenger, D=Destination, #=Obstacle, T/H=Traffic
-------------------
  # S > > v # . .
  . . . T v # . .
  # . T H v . T H
  T # . . v . T H
  T T . T v # . .
  . T . . v . . .
  . . . . v . . .
  . . # . > > > D
-------------------
```

![Learned Route](results/learned_route.png)

### C. Benchmark Comparison Across Policies

| Policy | Mean Cumulative Reward | Mean Steps to Delivery | Success Rate (%) |
| :--- | :---: | :---: | :---: |
| **Random Policy** | $-232.5 \pm 76.2$ | $96.6 \pm 12.7$ | $12.0\%$ |
| **Shortest-Path Baseline** | $+93.4 \pm 10.4$ | $15.2 \pm 4.7$ | $100.0\%$ |
| **Q-Learning Agent (Ours)** | $+86.8 \pm 14.1$ | $16.8 \pm 5.2$ | **$98.0\%$** |

![Baseline Comparison](results/baseline_comparison.png)

---

## 9. Project Directory Structure

```text
taxi-rl/
│
├── environment/
│   ├── __init__.py
│   ├── grid.py            # City grid topology, obstacles, BFS reachability
│   ├── traffic.py         # Multi-tier traffic manager (LOW, MED, HIGH)
│   └── taxi_env.py        # Custom Gymnasium-like MDP Environment
│
├── agent/
│   ├── __init__.py
│   └── q_learning.py      # Tabular Q-Learning implemented from scratch
│
├── baselines/
│   ├── __init__.py
│   ├── random_policy.py   # Uniform random baseline
│   └── shortest_path.py   # BFS shortest-path graph baseline
│
├── training/
│   ├── __init__.py
│   └── train.py           # Training loop, metrics logging, moving averages
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py        # Multi-policy benchmark evaluator
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py           # Matplotlib publication curves & bar charts
│   └── route_visualizer.py# Trajectory arrow renderer & grid maps
│
├── experiments/
│   ├── __init__.py
│   └── run_experiments.py # Hyperparameter studies (alpha, gamma, decay, traffic)
│
├── results/               # Generated figures, CSV data, and trained model files
│   ├── training_curves.png
│   ├── learned_route.png
│   ├── baseline_comparison.png
│   ├── training_metrics.csv
│   └── q_table_trained.pkl
│
├── config.py              # Centralized dataclass configurations
├── main.py                # Single-entrypoint executable pipeline
├── test_phases_1_to_5.py  # Unit and integration test suite
├── requirements.txt       # Dependencies
├── VIVA.md                # Viva voce oral examination Q&A guide
└── README.md              # Project documentation
```

---

## 10. How to Run the Project

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run Complete End-to-End Pipeline
Executes training, generates all graphs, runs baseline benchmarks, and plots the learned route:
```bash
python main.py
```

### 2. Interactive Python Frontends (GUI & Web Dashboard)
We provide **two** interactive frontend interfaces for live presentations and viva demonstrations:

- **Desktop GUI (Tkinter - Native Window):**
  ```bash
  python gui.py
  ```
  Opens an immediate desktop GUI window featuring live animated taxi routing, real-time telemetry, step-by-step playback, speed control, preset mission selector, and a retrain button.

- **Modern Browser Dashboard (Zero External Dependencies):**
  ```bash
  python web_app.py
  ```
  Launches a modern dark-mode web application at `http://localhost:5000` with HTML5 Canvas simulation, traffic heatmap, policy switcher (Q-Learning vs BFS vs Random), and station presets.

### 2. Custom Command-Line Options
```bash
# Train on a 10x10 grid with dynamic traffic for 2000 episodes
python main.py --grid_size 10 --episodes 2000 --dynamic_traffic

# Run full academic hyperparameter experiments suite (Exps 1 to 5)
python main.py --run_experiments
```

### 3. Run Individual Experiments
```bash
# Run Experiment 1 (Learning rates study)
python experiments/run_experiments.py --exp 1

# Run Experiment 4 (Static vs Dynamic Traffic)
python experiments/run_experiments.py --exp 4
```

### 4. Run Verification Test Suite
```bash
python test_phases_1_to_5.py
```

---

## 11. Limitations & Future Extensions

1. **Tabular State Space Limit:** In extremely large grids ($100 \times 100$), the tabular lookup table grows too large to explore exhaustively.
   - *Future Work:* Implement **Deep Q-Networks (DQN)** or Prioritized Experience Replay to approximate state values continuously.
2. **Partial Observability (POMDP):** If traffic congestion is only visible within a local sensing radius of the taxi, the problem becomes a POMDP.
   - *Future Work:* Integrate recurrent architectures (DRQN) with memory gates.
3. **Multi-Agent Fleet Coordination:** Multiple taxis competing for passengers and sharing road bandwidth.

---

## 12. References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279-292.
- Dietterich, T. G. (2000). *Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition*. JAIR.
