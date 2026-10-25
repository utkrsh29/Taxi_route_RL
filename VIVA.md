# Academic Viva Voce Guide
## Project: RL-Based Taxi Route Optimization in a Dynamic Grid City

This document is prepared specifically for oral examination, project presentation, and professor questioning for university Continuous Assessment.

---

### Core RL Conceptual Questions & Answers

#### Q1: What is the Agent in this project?
**Answer:**
The agent is the autonomous decision-maker—here, the **virtual taxi**. It senses the current state of the city grid, selects actions (moving UP, DOWN, LEFT, RIGHT, or WAIT) based on its policy, and learns from environmental feedback (rewards and penalties) to maximize its cumulative expected return.

---

#### Q2: What is the Environment?
**Answer:**
The environment is our custom-built **dynamic grid city** simulation (`TaxiCityEnv`). It contains:
- Road network (open cells `.`)
- Physical roadblocks / buildings (`#`)
- Congestion zones with variable traffic levels (`LOW`, `MEDIUM`, `HIGH`)
- Passengers (`P`) and destination drop-off points (`D`)
- Transition dynamics (handling valid moves, collisions, and state updates).

---

#### Q3: How is the State represented, and why is this representation chosen?
**Answer:**
The state is represented as a discrete 7-tuple:
$$\mathcal{S} = (\text{taxi\_x},\, \text{taxi\_y},\, \text{pass\_x},\, \text{pass\_y},\, \text{dest\_x},\, \text{dest\_y},\, \text{passenger\_picked})$$
- `(taxi_x, taxi_y)`: Current coordinates of the taxi.
- `(pass_x, pass_y)`: Coordinates where the passenger is waiting.
- `(dest_x, dest_y)`: Destination coordinates where passenger must be dropped.
- `passenger_picked`: Boolean flag ($0$: waiting for pickup, $1$: passenger on-board).

**Why this design?**
It provides complete spatial and task observability, allowing the agent to know whether it should navigate toward the passenger or toward the destination, satisfying the Markov property. By anchoring pickup/dropoff tasks to designated transit hubs, the discrete tabular state space remains compact ($\sim 1{,}500$ to $3{,}200$ reachable states), ensuring convergence within practical training budgets.

---

#### Q4: Why is this problem formulated as a Markov Decision Process (MDP)?
**Answer:**
A problem qualifies as an MDP if it satisfies the **Markov Property**:
$$\mathbb{P}(S_{t+1} = s',\, R_{t+1} = r \mid S_t = s_t,\, A_t = a_t,\, S_{t-1} = s_{t-1},\, \dots,\, S_0 = s_0) = \mathbb{P}(S_{t+1} = s',\, R_{t+1} = r \mid S_t = s_t,\, A_t = a_t)$$
The next state $S_{t+1}$ and reward $R_{t+1}$ depend **only** on the current state $S_t$ and action $A_t$, not on the entire prior history of moves. In our environment, knowing the taxi's present position and passenger status is sufficient to determine all valid transitions and rewards.

An MDP is formally defined by the 5-tuple:
$$(\mathcal{S},\, \mathcal{A},\, \mathcal{P},\, \mathcal{R},\, \gamma)$$
- $\mathcal{S}$: State space
- $\mathcal{A}$: Action space $\{0: \text{UP}, 1: \text{DOWN}, 2: \text{LEFT}, 3: \text{RIGHT}, 4: \text{WAIT}\}$
- $\mathcal{P}(s' \mid s, a)$: State transition probability function
- $\mathcal{R}(s, a, s')$: Reward function
- $\gamma \in [0, 1)$: Discount factor

---

#### Q5: What is the Policy ($\pi$)?
**Answer:**
A policy is the agent's strategy or mapping from states to actions:
- **During Training ($\epsilon$-Greedy Policy):**
  $$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|} & \text{if } a \neq \arg\max_{a'} Q(s, a') \end{cases}$$
- **During Testing (Greedy Policy):**
  $$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$

---

#### Q6: What is a Q-Value ($Q(s, a)$)?
**Answer:**
$Q(s, a)$, or the State-Action Value, represents the **expected cumulative discounted future return** starting from state $s$, taking action $a$, and thereafter following policy $\pi$:
$$Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s,\, A_t = a \right]$$

---

#### Q7: Why Q-Learning? Why model-free?
**Answer:**
1. **Model-Free:** The agent does not need prior mathematical knowledge of the transition probability matrix $\mathcal{P}(s' \mid s, a)$ or analytical traffic flow formulas. It learns purely through trial-and-error interaction with the simulated environment.
2. **Off-Policy:** Q-Learning evaluates and improves the target optimal greedy policy while behaving according to an exploratory $\epsilon$-greedy behavior policy:
   $$\text{Target: } R + \gamma \max_{a'} Q(S_{t+1}, a')$$
   This decouples exploration from the optimal policy being estimated.

---

#### Q8: Write and explain the Q-Learning Update Equation.
**Answer:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

Breaking down each term:
- $Q(S_t, A_t)$: Current estimated value of taking action $A_t$ in state $S_t$.
- $\alpha \in (0, 1]$: **Learning Rate (Step Size)**. Controls how much newly observed experience overrides old knowledge.
- $R_{t+1}$: Immediate scalar reward received after transitioning.
- $\gamma \in [0, 1)$: **Discount Factor**. Balances immediate versus long-term future rewards.
- $\max_{a'} Q(S_{t+1}, a')$: Maximum expected Q-value achievable from the next state.
- $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$: **TD Target** (Bellman optimality projection).
- $\left[ \text{Target} - Q(S_t, A_t) \right]$: **Temporal Difference (TD) Error** ($\delta_t$).

---

#### Q9: What does $\alpha$ (Learning Rate) control?
**Answer:**
- If $\alpha = 0$: Agent learns nothing; Q-values never update.
- If $\alpha = 1$: Agent considers only the most recent experience, completely overwriting prior knowledge (causes catastrophic oscillation in stochastic environments).
- Optimal $\alpha \in [0.05, 0.2]$: Gradually incorporates new evidence while smoothing out stochastic environmental variations (such as fluctuating dynamic traffic).

---

#### Q10: What does $\gamma$ (Discount Factor) control?
**Answer:**
- If $\gamma = 0$: Myopic (short-sighted) agent; only cares about immediate step reward ($R_{t+1}$). It will avoid moving because each step incurs $-1$, failing to reach the $+100$ destination.
- If $\gamma \to 1$ (e.g., $0.99$): Far-sighted agent; values future rewards almost equally to immediate rewards, motivating the agent to invest $-1$ steps to reach the $+20$ pickup and $+100$ delivery terminal goals.

---

#### Q11: Explain Exploration vs. Exploitation and Epsilon Decay.
**Answer:**
- **Exploration:** Selecting unknown or non-greedy actions to discover high-reward corridors, avoid local optima, and map the obstacle topology.
- **Exploitation:** Selecting the best-known action ($\arg\max_a Q(s,a)$) to maximize accumulated reward.
- **$\epsilon$-Decay Strategy:**
  $$\epsilon_{k+1} = \max(\epsilon_{\min},\, \epsilon_k \times \text{decay})$$
  At episode $0$, $\epsilon = 1.0$ (100% random exploration). Over training, $\epsilon$ smoothly decays to $\epsilon_{\min} = 0.05$, shifting behavior smoothly from discovery to optimal exploitation.

---

#### Q12: Why does Tabular Q-Learning converge?
**Answer:**
According to the Watkins-Dayan Theorem (1992), Tabular Q-learning is guaranteed to converge to the optimal action-value function $Q^*(s, a)$ with probability 1 if:
1. The state and action spaces are finite discrete sets.
2. All state-action pairs are visited infinitely often (guaranteed during training by $\epsilon > 0$).
3. The learning rate satisfies the Robbins-Monro conditions:
   $$\sum_{t=1}^{\infty} \alpha_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty$$
   In practice, a small constant or decaying $\alpha$ guarantees empirical numerical convergence to a stable policy.

---

#### Q13: Why can Reinforcement Learning outperform a fixed routing algorithm (like static Dijkstra / A*) under changing traffic?
**Answer:**
- **Dijkstra / A\*:** Calculate the shortest geometric path based on a static distance metric or instantaneous snapshot. When dynamic traffic congestion suddenly spikes on the main avenue, static planners either route directly into the bottleneck or require re-computing the entire path graph at every single time step.
- **RL Agent:** Learns an expected value policy that accounts for stochastic transition costs. If a road segment frequently experiences heavy traffic penalties, the Q-values along that path decrease. The agent automatically learns to prefer a slightly longer spatial detour with smooth traffic flow over a congested corridor!

---

#### Q14: What are the limitations of Tabular Q-Learning?
**Answer:**
1. **Curse of Dimensionality:** Tabular Q-learning requires storing a lookup table of size $|\mathcal{S}| \times |\mathcal{A}|$. On a continuous map or very large grid ($1000 \times 1000$), storing and updating millions of states in memory becomes computationally infeasible.
2. **Lack of Generalization:** In tabular Q-learning, updating $Q(s, a)$ gives zero information about neighboring states $Q(s', a)$.
3. **Remedy (Future Improvement):** Use Function Approximation, specifically **Deep Q-Networks (DQN)** with convolutional/neural network layers to generalize across continuous state spaces.

---

#### Q15: How does this project compare against classic Gymnasium Taxi-v3?
**Answer:**
- **Gymnasium Taxi-v3:** Uses a rigid $5 \times 5$ grid with fixed barrier walls, static pickup/dropoff spots, and **zero traffic dynamics**.
- **Our Custom City Environment:**
  - Configurable $N \times M$ city grid ($8 \times 8$, $10 \times 10$, etc.)
  - Procedurally generated obstacle networks with BFS reachability verification
  - Three-tier stochastic traffic system (`LOW`, `MEDIUM`, `HIGH`) with real-time dynamic shifts
  - Pure from-scratch NumPy implementation without SB3 dependencies
  - Comprehensive baseline comparison suite (Random, Shortest-Path, Q-Learning).
