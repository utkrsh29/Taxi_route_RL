"""
Verification Script for Phases 1 to 5.
Validates each component individually before training:
- Phase 1: Grid Environment & Reachability
- Phase 2: Taxi Movement & Boundaries
- Phase 3: Passenger Pickup & Destination Delivery
- Phase 4: Reward Function
- Phase 5: Q-Learning Algorithm from Scratch
"""

import os
import sys
import numpy as np

from config import Action, EnvConfig, RewardConfig, QLearningConfig, TrafficLevel
from environment.grid import CityGrid
from environment.traffic import TrafficManager
from environment.taxi_env import TaxiCityEnv
from agent.q_learning import QLearningAgent


def test_phase_1_grid():
    print("\n--- Testing Phase 1: City Grid & Reachability ---")
    grid = CityGrid(rows=6, cols=6, obstacle_density=0.15, seed=42)
    assert grid.rows == 6 and grid.cols == 6, "Grid dimensions mismatch"
    assert np.sum(grid.obstacles) > 0, "No obstacles generated"

    taxi, pass_pos, dest = grid.sample_reachable_endpoints()
    print(f"Sampled endpoints -> Taxi: {taxi}, Passenger: {pass_pos}, Dest: {dest}")
    assert grid.is_reachable(taxi, pass_pos), "Passenger unreachable from Taxi"
    assert grid.is_reachable(pass_pos, dest), "Destination unreachable from Passenger"

    # Traffic Manager test
    traffic = TrafficManager(shape=(6, 6), traffic_density=0.3, seed=42)
    traffic.reset(obstacle_mask=grid.obstacles)
    assert traffic.traffic_grid.shape == (6, 6)

    print("Phase 1 Grid ASCII Rendering:")
    print(grid.render_ascii(taxi, pass_pos, dest, False, traffic.traffic_grid))
    print(">>> Phase 1 PASSED! [OK]")


def test_phase_2_movement():
    print("\n--- Testing Phase 2: Taxi Movement & Collisions ---")
    env_cfg = EnvConfig(grid_size=(6, 6), obstacle_density=0.0, seed=42)
    env = TaxiCityEnv(env_config=env_cfg)
    env.reset(fixed_endpoints=((2, 2), (0, 0), (5, 5)))

    # Move UP -> (1, 2)
    state, r, term, trunc, info = env.step(Action.UP)
    assert env.taxi_pos == (1, 2), f"Expected (1, 2), got {env.taxi_pos}"

    # Move LEFT -> (1, 1)
    state, r, term, trunc, info = env.step(Action.LEFT)
    assert env.taxi_pos == (1, 1), f"Expected (1, 1), got {env.taxi_pos}"

    # Move to top boundary and test violation
    env.taxi_pos = (0, 1)
    state, r, term, trunc, info = env.step(Action.UP)
    assert env.taxi_pos == (0, 1), "Taxi should not teleport off-grid"
    assert info["event"] == "hit_boundary", "Boundary event not triggered"

    print(">>> Phase 2 PASSED! [OK]")


def test_phase_3_pickup_delivery():
    print("\n--- Testing Phase 3: Passenger Pickup & Destination Delivery ---")
    env_cfg = EnvConfig(grid_size=(6, 6), obstacle_density=0.0, seed=42)
    env = TaxiCityEnv(env_config=env_cfg)
    env.reset(fixed_endpoints=((1, 1), (1, 2), (1, 3)))

    assert env.passenger_picked == 0, "Passenger should start unpicked"

    # Step RIGHT to reach passenger at (1, 2)
    state, r, term, trunc, info = env.step(Action.RIGHT)
    assert env.passenger_picked == 1, "Passenger should be picked"
    assert info["event"] == "passenger_pickup"
    assert not term, "Should not terminate upon pickup"

    # Step RIGHT to reach destination at (1, 3)
    state, r, term, trunc, info = env.step(Action.RIGHT)
    assert term is True, "Episode should terminate on successful delivery"
    assert info["event"] == "delivery_success"

    print(">>> Phase 3 PASSED! [OK]")


def test_phase_4_rewards():
    print("\n--- Testing Phase 4: Reward Function Verification ---")
    rc = RewardConfig(
        step_penalty=-1.0,
        boundary_penalty=-5.0,
        obstacle_penalty=-5.0,
        pickup_reward=20.0,
        delivery_reward=100.0,
        traffic_med_penalty=-2.0
    )
    env_cfg = EnvConfig(grid_size=(5, 5), obstacle_density=0.0, seed=42)
    env = TaxiCityEnv(env_config=env_cfg, reward_config=rc)
    env.reset(fixed_endpoints=((0, 0), (0, 1), (0, 2)))

    # 1. Boundary penalty
    _, r_bound, _, _, _ = env.step(Action.UP)
    assert r_bound == -5.0, f"Expected -5.0, got {r_bound}"

    # 2. Pickup transition: step penalty (-1.0) + pickup reward (+20.0) = +19.0
    _, r_pickup, _, _, _ = env.step(Action.RIGHT)
    assert r_pickup == 19.0, f"Expected 19.0, got {r_pickup}"

    # 3. Delivery transition: step penalty (-1.0) + delivery reward (+100.0) = +99.0
    _, r_deliv, term, _, _ = env.step(Action.RIGHT)
    assert r_deliv == 99.0, f"Expected 99.0, got {r_deliv}"
    assert term is True

    print(">>> Phase 4 PASSED! [OK]")


def test_phase_5_q_learning():
    print("\n--- Testing Phase 5: Q-Learning from Scratch ---")
    agent = QLearningAgent(
        num_actions=4,
        config=QLearningConfig(alpha=0.5, gamma=0.9, epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.1),
        seed=42
    )

    s1 = (0, 0, 1, 1, 2, 2, 0)
    s2 = (0, 1, 1, 1, 2, 2, 0)

    # Test Q update
    # Initial Q(s1, Action.RIGHT) = 0.0
    # Next max Q(s2) = 0.0
    # Target = R + gamma * 0 = -1.0
    # New Q = 0 + 0.5 * (-1.0 - 0) = -0.5
    td_err = agent.update(state=s1, action=Action.RIGHT, reward=-1.0, next_state=s2, done=False)
    assert agent.get_q_values(s1)[Action.RIGHT] == -0.5, f"Q-value calculation error: {agent.get_q_values(s1)}"
    print(f"Verified manual Q-update: Q(s1, RIGHT) = {agent.get_q_values(s1)[Action.RIGHT]}")

    # Test epsilon decay
    old_eps = agent.epsilon
    new_eps = agent.decay_epsilon()
    assert new_eps == old_eps * 0.9, "Epsilon decay error"

    # Test Save & Load
    test_file = "test_q_table.pkl"
    agent.save(test_file)
    assert os.path.exists(test_file), "Q-table save failed"

    loaded_agent = QLearningAgent(num_actions=4)
    loaded_agent.load(test_file)
    assert loaded_agent.get_q_values(s1)[Action.RIGHT] == -0.5, "Loaded Q-table mismatch"
    if os.path.exists(test_file):
        os.remove(test_file)

    # Test 5 short rollouts with agent in environment
    env = TaxiCityEnv(env_config=EnvConfig(grid_size=(6, 6), seed=42))
    for ep in range(5):
        st = env.reset()
        done = False
        step_cnt = 0
        while not done and step_cnt < 20:
            act = agent.select_action(st, explore=True)
            nxt_st, r, term, trunc, _ = env.step(act)
            done = term or trunc
            agent.update(st, act, r, nxt_st, done)
            st = nxt_st
            step_cnt += 1
        agent.decay_epsilon()

    print(f"Explored states in Q-table after 5 mini episodes: {agent.table_size}")
    print(">>> Phase 5 PASSED! [OK]")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING UNIT & INTEGRATION TESTS (PHASES 1 TO 5)")
    print("==================================================")
    test_phase_1_grid()
    test_phase_2_movement()
    test_phase_3_pickup_delivery()
    test_phase_4_rewards()
    test_phase_5_q_learning()
    print("==================================================")
    print("ALL TESTS (PHASES 1 - 5) PASSED SUCCESSFULLY! [DONE]")
    print("==================================================")
