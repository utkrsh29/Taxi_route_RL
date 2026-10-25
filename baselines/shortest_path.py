"""
Shortest-Path Baseline for Taxi City Navigation.
Uses Breadth-First Search (BFS) / Manhattan pathfinding to route the taxi to the
passenger and subsequently to the destination, navigating around obstacles.
"""

from collections import deque
from typing import Tuple, List, Optional, Any
from config import Action


class ShortestPathAgent:
    """
    Classic algorithmic baseline using BFS shortest-path on the obstacle grid.
    Routes: Taxi -> Passenger (phase 1), then Taxi -> Destination (phase 2).
    """

    def __init__(self, action_count: int = 4):
        self.action_count = action_count
        self.action_deltas = {
            (-1, 0): Action.UP,
            (1, 0): Action.DOWN,
            (0, -1): Action.LEFT,
            (0, 1): Action.RIGHT,
        }

    def _find_path(self, grid, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute shortest path of road coordinates from start to target."""
        if start == target:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path

            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                neighbor = (nr, nc)
                if grid.is_road(nr, nc) and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found due to obstruction, return empty
        return []

    def select_action_from_env(self, env) -> int:
        """Select next action given full environment access."""
        taxi_pos = env.taxi_pos
        pass_picked = env.passenger_picked

        # Determine target
        target = env.dest_pos if pass_picked else env.passenger_pos

        if taxi_pos == target:
            return Action.WAIT if self.action_count > 4 else Action.UP

        path = self._find_path(env.grid, taxi_pos, target)
        if len(path) >= 2:
            next_cell = path[1]
            dr = next_cell[0] - taxi_pos[0]
            dc = next_cell[1] - taxi_pos[1]
            return int(self.action_deltas.get((dr, dc), Action.UP))

        # Fallback: simple greedy Manhattan step
        dr = target[0] - taxi_pos[0]
        dc = target[1] - taxi_pos[1]
        if abs(dr) >= abs(dc):
            return Action.DOWN if dr > 0 else Action.UP
        else:
            return Action.RIGHT if dc > 0 else Action.LEFT

    def select_action(self, state: Tuple[int, ...], explore: bool = False, env: Optional[Any] = None) -> int:
        """
        State format: (taxi_r, taxi_c, pass_r, pass_c, dest_r, dest_c, passenger_picked)
        """
        if env is not None:
            return self.select_action_from_env(env)

        taxi_pos = (state[0], state[1])
        pass_pos = (state[2], state[3])
        dest_pos = (state[4], state[5])
        pass_picked = state[6]

        target = dest_pos if pass_picked else pass_pos
        dr = target[0] - taxi_pos[0]
        dc = target[1] - taxi_pos[1]

        if dr == 0 and dc == 0:
            return Action.WAIT if self.action_count > 4 else Action.UP

        # Greedy coordinate step
        if abs(dr) >= abs(dc) and dr != 0:
            return int(Action.DOWN if dr > 0 else Action.UP)
        elif dc != 0:
            return int(Action.RIGHT if dc > 0 else Action.LEFT)
        return int(Action.UP)

    def __call__(self, env: Any, state: Tuple[int, ...]) -> int:
        return self.select_action_from_env(env)
