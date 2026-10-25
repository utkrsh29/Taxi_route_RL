"""
City Grid module: Handles obstacles, road network, coordinates, and reachability validation.
"""

from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class CityGrid:
    """Represents the discrete spatial topology of the city."""

    def __init__(self, rows: int = 10, cols: int = 10, obstacle_density: float = 0.15,
                 seed: int = 42, num_stations: int = 4):
        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.rng = np.random.RandomState(seed)
        self.num_stations = num_stations
        # True indicates an obstacle / blocked road cell
        self.obstacles = np.zeros((rows, cols), dtype=bool)
        self.stations: List[Tuple[int, int]] = []
        self.generate_obstacles()
        self._init_stations()

    def _init_stations(self):
        """Select designated transit stations in well-separated parts of the city."""
        open_cells = self.get_open_cells()
        if not open_cells or self.num_stations <= 0:
            self.stations = []
            return

        # Find largest connected component of roads
        visited = set()
        components = []
        for cell in open_cells:
            if cell not in visited:
                comp = []
                q = deque([cell])
                visited.add(cell)
                while q:
                    curr = q.popleft()
                    comp.append(curr)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr[0] + dr, curr[1] + dc
                        if self.is_road(nr, nc) and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)

        largest_comp = max(components, key=len) if components else open_cells
        k = min(self.num_stations, len(largest_comp))
        if k < 2:
            self.stations = largest_comp[:k]
            return

        # Choose spread out stations (Greedy Max-Min distance)
        selected = [largest_comp[0]]
        while len(selected) < k:
            best_cand = None
            best_dist = -1
            for cand in largest_comp:
                if cand in selected:
                    continue
                min_d = min(abs(cand[0] - s[0]) + abs(cand[1] - s[1]) for s in selected)
                if min_d > best_dist:
                    best_dist = min_d
                    best_cand = cand
            if best_cand:
                selected.append(best_cand)
            else:
                break
        self.stations = selected

    def generate_obstacles(self):
        """Randomly populate obstacles while keeping density controlled."""
        self.obstacles.fill(False)
        total_cells = self.rows * self.cols
        num_obstacles = int(total_cells * self.obstacle_density)

        # Place obstacles uniformly at random
        indices = self.rng.choice(total_cells, size=num_obstacles, replace=False)
        for idx in indices:
            r = idx // self.cols
            c = idx % self.cols
            self.obstacles[r, c] = True

    def is_valid_coord(self, row: int, col: int) -> bool:
        """Check if (row, col) is within grid boundaries."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_road(self, row: int, col: int) -> bool:
        """Check if (row, col) is an unblocked, valid road cell."""
        return self.is_valid_coord(row, col) and not self.obstacles[row, col]

    def get_open_cells(self) -> List[Tuple[int, int]]:
        """Return list of all coordinate tuples that are valid unblocked roads."""
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.obstacles[r, c]:
                    cells.append((r, c))
        return cells

    def is_reachable(self, start: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Use Breadth-First Search (BFS) to verify target is reachable from start."""
        if not self.is_road(*start) or not self.is_road(*target):
            return False
        if start == target:
            return True

        queue = deque([start])
        visited = {start}

        while queue:
            curr_r, curr_c = queue.popleft()
            if (curr_r, curr_c) == target:
                return True

            # 4 directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if self.is_road(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def sample_reachable_endpoints(self, use_stations: bool = False) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Sample valid (taxi_start, passenger, destination) locations ensuring
        mutual reachability: taxi -> passenger and passenger -> destination.
        """
        if use_stations and len(self.stations) >= 2:
            st_indices = self.rng.choice(len(self.stations), size=2, replace=False)
            passenger = self.stations[st_indices[0]]
            destination = self.stations[st_indices[1]]
            # Taxi can start at any open cell that can reach passenger
            open_cells = self.get_open_cells()
            for _ in range(50):
                taxi_cand = open_cells[int(self.rng.choice(len(open_cells)))]
                if self.is_reachable(taxi_cand, passenger):
                    return taxi_cand, passenger, destination
            return passenger, passenger, destination

        open_cells = self.get_open_cells()
        if len(open_cells) < 3:
            raise ValueError("Not enough open road cells in grid.")

        for _ in range(500):  # Maximum attempts to sample mutually reachable triplet
            idx_choices = self.rng.choice(len(open_cells), size=3, replace=False)
            taxi = open_cells[idx_choices[0]]
            passenger = open_cells[idx_choices[1]]
            destination = open_cells[idx_choices[2]]

            if self.is_reachable(taxi, passenger) and self.is_reachable(passenger, destination):
                return taxi, passenger, destination

        # Fallback: clear obstacles on a direct corridor if density is too tight
        self.generate_obstacles()
        self._init_stations()
        return self.sample_reachable_endpoints(use_stations=use_stations)

    def render_ascii(self, taxi: Tuple[int, int], passenger: Tuple[int, int],
                     destination: Tuple[int, int], passenger_picked: bool,
                     traffic_grid: Optional[np.ndarray] = None) -> str:
        """Return an ASCII string representation of the grid city."""
        lines = []
        header = f"City Grid ({self.rows}x{self.cols}) [Passenger Picked: {passenger_picked}]"
        lines.append(header)
        lines.append("-" * (self.cols * 2 + 1))

        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                coord = (r, c)
                if coord == taxi:
                    row_str.append("S" if not passenger_picked else "T*")
                elif coord == passenger and not passenger_picked:
                    row_str.append("P")
                elif coord == destination:
                    row_str.append("D")
                elif self.obstacles[r, c]:
                    row_str.append("#")
                elif traffic_grid is not None and traffic_grid[r, c] > 0:
                    row_str.append("T" if traffic_grid[r, c] == 1 else "H")
                else:
                    row_str.append(".")
            lines.append(" ".join(row_str))

        lines.append("-" * (self.cols * 2 + 1))
        return "\n".join(lines)
