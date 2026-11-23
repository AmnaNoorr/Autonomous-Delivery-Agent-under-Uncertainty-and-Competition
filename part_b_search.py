# part_b_search.py
# -----------------------------------------------------------------------------
# Imports
# Standard library tools for heaps (priority queues), time/memory measurement,
# queues, immutable state storage, and our project-specific settings and world rules.
# -----------------------------------------------------------------------------
import heapq
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
import numpy as np

from config import MEMBERS, inst, grid, stamp
from part_a_world import generate_twist_features, build_cost_map_vectorized, MUD_TERRAIN_ID


# -----------------------------------------------------------------------------
# DeliveryState Data Structure
# Represents a snapshot of the delivery agent at any given moment.
# It is 'frozen' (immutable) so it can be hashed and stored in visited sets or dictionaries:
# - r, c: current grid coordinates
# - delivered_mask: bitmask tracking which delivery packages have been dropped off
# - fuel: remaining fuel units
# - damaged_mud: set of mud tiles stepped on once (now twice as expensive)
# - collapsed_mud: set of mud tiles stepped on twice (now completely blocked)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DeliveryState:
    r: int
    c: int
    delivered_mask: int
    fuel: int
    damaged_mud: frozenset = frozenset()
    collapsed_mud: frozenset = frozenset()


# -----------------------------------------------------------------------------
# Search Problem Formulation
# Packages the grid environment, delivery targets, and all twists into
# a clean interface that our search algorithms can explore.
# -----------------------------------------------------------------------------
class DeliverySearchProblem:
    """
    Single search problem formulation incorporating:
    - Multi-stop parcel delivery via bitmask
    - Fuel limit & charging station refueling
    - Bidirectional teleport tunnels
    - Fragile mud cells (3rd Twist: 2x cost on hit 1, permanent collapse on hit 2)
    """

    def __init__(
        self,
        grid,
        cost_map,
        start,
        stops,
        tunnels,
        chargers,
        max_fuel,
        enable_fuel=True,
        enable_tunnels=True,
        enable_fragile_mud=True,
    ):
        # Save grid and movement costs
        self.grid = grid
        self.cost_map = cost_map
        self.n = grid.shape[0]
        self.start = start
        self.stops = list(stops)

        # Assign each delivery stop its own bit (e.g. stop 0 -> 1, stop 1 -> 2, stop 2 -> 4)
        # That way, goal_mask has all bits set to 1 when every stop is visited.
        self.stop_to_bit = {pos: (1 << i) for i, pos in enumerate(self.stops)}
        self.goal_mask = (1 << len(self.stops)) - 1

        # Configuration flags for ablations and twists
        self.tunnels = tunnels if enable_tunnels else {}
        self.chargers = chargers if enable_fuel else set()
        self.max_fuel = max_fuel
        self.enable_fuel = enable_fuel
        self.enable_fragile_mud = enable_fragile_mud

        # The cheapest single step on the board (used to scale heuristics safely)
        self.min_step_cost = float(np.min(cost_map[grid >= 0]))
        self.reserved = {start, *self.stops}

        # Keep a list of tunnel entrances for fast distance checks
        self.portal_list = list(self.tunnels.keys())

    # Build the starting state (agent at start, 0 parcels delivered, full fuel tank)
    def initial_state(self) -> DeliveryState:
        r, c = self.start
        mask = self.stop_to_bit.get((r, c), 0)
        fuel = self.max_fuel if self.enable_fuel else 0
        return DeliveryState(r, c, mask, fuel, frozenset(), frozenset())

    # Goal check: have all delivery packages been dropped off?
    def is_goal(self, state: DeliveryState) -> bool:
        return state.delivered_mask == self.goal_mask

    # Successor generator: finds all legal moves the agent can make from the current state
    def get_successors(self, state: DeliveryState):
        """Yields (action_name, next_state, step_cost)."""
        # 1. Normal 4-directional moves: UP, DOWN, LEFT, RIGHT
        for action, (dr, dc) in (("UP", (-1, 0)), ("DOWN", (1, 0)), ("LEFT", (0, -1)), ("RIGHT", (0, 1))):
            nr, nc = state.r + dr, state.c + dc
            # Ensure next cell is within board boundaries and not an obstacle
            if 0 <= nr < self.n and 0 <= nc < self.n and self.grid[nr, nc] >= 0:
                # Mud that already collapsed into a wall is impassable
                if self.enable_fragile_mud and (nr, nc) in state.collapsed_mud:
                    continue

                base_cost = float(self.cost_map[nr, nc])
                is_fragile_mud = (
                    self.enable_fragile_mud
                    and (self.grid[nr, nc] == MUD_TERRAIN_ID)
                    and ((nr, nc) not in self.reserved)
                )
                is_damaged = is_fragile_mud and ((nr, nc) in state.damaged_mud)

                # If mud was already stepped on once, stepping on it again costs double
                step_cost = base_cost * 2.0 if is_damaged else base_cost
                fuel_needed = int(step_cost)

                # Skip move if we don't have enough fuel left
                if self.enable_fuel and state.fuel < fuel_needed:
                    continue

                # Deduct fuel, or refill to 100% if landing on a charging station
                next_fuel = (state.fuel - fuel_needed) if self.enable_fuel else 0
                if self.enable_fuel and (nr, nc) in self.chargers:
                    next_fuel = self.max_fuel

                # Update mud status: first step marks it damaged; second step collapses it
                next_damaged = state.damaged_mud
                next_collapsed = state.collapsed_mud
                if is_fragile_mud:
                    if is_damaged:
                        next_damaged = state.damaged_mud - {(nr, nc)}
                        next_collapsed = state.collapsed_mud | {(nr, nc)}
                    else:
                        next_damaged = state.damaged_mud | {(nr, nc)}

                # Update delivery bitmask if this cell is one of our delivery stops
                next_mask = state.delivered_mask | self.stop_to_bit.get((nr, nc), 0)
                yield action, DeliveryState(nr, nc, next_mask, next_fuel, next_damaged, next_collapsed), step_cost

        # 2. Teleport tunnel move: if standing on a tunnel entrance, can jump to exit for 1 cost
        if (state.r, state.c) in self.tunnels:
            tr, tc = self.tunnels[(state.r, state.c)]
            if not self.enable_fuel or state.fuel >= 1:
                next_fuel = (state.fuel - 1) if self.enable_fuel else 0
                if self.enable_fuel and (tr, tc) in self.chargers:
                    next_fuel = self.max_fuel
                next_mask = state.delivered_mask | self.stop_to_bit.get((tr, tc), 0)
                yield "TELEPORT", DeliveryState(
                    tr, tc, next_mask, next_fuel, state.damaged_mud, state.collapsed_mud
                ), 1.0

    # -------------------------------------------------------------------------
    # Heuristics (Distance Estimators)
    # These guide informed algorithms (like A* and Greedy) toward the goal.
    # -------------------------------------------------------------------------

    # Helper: shortest theoretical distance between two points, considering teleports
    def _tunnel_aware_dist(self, p1, p2) -> float:
        """
        Minimum distance between p1 and p2 considering either:
        - Direct Manhattan distance, OR
        - Traveling from p1 to a tunnel entrance, teleporting (cost 1), and walking to p2.
        """
        # Baseline: direct Manhattan walk
        best = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        # Check if using any tunnel shortcut makes the trip faster
        for entrance, exit_pos in self.tunnels.items():
            d_via_tunnel = (
                abs(p1[0] - entrance[0]) + abs(p1[1] - entrance[1])
                + 1
                + abs(exit_pos[0] - p2[0]) + abs(exit_pos[1] - p2[1])
            )
            if d_via_tunnel < best:
                best = d_via_tunnel
        return float(best)

    # Custom Admissible & Consistent Heuristic (MST + Nearest Stop)
    # Combines: (distance to closest remaining stop) + (MST of remaining stops)
    # Multiplied by min_step_cost so it never overestimates real cost.
    def h_custom_admissible(self, state: DeliveryState) -> float:
        """
        Custom Admissible & Consistent Heuristic (MST + Nearest Stop over Tunnel-Aware Metric):
        1. Finds the minimum tunnel-aware distance from current position to closest undelivered stop.
        2. Adds the Minimum Spanning Tree (MST) cost connecting all remaining undelivered stops.
        3. Scales the total step count by min(1.0, min_step_cost).
        """
        # Find which packages still need delivery
        remaining = [
            pos for i, pos in enumerate(self.stops) if not (state.delivered_mask & (1 << i))
        ]
        if not remaining:
            return 0.0

        curr = (state.r, state.c)
        # Step 1: Distance to the nearest undelivered stop
        dist_to_first = min(self._tunnel_aware_dist(curr, s) for s in remaining)

        # Step 2: Minimum Spanning Tree connecting all remaining stops (via Prim's algorithm)
        mst_cost = 0.0
        if len(remaining) > 1:
            in_mst = {0}
            min_edge = [self._tunnel_aware_dist(remaining[0], remaining[j]) for j in range(len(remaining))]
            while len(in_mst) < len(remaining):
                u = min((j for j in range(len(remaining)) if j not in in_mst), key=lambda j: min_edge[j])
                in_mst.add(u)
                mst_cost += min_edge[u]
                for v in range(len(remaining)):
                    if v not in in_mst:
                        d = self._tunnel_aware_dist(remaining[u], remaining[v])
                        if d < min_edge[v]:
                            min_edge[v] = d

        # Step 3: Scale by minimum step cost so the estimate is strictly admissible
        return (dist_to_first + mst_cost) * min(1.0, self.min_step_cost)

    # Baseline 1: only looks at distance to the single nearest remaining stop
    def h_baseline_nearest(self, state: DeliveryState) -> float:
        """Baseline 1: Tunnel-aware distance to nearest undelivered stop."""
        remaining = [
            pos for i, pos in enumerate(self.stops) if not (state.delivered_mask & (1 << i))
        ]
        if not remaining:
            return 0.0
        curr = (state.r, state.c)
        return min(self._tunnel_aware_dist(curr, s) for s in remaining) * min(1.0, self.min_step_cost)

    # Baseline 2: only looks at distance to the single farthest remaining stop
    def h_baseline_farthest(self, state: DeliveryState) -> float:
        """Baseline 2: Tunnel-aware distance to farthest undelivered stop."""
        remaining = [
            pos for i, pos in enumerate(self.stops) if not (state.delivered_mask & (1 << i))
        ]
        if not remaining:
            return 0.0
        curr = (state.r, state.c)
        return max(self._tunnel_aware_dist(curr, s) for s in remaining) * min(1.0, self.min_step_cost)

    # Inadmissible variant: inflates the heuristic by 3.5x to find solutions fast, risking sub-optimal paths
    def h_inadmissible_fast(self, state: DeliveryState) -> float:
        """Inadmissible weighted variant (3.5x multiplier) that finds paths fast but sacrifices optimality."""
        return 3.5 * self.h_custom_admissible(state)


# -----------------------------------------------------------------------------
# Helpers: Path Reconstruction & Performance Profiling
# -----------------------------------------------------------------------------

# Backtracks from the goal state through the came_from map to reconstruct the final path
def _reconstruct(came_from, goal_state):
    path = []
    curr = goal_state
    while curr is not None:
        path.append((curr.r, curr.c))
        curr = came_from[curr]
    path.reverse()
    return path


# Wraps any search function to track elapsed time (ms) and peak RAM usage (KB)
def _run_with_metrics(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res["runtime_ms"] = round(runtime_ms, 3)
    res["peak_kb"] = round(peak_bytes / 1024.0, 2)
    return res


# -----------------------------------------------------------------------------
# Search Algorithms Implemented From Scratch
# -----------------------------------------------------------------------------

# 1. Breadth-First Search (BFS)
# Explores level by level using a FIFO queue. Optimal in step count, ignores terrain cost.
def bfs(problem: DeliverySearchProblem, max_expansions: int = 60000):
    def _solve():
        start = problem.initial_state()
        if problem.is_goal(start):
            return {"found": True, "path": [(start.r, start.c)], "cost": 0.0, "expanded": 0}

        queue = deque([(start, 0.0)])
        came_from = {start: None}
        # Dominance key ignores harmless 1-hit mud history unless a tile actually collapsed
        seen_core = {(start.r, start.c, start.delivered_mask, start.fuel, start.collapsed_mud)}
        expanded = 0

        while queue:
            if expanded >= max_expansions:
                break
            state, g_cost = queue.popleft()
            expanded += 1

            for _, nxt, step_cost in problem.get_successors(state):
                core_key = (nxt.r, nxt.c, nxt.delivered_mask, nxt.fuel, nxt.collapsed_mud)
                if nxt not in came_from and core_key not in seen_core:
                    seen_core.add(core_key)
                    came_from[nxt] = state
                    new_cost = g_cost + step_cost
                    if problem.is_goal(nxt):
                        return {"found": True, "path": _reconstruct(came_from, nxt), "cost": new_cost, "expanded": expanded}
                    queue.append((nxt, new_cost))

        return {"found": False, "path": [], "cost": float("inf"), "expanded": expanded}

    return _run_with_metrics(_solve)


# 2. Uniform Cost Search (UCS / Dijkstra)
# Always expands the cheapest cumulative path first using a min-heap priority queue.
# Guaranteed to find the lowest-cost path.
def ucs(problem: DeliverySearchProblem, max_expansions: int = 80000):
    def _solve():
        start = problem.initial_state()
        counter = 0
        pq = [(0.0, counter, start)]
        came_from = {start: None}
        best_g = {start: 0.0}
        best_core_g = {(start.r, start.c, start.delivered_mask, start.fuel, start.collapsed_mud): 0.0}
        expanded = 0

        while pq:
            if expanded >= max_expansions:
                break
            g_cost, _, state = heapq.heappop(pq)
            if g_cost > best_g.get(state, float("inf")):
                continue
            expanded += 1

            if problem.is_goal(state):
                return {"found": True, "path": _reconstruct(came_from, state), "cost": g_cost, "expanded": expanded}

            for _, nxt, step_cost in problem.get_successors(state):
                new_g = g_cost + step_cost
                core_key = (nxt.r, nxt.c, nxt.delivered_mask, nxt.fuel, nxt.collapsed_mud)
                if new_g < best_g.get(nxt, float("inf")) and new_g < best_core_g.get(core_key, float("inf")):
                    best_g[nxt] = new_g
                    best_core_g[core_key] = new_g
                    came_from[nxt] = state
                    counter += 1
                    heapq.heappush(pq, (new_g, counter, nxt))

        return {"found": False, "path": [], "cost": float("inf"), "expanded": expanded}

    return _run_with_metrics(_solve)


# 3. Depth-Limited Search (DLS)
# DFS restricted to a maximum depth cutoff, with memory guardrails to prevent crashes.
def dls(problem: DeliverySearchProblem, depth_limit: int, memory_limit_states: int = 50000, max_expansions: int = 25000):
    def _solve():
        start = problem.initial_state()
        expanded = 0
        cutoff_occurred = False
        memory_exceeded = False

        # Iterative stack DLS tracking current branch path_set to prevent cycles
        stack = [(start, 0, 0.0, [(start.r, start.c)], {start})]

        while stack:
            if len(stack) > memory_limit_states or expanded >= max_expansions:
                memory_exceeded = True
                break

            state, depth, g_cost, path, path_set = stack.pop()
            expanded += 1

            if problem.is_goal(state):
                return {
                    "found": True,
                    "path": path,
                    "cost": g_cost,
                    "expanded": expanded,
                    "cutoff": False,
                    "memory_exceeded": False,
                }

            if depth >= depth_limit:
                cutoff_occurred = True
                continue

            for _, nxt, step_cost in problem.get_successors(state):
                if nxt not in path_set:  # Cycle check strictly on current branch
                    stack.append((nxt, depth + 1, g_cost + step_cost, path + [(nxt.r, nxt.c)], path_set | {nxt}))

        return {
            "found": False,
            "path": [],
            "cost": float("inf"),
            "expanded": expanded,
            "cutoff": cutoff_occurred,
            "memory_exceeded": memory_exceeded,
        }

    return _run_with_metrics(_solve)


# 4. Iterative Deepening Search (IDS)
# Runs DLS repeatedly, increasing the depth limit by 1 each round until the goal is found.
def ids(problem: DeliverySearchProblem, max_depth: int = 45, memory_limit_states: int = 25000, max_expansions: int = 25000):
    def _solve():
        total_expanded = 0
        for depth in range(max_depth + 1):
            remaining_budget = max(1, max_expansions - total_expanded)
            res = dls(
                problem,
                depth_limit=depth,
                memory_limit_states=memory_limit_states,
                max_expansions=remaining_budget,
            )
            total_expanded += res["expanded"]
            if res["found"]:
                res["expanded"] = total_expanded
                res["depth_found"] = depth
                return res
            # Stop early if out of memory or budget, or if the search space was fully exhausted
            if res["memory_exceeded"] or total_expanded >= max_expansions or not res["cutoff"]:
                return {
                    "found": False,
                    "path": [],
                    "cost": float("inf"),
                    "expanded": total_expanded,
                    "depth_reached": depth,
                    "memory_exceeded": True,
                }
        return {"found": False, "path": [], "cost": float("inf"), "expanded": total_expanded, "depth_reached": max_depth}

    return _run_with_metrics(_solve)


# 5. Greedy Best-First Search
# Explores nodes purely based on heuristic score h(n). Fast, but ignores path cost so not optimal.
def greedy_best_first(problem: DeliverySearchProblem, heuristic_fn=None, max_expansions: int = 50000):
    if heuristic_fn is None:
        heuristic_fn = problem.h_custom_admissible

    def _solve():
        start = problem.initial_state()
        counter = 0
        pq = [(heuristic_fn(start), counter, 0.0, start)]
        came_from = {start: None}
        visited = {(start.r, start.c, start.delivered_mask, start.fuel, start.collapsed_mud)}
        expanded = 0

        while pq:
            if expanded >= max_expansions:
                break
            _, _, g_cost, state = heapq.heappop(pq)
            expanded += 1

            if problem.is_goal(state):
                return {"found": True, "path": _reconstruct(came_from, state), "cost": g_cost, "expanded": expanded}

            for _, nxt, step_cost in problem.get_successors(state):
                core_key = (nxt.r, nxt.c, nxt.delivered_mask, nxt.fuel, nxt.collapsed_mud)
                if core_key not in visited:
                    visited.add(core_key)
                    came_from[nxt] = state
                    counter += 1
                    heapq.heappush(pq, (heuristic_fn(nxt), counter, g_cost + step_cost, nxt))

        return {"found": False, "path": [], "cost": float("inf"), "expanded": expanded}

    return _run_with_metrics(_solve)


# 6. A* Search
# Combines actual cost so far g(n) with estimated remaining cost h(n): f(n) = g(n) + h(n).
# Guarantees the optimal lowest-cost path when using an admissible heuristic.
def astar(problem: DeliverySearchProblem, heuristic_fn=None, max_expansions: int = 80000):
    if heuristic_fn is None:
        heuristic_fn = problem.h_custom_admissible

    def _solve():
        start = problem.initial_state()
        counter = 0
        pq = [(heuristic_fn(start), counter, 0.0, start)]
        came_from = {start: None}
        best_g = {start: 0.0}
        best_core_g = {(start.r, start.c, start.delivered_mask, start.fuel, start.collapsed_mud): 0.0}
        expanded = 0

        while pq:
            if expanded >= max_expansions:
                break
            f_cost, _, g_cost, state = heapq.heappop(pq)
            if g_cost > best_g.get(state, float("inf")):
                continue
            expanded += 1

            if problem.is_goal(state):
                return {"found": True, "path": _reconstruct(came_from, state), "cost": g_cost, "expanded": expanded}

            for _, nxt, step_cost in problem.get_successors(state):
                new_g = g_cost + step_cost
                core_key = (nxt.r, nxt.c, nxt.delivered_mask, nxt.fuel, nxt.collapsed_mud)
                if new_g < best_g.get(nxt, float("inf")) and new_g < best_core_g.get(core_key, float("inf")):
                    best_g[nxt] = new_g
                    best_core_g[core_key] = new_g
                    came_from[nxt] = state
                    counter += 1
                    heapq.heappush(pq, (new_g + heuristic_fn(nxt), counter, new_g, nxt))

        return {"found": False, "path": [], "cost": float("inf"), "expanded": expanded}

    return _run_with_metrics(_solve)


# -----------------------------------------------------------------------------
# Empirical Consistency Check
# Mathematically tests the triangle inequality: h(s) <= cost(s, a, s') + h(s').
# If consistency holds everywhere, A* will never need to re-open closed nodes.
# -----------------------------------------------------------------------------
def test_heuristic_consistency(problem: DeliverySearchProblem, sample_states=500):
    rng = np.random.default_rng(int(inst["instance_id"], 16))
    walkable = np.argwhere(problem.grid >= 0)
    violations = 0
    checked = 0

    for _ in range(sample_states):
        r, c = map(int, walkable[rng.integers(len(walkable))])
        mask = int(rng.integers(0, problem.goal_mask + 1))
        fuel = int(rng.integers(5, problem.max_fuel + 1))
        s = DeliveryState(r, c, mask, fuel)
        h_s = problem.h_custom_admissible(s)

        for _, nxt, cost in problem.get_successors(s):
            h_nxt = problem.h_custom_admissible(nxt)
            checked += 1
            # Check triangle inequality (with tiny epsilon for float precision)
            if h_s > cost + h_nxt + 1e-9:
                violations += 1

    return checked, violations


# -----------------------------------------------------------------------------
# Main Execution Block
# Tests heuristic consistency and runs a tournament of all search algorithms,
# printing a comparison table with cost, expanded nodes, runtime, and memory.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 85)
    print(stamp(MEMBERS))
    print("=" * 85)

    twists = generate_twist_features(inst, grid)
    cost_map = build_cost_map_vectorized(grid, inst["terrain_cost"])
    problem = DeliverySearchProblem(
        grid=grid,
        cost_map=cost_map,
        start=inst["start"],
        stops=inst["delivery_stops"],
        tunnels=twists["tunnels"],
        chargers=twists["chargers"],
        max_fuel=twists["max_fuel"],
    )

    checked, violations = test_heuristic_consistency(problem)
    print(f"Heuristic Consistency Check: {checked} transitions tested, {violations} violations found.")
    print("-" * 85)

    experiments = [
        ("BFS", lambda: bfs(problem)),
        ("UCS", lambda: ucs(problem)),
        ("Greedy (Custom h)", lambda: greedy_best_first(problem, problem.h_custom_admissible)),
        ("A* (Custom Admissible)", lambda: astar(problem, problem.h_custom_admissible)),
        ("A* (Baseline Nearest)", lambda: astar(problem, problem.h_baseline_nearest)),
        ("A* (Baseline Farthest)", lambda: astar(problem, problem.h_baseline_farthest)),
        ("A* (Inadmissible 3.5x)", lambda: astar(problem, problem.h_inadmissible_fast)),
        ("IDS (Memory-Limited)", lambda: ids(problem, max_depth=35, memory_limit_states=15000)),
    ]

    print(f"{'Algorithm':<25} | {'Found':<5} | {'Cost':<8} | {'Expanded':<10} | {'Time (ms)':<10} | {'Peak KB':<8}")
    print("-" * 85)
    for name, runner in experiments:
        res = runner()
        cost_str = f"{res['cost']:.1f}" if res["found"] else "INF"
        print(
            f"{name:<25} | {str(res['found']):<5} | {cost_str:<8} | {res['expanded']:<10} | "
            f"{res['runtime_ms']:<10.2f} | {res['peak_kb']:<8.1f}"
        )


# =============================================================================
# FILE SUMMARY: part_b_search.py
# -----------------------------------------------------------------------------
# This file handles multi-goal route planning across our grid world:
# 1. State & Problem: Stores agent location, parcels delivered (bitmask),
#    fuel level, and fragile mud wear using an immutable dataclass.
# 2. Heuristics: Implements an admissible & consistent heuristic using
#    Minimum Spanning Trees (MST) and portal-aware Manhattan distances,
#    alongside simpler baselines and an inflated inadmissible version.
# 3. Solvers: Implements BFS, UCS, DLS, IDS, Greedy Best-First, and A*
#    from scratch, profiling their paths, node expansions, runtime, and memory.
# =============================================================================