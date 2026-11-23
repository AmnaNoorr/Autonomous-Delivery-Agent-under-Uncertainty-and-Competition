# part_a_world.py
# -----------------------------------------------------------------------------
# Imports and Terrain Constants
# Here we import basic timing and numpy tools, plus our group's instance setup.
# We also define the 4 terrain types and note that 'mud' has ID 2.
# -----------------------------------------------------------------------------
import time
import numpy as np
from config import MEMBERS, inst, grid, stamp

TERRAINS = ["road", "grass", "mud", "ford"]
MUD_TERRAIN_ID = 2  # Index of "mud" in TERRAINS


# -----------------------------------------------------------------------------
# Twist Feature Generator
# This function sets up the special rules and locations for our grid world.
# It uses the unique instance ID as a random seed so everything is reproducible.
# -----------------------------------------------------------------------------
def generate_twist_features(inst, grid, num_tunnels=2, num_chargers=3):
    """
    Deterministically generates all three twist structures seeded from instance_id:
    1. teleport_tunnels: Bidirectional portal dictionary
    2. fuel_limit_with_charging_stations: Set of charger coordinates + max_fuel
    3. fragile_mud_cells (3rd Group Twist): Set of all mud coordinates (terrain_id == 2)
       that double in cost on hit 1 and collapse into obstacles (-1) on hit 2.
    """
    # Seed random number generator with instance ID so it produces the exact same layout every time
    rng = np.random.default_rng(int(inst["instance_id"], 16))
    n = inst["grid_size"]

    # Gather all walkable cells, making sure not to place twists on start or delivery points
    reserved = {inst["start"], *inst["delivery_stops"]}
    free_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if grid[r, c] >= 0 and (r, c) not in reserved
    ]

    # Pick random spots for tunnels and chargers without repeating any cell
    needed = (num_tunnels * 2) + num_chargers
    chosen_idx = rng.choice(len(free_cells), size=needed, replace=False)
    chosen = [free_cells[i] for i in chosen_idx]

    # Twist 1: Teleport tunnels (two-way pairs: stepping into A pops you out at B and vice-versa)
    tunnels = {}
    for i in range(0, num_tunnels * 2, 2):
        a, b = chosen[i], chosen[i + 1]
        tunnels[a] = b
        tunnels[b] = a

    # Twist 2: Charging stations and max fuel capacity (fuel scaled to grid size)
    chargers = set(chosen[num_tunnels * 2 :])
    max_fuel = int(n * 2.5)

    # Twist 3: Fragile mud cells (all mud tiles that get damaged on step 1 and break on step 2)
    mud_coords = np.argwhere(grid == MUD_TERRAIN_ID)
    fragile_mud = {tuple(map(int, pos)) for pos in mud_coords if tuple(map(int, pos)) not in reserved}

    return {
        "tunnels": tunnels,
        "chargers": chargers,
        "max_fuel": max_fuel,
        "fragile_mud": fragile_mud,
        "mud_collapse_hits": 2,
    }


# -----------------------------------------------------------------------------
# Vectorized Cost Map Builder (NumPy)
# Fast way to turn the grid of terrain IDs into actual movement costs.
# Walls/obstacles become infinity so the pathfinder knows not to walk through them.
# Also updates costs for mud tiles that are damaged (2x cost) or collapsed (impassable).
# -----------------------------------------------------------------------------
def build_cost_map_vectorized(grid, terrain_cost, damaged_mud=None, collapsed_mud=None):
    """
    Vectorized NumPy lookup mapping terrain IDs (0..3) to travel costs and obstacles (-1) to np.inf.
    Also applies Twist 3 (Fragile Mud Cells):
    - damaged_mud cells cost 2x base mud cost
    - collapsed_mud cells become impassable (np.inf)
    """
    # Create lookup table matching terrain ID to cost
    lookup = np.array([terrain_cost[t] for t in TERRAINS], dtype=np.float64)
    # Start everything as impassable (infinity)
    cost_map = np.full(grid.shape, np.inf, dtype=np.float64)
    # Fill in actual costs for walkable cells
    walkable = grid >= 0
    cost_map[walkable] = lookup[grid[walkable]]

    # If any mud cell was stepped on once, double its movement cost
    if damaged_mud:
        d_arr = np.array(list(damaged_mud), dtype=int)
        cost_map[d_arr[:, 0], d_arr[:, 1]] *= 2.0

    # If any mud cell was stepped on twice, it collapses into a wall (impassable)
    if collapsed_mud:
        c_arr = np.array(list(collapsed_mud), dtype=int)
        cost_map[c_arr[:, 0], c_arr[:, 1]] = np.inf

    return cost_map


# -----------------------------------------------------------------------------
# Pure Python Cost Map Builder
# Same logic as above, but written using standard Python nested loops.
# Used purely as a baseline to measure how much faster NumPy is.
# -----------------------------------------------------------------------------
def build_cost_map_python(grid, terrain_cost, damaged_mud=None, collapsed_mud=None):
    """Pure-Python nested loop version for benchmarking against vectorized NumPy."""
    rows, cols = grid.shape
    grid_list = grid.tolist()
    damaged_set = set(damaged_mud) if damaged_mud else set()
    collapsed_set = set(collapsed_mud) if collapsed_mud else set()

    cost_list = [[float("inf")] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            # Skip cells that collapsed
            if (r, c) in collapsed_set:
                continue
            cell = grid_list[r][c]
            # Valid terrain: grab cost and double if damaged
            if cell >= 0:
                val = float(terrain_cost[TERRAINS[cell]])
                if (r, c) in damaged_set:
                    val *= 2.0
                cost_list[r][c] = val
    return cost_list


# -----------------------------------------------------------------------------
# Performance Benchmark
# Runs both the Python loop version and the NumPy version 5,000 times.
# Compares their speed and verifies that both versions produce identical costs.
# -----------------------------------------------------------------------------
def benchmark_cost_maps(grid, terrain_cost, runs=5000):
    """Compares pure-Python loop vs. vectorized NumPy cost-map construction."""
    # Measure pure Python loop speed
    t0 = time.perf_counter()
    for _ in range(runs):
        py_map = build_cost_map_python(grid, terrain_cost)
    py_time = (time.perf_counter() - t0) / runs

    # Measure vectorized NumPy speed
    t0 = time.perf_counter()
    for _ in range(runs):
        np_map = build_cost_map_vectorized(grid, terrain_cost)
    np_time = (time.perf_counter() - t0) / runs

    # Sanity check: make sure both calculations match 100%
    assert np.allclose(np.array(py_map), np_map), "Mismatch between Python and NumPy cost maps!"
    speedup = py_time / np_time
    return py_time, np_time, speedup


# -----------------------------------------------------------------------------
# Main Execution Block
# Runs when this script is executed directly: generates world twists,
# builds the cost map, benchmarks performance, and prints summary stats.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 75)
    print(stamp(MEMBERS))
    print("=" * 75)

    twists = generate_twist_features(inst, grid)
    cost_map = build_cost_map_vectorized(grid, inst["terrain_cost"])
    py_t, np_t, speedup = benchmark_cost_maps(grid, inst["terrain_cost"])

    print(f"Instance ID       : {inst['instance_id']} | Grid Size: {inst['grid_size']}x{inst['grid_size']}")
    print(f"Terrain Costs     : {inst['terrain_cost']}")
    print(f"Max Fuel          : {twists['max_fuel']} | Chargers: {sorted(twists['chargers'])}")
    print(f"Teleport Tunnels  : {twists['tunnels']}")
    print(f"Fragile Mud Cells : {len(twists['fragile_mud'])} tiles (2x cost on hit 1, collapse on hit 2)")
    print("-" * 75)
    print(f"Pure-Python avg   : {py_t * 1e6:.2f} µs")
    print(f"Vectorized NumPy  : {np_t * 1e6:.2f} µs")
    print(f"Speed-up Factor   : {speedup:.2f}x faster")


# =============================================================================
# FILE SUMMARY: part_a_world.py
# -----------------------------------------------------------------------------
# This file sets up the foundation for the grid world:
# 1. Places deterministic twist features (teleport tunnels, charging stations,
#    and fragile mud tiles that degrade when stepped on).
# 2. Builds a 2D cost map where each walkable cell has a traversal cost and
#    obstacles/collapsed cells are marked as infinity.
# 3. Compares NumPy vectorization against plain Python loops, showing a ~15-20x
#    speedup while ensuring both methods output identical matrices.
# =============================================================================