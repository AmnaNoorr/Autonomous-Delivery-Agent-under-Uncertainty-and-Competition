# run_all.py
# -----------------------------------------------------------------------------
# Imports and Setup
# Standard libraries for file I/O, math, JSON, dataframes (pandas), and charts (matplotlib).
# Also pulls in the problem instance, world rules, search planners, adversarial logic,
# and LLM dispatcher from Parts A, B, C, and D.
# -----------------------------------------------------------------------------
import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oel_instance import reachable
from config import MEMBERS, inst, grid, stamp
from part_a_world import generate_twist_features, build_cost_map_vectorized, benchmark_cost_maps
from part_b_search import (
    DeliverySearchProblem,
    bfs,
    ucs,
    greedy_best_first,
    astar,
    ids,
    test_heuristic_consistency,
)
from part_c_adversarial import (
    InterceptorGame,
    minimax_search,
    alphabeta_search,
    tune_evaluation_weights,
    run_tournament,
)
from part_d_llm import DispatcherToolkit, LLMDispatcherAgent, DISPATCHER_REQUESTS

# Ensure output directory exists and prepare team stamp for all tables
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
STAMP_LINE = stamp(MEMBERS)


# -----------------------------------------------------------------------------
# Statistical and File Helpers
# -----------------------------------------------------------------------------

# Calculates the mean and 95% confidence interval margin (1.96 * standard error)
def mean_ci95(values):
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return mean, 1.96 * se


# Exports a pandas DataFrame to CSV with the student stamp at the very top
def save_stamped_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {STAMP_LINE}\n")
        df.to_csv(f, index=False)
    print(f"Saved: {path}")


# -----------------------------------------------------------------------------
# Main Reproducibility Pipeline
# Executes the complete test suite across Parts A, B, C, D, and E,
# saving all tables and graphs required for the project report.
# -----------------------------------------------------------------------------
def main():
    print("=" * 90)
    print(STAMP_LINE)
    print("CS-370 OEL — Full Reproducibility Pipeline (All 3 Twists Enabled)")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # PART A: Vectorized Cost-Map Benchmark
    # Times 3,000 iterations of Python loops vs. NumPy vectorization and saves Table 1.
    # -------------------------------------------------------------------------
    py_t, np_t, speedup = benchmark_cost_maps(grid, inst["terrain_cost"], runs=3000)
    df_part_a = pd.DataFrame(
        [
            {
                "Members": STAMP_LINE,
                "Instance_ID": inst["instance_id"],
                "Grid_Size": f"{inst['grid_size']}x{inst['grid_size']}",
                "Python_Loop_us": round(py_t * 1e6, 2),
                "NumPy_Vectorized_us": round(np_t * 1e6, 2),
                "Speedup_Factor": round(speedup, 2),
            }
        ]
    )
    save_stamped_csv(df_part_a, "table1_part_a_benchmark.csv")

    # -------------------------------------------------------------------------
    # PART B: Search Comparison & Heuristic Ablations
    # Runs BFS, UCS, Greedy, A* (with various heuristics), and memory-guarded IDS.
    # Generates Table 2 and Figure 1 (nodes expanded horizontal bar chart).
    # -------------------------------------------------------------------------
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
        enable_fragile_mud=True,
    )

    checked, violations = test_heuristic_consistency(problem, sample_states=1000)
    print(f"[Part B] Empirical Consistency Check: {checked} edges checked, {violations} violations.")

    planners = [
        ("BFS", lambda: bfs(problem)),
        ("UCS", lambda: ucs(problem)),
        ("Greedy (Custom h)", lambda: greedy_best_first(problem, problem.h_custom_admissible)),
        ("A* (Custom Admissible MST+Tunnel)", lambda: astar(problem, problem.h_custom_admissible)),
        ("A* (Ablation: Nearest-Stop Only)", lambda: astar(problem, problem.h_baseline_nearest)),
        ("A* (Baseline: Farthest-Stop)", lambda: astar(problem, problem.h_baseline_farthest)),
        ("A* (Inadmissible 3.5x Fast)", lambda: astar(problem, problem.h_inadmissible_fast)),
        ("IDS (Memory-Limited 15k states)", lambda: ids(problem, max_depth=35, memory_limit_states=15000)),
    ]

    part_b_rows = []
    for label, fn in planners:
        r = fn()
        part_b_rows.append(
            {
                "Members": STAMP_LINE,
                "Algorithm": label,
                "Found": r["found"],
                "Solution_Cost": round(r["cost"], 2) if r["found"] else "INF",
                "Nodes_Expanded": r["expanded"],
                "Runtime_ms": r["runtime_ms"],
                "Peak_Memory_KB": r["peak_kb"],
            }
        )
    df_part_b = pd.DataFrame(part_b_rows)
    save_stamped_csv(df_part_b, "table2_part_b_search_comparison.csv")

    # Render Figure 1: Horizontal bar plot of nodes expanded (log scale)
    # Figure 1: Part B Nodes Expanded Comparison (A* Variants & Baselines)
    fig1, ax1 = plt.subplots(figsize=(9.0, 4.8))
    plot_rows = [row for row in part_b_rows if row["Found"] is True]
    if not plot_rows:
        # Fallback: plot all planners' expanded nodes so the figure is never blank
        plot_rows = part_b_rows
    colors = ["#2563eb" if r["Found"] else "#f59e0b" for r in plot_rows]
    bars = ax1.barh([r["Algorithm"] for r in plot_rows], [max(1, r["Nodes_Expanded"]) for r in plot_rows], color=colors)
    ax1.set_xlabel("Nodes Expanded (Log Scale)")
    ax1.set_xscale("log")
    ax1.set_title(f"{STAMP_LINE}\nPart B: Search Efficiency & Heuristic Ablation Study")
    for bar, r in zip(bars, plot_rows):
        status_tag = f"Cost={r['Solution_Cost']}" if r["Found"] else "Cap Reached"
        ax1.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{r['Nodes_Expanded']:,} ({status_tag})",
            va="center",
            fontsize=8,
        )
    ax1.set_xlim(right=max(r["Nodes_Expanded"] for r in plot_rows) * 4.5)
    plt.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "fig1_part_b_ablation.png"), dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # PART C: Adversarial Search (Minimax vs. Alpha-Beta & 30-Game Tournament)
    # Compares tree pruning efficiency, tunes weights, and simulates 30 stochastic matches.
    # Generates Table 3, Table 4, and Figure 2 (pruning curve).
    # -------------------------------------------------------------------------
    game = InterceptorGame(
        grid=grid,
        cost_map=cost_map,
        start=inst["start"],
        stops=inst["delivery_stops"],
        tunnels=twists["tunnels"],
        chargers=twists["chargers"],
        max_fuel=twists["max_fuel"],
        enable_fragile_mud=True,
    )
    s0 = game.initial_state()
    max_d = int(inst["adversary_search_depth"])
    ab_rows = []
    for d in range(2, max_d + 1):
        _, _, mm_nodes = minimax_search(game, s0, depth=d)
        _, _, ab_nodes = alphabeta_search(game, s0, depth=d)
        saved = ((mm_nodes - ab_nodes) / max(1, mm_nodes)) * 100.0
        ab_rows.append(
            {
                "Members": STAMP_LINE,
                "Depth": d,
                "Minimax_Nodes": mm_nodes,
                "AlphaBeta_Nodes": ab_nodes,
                "Nodes_Saved_Pct": round(saved, 2),
            }
        )
    df_ab = pd.DataFrame(ab_rows)
    save_stamped_csv(df_ab, "table3_part_c_pruning.csv")

    # Render Figure 2: Line chart comparing Minimax vs. Alpha-Beta growth across depths
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.plot(df_ab["Depth"], df_ab["Minimax_Nodes"], marker="o", label="Minimax", color="#dc2626")
    ax2.plot(df_ab["Depth"], df_ab["AlphaBeta_Nodes"], marker="s", label="Alpha-Beta (Move Ordered)", color="#16a34a")
    ax2.set_yscale("log")
    ax2.set_xlabel("Search Depth")
    ax2.set_ylabel("Nodes Evaluated (Log Scale)")
    ax2.set_title(f"{STAMP_LINE}\nPart C: Minimax vs. Alpha-Beta Pruning ({inst['adversary'].title()} Rival)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "fig2_part_c_pruning.png"), dpi=300)
    plt.close(fig2)

    # Tune evaluation weights and run 30-game tournament under environmental slip
    tuned_w = tune_evaluation_weights(game, slip_prob=inst["slip_probability"])
    stats, avg_rep, avg_extra, results = run_tournament(
        game, weights=tuned_w, num_games=30, search_depth=4, slip_prob=inst["slip_probability"]
    )
    avg_collapsed = float(np.mean([r.get("collapsed_mud_count", 0) for r in results]))
    df_tourn = pd.DataFrame(
        [
            {
                "Members": STAMP_LINE,
                "Outcome": k,
                "Rate_Pct": round(v[0], 2),
                "CI95_Low": round(v[1], 2),
                "CI95_High": round(v[2], 2),
                "Avg_Replans_Per_Game": round(avg_rep, 2),
                "Avg_Extra_Slip_Cost": round(avg_extra, 2),
                "Avg_Collapsed_Mud_Tiles": round(avg_collapsed, 2),
            }
            for k, v in stats.items()
        ]
    )
    save_stamped_csv(df_tourn, "table4_part_c_tournament_30games.csv")

    # -------------------------------------------------------------------------
    # PART D: LLM Dispatcher Benchmark (20 Requests)
    # Tests the agent across all 20 prompts with and without the validation guard.
    # Saves Table 5 summarizing success rate and average tool calls.
    # -------------------------------------------------------------------------
    tk = DispatcherToolkit(inst, grid)
    unval_res = [LLMDispatcherAgent(tk, use_validator=False).run_request(r) for r in DISPATCHER_REQUESTS]
    val_res = [LLMDispatcherAgent(tk, use_validator=True).run_request(r) for r in DISPATCHER_REQUESTS]

    df_llm = pd.DataFrame(
        [
            {
                "Members": STAMP_LINE,
                "Configuration": "LLM Dispatcher (Unvalidated)",
                "Success_Count": sum(r["success"] for r in unval_res),
                "Total_Requests": len(DISPATCHER_REQUESTS),
                "Success_Rate_Pct": round(100.0 * sum(r["success"] for r in unval_res) / len(DISPATCHER_REQUESTS), 1),
                "Avg_Tool_Calls": round(float(np.mean([r["tool_calls"] for r in unval_res])), 2),
            },
            {
                "Members": STAMP_LINE,
                "Configuration": "LLM Dispatcher (With Validation Layer)",
                "Success_Count": sum(r["success"] for r in val_res),
                "Total_Requests": len(DISPATCHER_REQUESTS),
                "Success_Rate_Pct": round(100.0 * sum(r["success"] for r in val_res) / len(DISPATCHER_REQUESTS), 1),
                "Avg_Tool_Calls": round(float(np.mean([r["tool_calls"] for r in val_res])), 2),
            },
        ]
    )
    save_stamped_csv(df_llm, "table5_part_d_llm_dispatcher.csv")

    # -------------------------------------------------------------------------
    # PART E: 30-Scenario Multi-Configuration Experiment
    # Tests 5 different algorithm and twist ablation configurations across 30
    # randomly generated, reachable start/stop delivery setups with 95% CIs.
    # Saves Table 6.
    # -------------------------------------------------------------------------
    base_seed = int(inst["instance_id"], 16) % 100000
    rng = np.random.default_rng(base_seed)
    walkable = np.argwhere(grid >= 0)
    grid_list = grid.tolist()

    # Pre-generate 30 reachable delivery missions
    scenarios = []
    while len(scenarios) < 30:
        pick = walkable[rng.choice(len(walkable), size=4, replace=False)]
        s_start = tuple(map(int, pick[0]))
        s_stops = [tuple(map(int, p)) for p in pick[1:]]
        if reachable(grid_list, s_start, s_stops):
            scenarios.append((s_start, s_stops))

    # Ablation configurations to compare
    configs = [
        ("A* (Full 3-Twist Custom Heuristic)", "custom", True, True, True),
        ("A* (Ablated: Fragile Mud Disabled)", "custom", True, True, False),
        ("A* (Ablated: No MST Term)", "nearest", True, True, True),
        ("A* (Ablated: Tunnels Disabled)", "custom", True, False, True),
        ("UCS (Uninformed Optimal)", "ucs", True, True, True),
    ]

    scenario_rows = []
    for cfg_name, mode, en_fuel, en_tun, en_mud in configs:
        costs, expandeds, runtimes = [], [], []
        for s_start, s_stops in scenarios:
            prob_s = DeliverySearchProblem(
                grid, cost_map, s_start, s_stops,
                twists["tunnels"], twists["chargers"], twists["max_fuel"],
                enable_fuel=en_fuel, enable_tunnels=en_tun, enable_fragile_mud=en_mud
            )
            if mode == "custom":
                out = astar(prob_s, prob_s.h_custom_admissible)
            elif mode == "nearest":
                out = astar(prob_s, prob_s.h_baseline_nearest)
            else:
                out = ucs(prob_s)

            if out["found"]:
                costs.append(out["cost"])
                expandeds.append(out["expanded"])
                runtimes.append(out["runtime_ms"])

        c_m, c_ci = mean_ci95(costs) if costs else (0.0, 0.0)
        e_m, e_ci = mean_ci95(expandeds) if expandeds else (0.0, 0.0)
        t_m, t_ci = mean_ci95(runtimes) if runtimes else (0.0, 0.0)
        scenario_rows.append(
            {
                "Members": STAMP_LINE,
                "Configuration": cfg_name,
                "Solved_Scenarios": f"{len(costs)}/30",
                "Cost_Mean_95CI": f"{c_m:.2f} ± {c_ci:.2f}",
                "Expanded_Mean_95CI": f"{e_m:.1f} ± {e_ci:.1f}",
                "Runtime_ms_Mean_95CI": f"{t_m:.2f} ± {t_ci:.2f}",
            }
        )

    df_part_e = pd.DataFrame(scenario_rows)
    save_stamped_csv(df_part_e, "table6_part_e_30scenarios_ci95.csv")
    print("=" * 90)
    print("All tables and figures generated successfully in ./outputs/")


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()


# =============================================================================
# FILE SUMMARY: run_all.py
# -----------------------------------------------------------------------------
# This file is the master automated experiment runner for the entire project:
# 1. Part A: Benchmarks cost-map generation (Python vs NumPy) -> Table 1.
# 2. Part B: Runs search algorithms and heuristic ablations -> Table 2 & Figure 1.
# 3. Part C: Measures Minimax vs. Alpha-Beta node savings across depths and runs
#    a 30-game stochastic tournament -> Table 3, Table 4 & Figure 2.
# 4. Part D: Benchmarks the LLM dispatcher across 20 prompts (with/without validator) -> Table 5.
# 5. Part E: Tests 5 configurations over 30 random reachable delivery scenarios,
#    computing 95% confidence intervals on costs, nodes expanded, and time -> Table 6.
# =============================================================================