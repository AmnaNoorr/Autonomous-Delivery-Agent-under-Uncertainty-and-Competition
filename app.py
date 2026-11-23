# app.py
# -----------------------------------------------------------------------------
# Imports
# Standard tools for time, array math, and plotting (matplotlib), alongside Streamlit
# for the web dashboard, plus our world generation, search planners, adversarial models,
# and LLM dispatcher from Parts A, B, C, and D.
# -----------------------------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import streamlit as st

from config import MEMBERS, inst, grid, stamp
from part_a_world import generate_twist_features, build_cost_map_vectorized
from part_b_search import DeliverySearchProblem, astar, ucs, greedy_best_first, bfs, ids, dls
from part_c_adversarial import InterceptorGame, alphabeta_search, minimax_search
from part_d_llm import DispatcherToolkit, LLMDispatcherAgent


# -----------------------------------------------------------------------------
# Page Configuration & Dark Command-Center Theme
# Sets browser tab title, wide layout, and custom CSS styling for panels,
# cards, buttons, fonts, and dark theme colors.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CS-370 OEL — Autonomous Delivery Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STAMP_LINE = stamp(MEMBERS)

st.markdown(
    """
    <style>
    /* Main dark background */
    .stApp {
        background-color: #16181d;
        color: #e2e8f0;
    }
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    /* Panel card styling */
    .panel-box {
        background-color: #1f2229;
        border: 1px solid #2e333d;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .panel-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
        border-bottom: 1px solid #2e333d;
        padding-bottom: 6px;
    }
    /* Corner & Header CMS stamps */
    .corner-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.76rem;
        color: #94a3b8;
        font-family: monospace;
        margin-bottom: 2px;
    }
    .header-banner {
        text-align: center;
        padding: 4px 0 12px 0;
        border-bottom: 1px solid #2e333d;
        margin-bottom: 14px;
    }
    .header-banner h1 {
        font-size: 1.48rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .header-banner p {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin: 4px 0 0 0;
    }
    /* 4-Card KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }
    .kpi-card {
        background-color: #262a33;
        border: 1px solid #353b48;
        border-radius: 6px;
        padding: 8px 6px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #94a3b8;
        line-height: 1.2;
    }
    .kpi-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.65rem;
        color: #94a3b8;
    }
    /* Legend badges */
    .legend-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 9px;
        font-size: 0.78rem;
        color: #cbd5e1;
        line-height: 1.25;
    }
    .dot {
        min-width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.68rem;
        font-weight: 700;
        color: #fff;
        margin-top: 1px;
    }
    /* Validation banners */
    .val-pass {
        background-color: #163828;
        border: 1px solid #22c55e;
        color: #bbf7d0;
        padding: 8px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-top: 8px;
    }
    .val-fail {
        background-color: #3f181c;
        border: 1px solid #ef4444;
        color: #fecaca;
        padding: 8px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-top: 8px;
    }
    /* Dark inputs & buttons */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #262a33 !important;
        color: #f8fafc !important;
        border-color: #3b4252 !important;
    }
    .stButton > button {
        background-color: #2b303b;
        color: #f8fafc;
        border: 1px solid #434b5c;
        border-radius: 5px;
        width: 100%;
        font-size: 0.82rem;
    }
    .stButton > button:hover {
        border-color: #60a5fa;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header Banner and Team Identification
# Displays the group member names and CMS IDs across the top of the interface.
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="corner-row">
        <span>{STAMP_LINE}</span>
        <span>{STAMP_LINE}</span>
    </div>
    <div class="header-banner">
        <h1>CS-370 Artificial Intelligence — Open Ended Lab (OEL): Autonomous Delivery Agent</h1>
        <p>Group Members: {STAMP_LINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Simulation State & Backend Logic
# Sets up grid twists, cost maps, session state variables (random seed, timeline cursor),
# and cached live performance comparison metrics.
# -----------------------------------------------------------------------------
base_twists = generate_twist_features(inst, grid)
cost_map = build_cost_map_vectorized(grid, inst["terrain_cost"])

if "sim_seed" not in st.session_state:
    st.session_state.sim_seed = int(inst["instance_id"], 16) % 10000
if "step_cursor" not in st.session_state:
    st.session_state.step_cursor = None
if "llm_result" not in st.session_state:
    tk_init = DispatcherToolkit(inst, grid)
    st.session_state.llm_result = LLMDispatcherAgent(tk_init, use_validator=True).run_request(
        {
            "id": 13,
            "difficulty": "Hard",
            "prompt": "Deliver to stop 3 first, avoid mud, use less than 60 fuel",
            "gt_args": {"stop_order": [3], "avoid_terrains": ["mud"], "max_fuel_budget": 60},
        }
    )


# Caches baseline search benchmarks so they compute once and render instantly
@st.cache_data(show_spinner=False)
def compute_comparison_metrics():
    """Computes live benchmark rows for the Performance Comparison Panel."""
    prob = DeliverySearchProblem(
        grid=grid,
        cost_map=cost_map,
        start=inst["start"],
        stops=inst["delivery_stops"],
        tunnels=base_twists["tunnels"],
        chargers=base_twists["chargers"],
        max_fuel=base_twists["max_fuel"],
        enable_fragile_mud=True,
    )
    rows = []
    for label, fn in [
        ("BFS", lambda: bfs(prob)),
        ("A*", lambda: astar(prob, prob.h_custom_admissible)),
        ("Greedy", lambda: greedy_best_first(prob, prob.h_custom_admissible)),
        ("UCS", lambda: ucs(prob)),
    ]:
        r = fn()
        rows.append(
            {
                "Algorithm": label,
                "Nodes Expanded": r["expanded"],
                "Runtime (ms)": round(r["runtime_ms"], 2),
            }
        )
    return rows


# Helper: runs whatever algorithm the user selected from the dropdown menu
def _solve_with_selected_algo(algo_label: str, prob: DeliverySearchProblem):
    """Dispatches strictly to the chosen Part B algorithm without falling back to A*."""
    if algo_label.startswith("A*"):
        return astar(prob, prob.h_custom_admissible)
    elif algo_label.startswith("UCS"):
        return ucs(prob)
    elif algo_label.startswith("Greedy"):
        return greedy_best_first(prob, prob.h_custom_admissible)
    elif algo_label.startswith("BFS"):
        # BFS treats all walkable terrain as equal cost 1.0
        unit_cost_map = np.where(prob.grid >= 0, 1.0, np.inf)
        unit_prob = DeliverySearchProblem(
            prob.grid,
            unit_cost_map,
            prob.start,
            prob.stops,
            prob.tunnels,
            prob.chargers,
            prob.max_fuel,
            enable_fragile_mud=True,
        )
        return bfs(unit_prob)
    elif algo_label.startswith("IDS"):
        return ids(prob, max_depth=40, memory_limit_states=30000, max_expansions=40000)
    else:  # DLS
        return dls(prob, depth_limit=35, memory_limit_states=30000, max_expansions=40000)


# Runs a step-by-step interactive simulation match, recording each turn for the scrubber
def run_simulation_episode(algo_label: str, slip_p: float, r_depth: int, seed: int):
    rng = np.random.default_rng(seed)
    start = inst["start"]
    stops = inst["delivery_stops"]
    tunnels = base_twists["tunnels"]
    chargers = base_twists["chargers"]
    max_fuel = base_twists["max_fuel"]

    planned_path = [start]
    is_adversarial = any(k in algo_label for k in ("Alpha-Beta", "Minimax"))

    # If using standard pathfinding (Part B), precompute route across stops
    if not is_adversarial:
        prob = DeliverySearchProblem(
            grid, cost_map, start, stops, tunnels, chargers, max_fuel, enable_fragile_mud=True
        )
        res = _solve_with_selected_algo(algo_label, prob)
        if res.get("found") and res.get("path"):
            planned_path = res["path"]
        else:
            # If multi-stop search hits limits, plan stop-by-stop
            curr_p = start
            curr_f = max_fuel
            rem = list(stops)
            stitched = [start]
            while rem:
                nxt_stop = min(rem, key=lambda s: abs(s[0] - curr_p[0]) + abs(s[1] - curr_p[1]))
                seg_prob = DeliverySearchProblem(
                    grid, cost_map, curr_p, [nxt_stop], tunnels, chargers, curr_f, enable_fragile_mud=True
                )
                seg_res = _solve_with_selected_algo(algo_label, seg_prob)
                if seg_res.get("found") and len(seg_res["path"]) > 1:
                    stitched.extend(seg_res["path"][1:])
                    curr_p = nxt_stop
                    rem.remove(nxt_stop)
                else:
                    break
            planned_path = stitched

    # Spawn rival at the farthest walkable point
    walkable = np.argwhere(grid >= 0)
    dists = np.abs(walkable[:, 0] - start[0]) + np.abs(walkable[:, 1] - start[1])
    rival_pos = tuple(map(int, walkable[np.argmax(dists)]))

    game = InterceptorGame(
        grid, cost_map, start, stops, tunnels, chargers, max_fuel, enable_fragile_mud=True
    )
    state = game.initial_state(rival_pos=rival_pos)

    history = [
        {
            "step": 0,
            "agent_pos": state.agent_pos,
            "rival_pos": state.rival_pos,
            "fuel": state.fuel,
            "total_cost": 0,
            "replans": 0,
            "delivered": bin(state.delivered_mask).count("1"),
            "planned_path": list(planned_path),
            "rival_path": [state.rival_pos],
            "damaged_mud": set(state.damaged_mud),
            "collapsed_mud": set(state.collapsed_mud),
        }
    ]

    total_cost = 0
    replans = 0
    path_idx = 1
    rival_trace = [rival_pos]

    # Advance game up to 75 steps
    for step in range(1, 75):
        if game.is_terminal(state, max_steps=75):
            break

        legal = dict(game.get_legal_moves(state))
        if is_adversarial:
            d_cap = min(r_depth, 4)
            if "Alpha-Beta" in algo_label:
                act, _, _ = alphabeta_search(game, state, depth=d_cap)
            else:
                act, _, _ = minimax_search(game, state, depth=min(d_cap, 3))
            intended_state = legal.get(act, next(iter(legal.values())))
        else:
            if path_idx < len(planned_path):
                target_cell = planned_path[path_idx]
                act = next((a for a, s_n in legal.items() if s_n.agent_pos == target_cell), None)
                if act is None:
                    # Dynamic replanning when path is blocked or off course
                    dynamic_cost_map = build_cost_map_vectorized(
                        grid,
                        inst["terrain_cost"],
                        damaged_mud=state.damaged_mud,
                        collapsed_mud=state.collapsed_mud,
                    )
                    rem_stops = [s for i, s in enumerate(stops) if not (state.delivered_mask & (1 << i))]
                    target_stop = (
                        min(rem_stops, key=lambda s: abs(s[0] - state.agent_pos[0]) + abs(s[1] - state.agent_pos[1]))
                        if rem_stops
                        else start
                    )
                    sub_p = DeliverySearchProblem(
                        grid,
                        dynamic_cost_map,
                        state.agent_pos,
                        [target_stop],
                        tunnels,
                        chargers,
                        state.fuel,
                        enable_fragile_mud=True,
                    )
                    re_res = _solve_with_selected_algo(algo_label, sub_p)
                    if re_res.get("found") and len(re_res["path"]) > 1:
                        planned_path = re_res["path"]
                        path_idx = 1
                        target_cell = planned_path[1]
                        act = next(
                            (a for a, s_n in legal.items() if s_n.agent_pos == target_cell),
                            next(iter(legal.keys())),
                        )
                    else:
                        act = next(iter(legal.keys()))
                else:
                    path_idx += 1
            else:
                act = next(iter(legal.keys()))
            intended_state = legal[act]

        # Environmental slip check
        if rng.random() < slip_p and len(legal) > 1:
            alt_keys = [k for k in legal.keys() if k != act]
            chosen_act = alt_keys[int(rng.integers(len(alt_keys)))]
            after_agent = legal[chosen_act]
            if chosen_act != act:
                replans += 1
        else:
            chosen_act = act
            after_agent = intended_state

        step_c = 1 if chosen_act == "TELEPORT" else max(1, state.fuel - after_agent.fuel)
        total_cost += step_c

        # Rival movement turn
        rival_moves = game.get_legal_moves(after_agent)
        state = min(rival_moves, key=lambda item: game.evaluate(item[1]))[1] if rival_moves else after_agent
        rival_trace.append(state.rival_pos)

        # Snapshot current step state into history
        history.append(
            {
                "step": step,
                "agent_pos": state.agent_pos,
                "rival_pos": state.rival_pos,
                "fuel": state.fuel,
                "total_cost": total_cost,
                "replans": replans,
                "delivered": bin(state.delivered_mask).count("1"),
                "planned_path": list(planned_path),
                "rival_path": list(rival_trace),
                "damaged_mud": set(state.damaged_mud),
                "collapsed_mud": set(state.collapsed_mud),
            }
        )

    return history


# -----------------------------------------------------------------------------
# Three-Column Dashboard Layout
# Column 1 (Left): Simulation Controls & LLM Prompt Input
# Column 2 (Center): Grid City Visual Map & Legend
# Column 3 (Right): Real-Time KPIs, Performance Table, and Validation Trace
# -----------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1.05, 2.35, 1.35], gap="medium")


# ==================== LEFT COLUMN: Simulation Controls ====================
with col_left:
    st.markdown('<div class="panel-box"><div class="panel-title">Simulation Controls</div>', unsafe_allow_html=True)

    # Algorithm selection dropdown
    algo_choice = st.selectbox(
        "Algorithm",
        [
            "A* Search",
            "Alpha-Beta",
            "Minimax",
            "Greedy Best-First",
            "BFS",
            "UCS",
            "DLS",
            "IDS",
        ],
    )

    # Sliders for slip chance and rival search depth
    slip_prob = st.slider("Slip Probability", 0.0, 0.50, float(inst["slip_probability"]), 0.05)
    rival_depth = st.slider("Rival Search Depth", 1, 6, int(inst["adversary_search_depth"]), 1)

    # Simulation execution and playback buttons
    st.markdown("**Playback & Simulation**")
    if st.button("Run Simulation"):
        st.session_state.sim_seed += 1
        st.session_state.step_cursor = None

    b1, b2 = st.columns(2)
    with b1:
        step_fwd_clicked = st.button("Step Forward")
    with b2:
        reset_clicked = st.button("Reset")

    # Interactive Natural-Language LLM Dispatcher Text Box
    st.markdown("<hr style='border-color:#2e333d;margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("**LLM Dispatcher Text Box**")
    llm_prompt = st.text_area(
        "LLM Dispatcher Prompt (Lab 2)",
        value="Deliver to stop 3 first, avoid mud, use less than 60 fuel",
        height=85,
    )
    if st.button("Submit Prompt"):
        tk = DispatcherToolkit(inst, grid)
        agent = LLMDispatcherAgent(tk, use_validator=True)
        st.session_state.llm_result = agent.run_request(
            {"id": 99, "difficulty": "Interactive", "prompt": llm_prompt, "gt_args": {}}
        )

    st.markdown("</div>", unsafe_allow_html=True)


# Run episode with selected controls and handle timeline scrubber state
history = run_simulation_episode(algo_choice, slip_prob, rival_depth, st.session_state.sim_seed)
max_idx = len(history) - 1

if reset_clicked:
    st.session_state.step_cursor = 0
elif step_fwd_clicked:
    curr = 0 if st.session_state.step_cursor is None else st.session_state.step_cursor
    st.session_state.step_cursor = min(max_idx, curr + 1)
elif st.session_state.step_cursor is None or st.session_state.step_cursor > max_idx:
    st.session_state.step_cursor = max_idx

snap = history[st.session_state.step_cursor]


# ==================== CENTER COLUMN: Grid City Visualization ====================
with col_center:
    n = inst["grid_size"]
    st.markdown(
        f'<div class="panel-box"><div class="panel-title">Grid City Visualization ({n}x{n})</div>',
        unsafe_allow_html=True,
    )

    grid_col, leg_col = st.columns([2.25, 1.0], gap="small")

    # Render matplotlib board with all overlays
    with grid_col:
        fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="#1f2229")
        ax.set_facecolor("#1f2229")

        # Color-coded terrain tiles: -1 obstacle, 0 road, 1 grass, 2 mud, 3 ford
        display_grid = grid + 1
        cmap = ListedColormap(["#4b5563", "#e5e7eb", "#a7f3d0", "#fca5a5", "#cbd5e1"])
        ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=4)

        # Twist 3 Overlays: Orange '!' for damaged mud; Dark 'X' for collapsed mud
        for mr, mc in snap.get("damaged_mud", set()):
            ax.scatter(mc, mr, marker="s", s=110, color="#f97316", alpha=0.85, zorder=2)
            ax.text(mc, mr, "!", color="white", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=3)

        for cr, cc in snap.get("collapsed_mud", set()):
            ax.scatter(cc, cr, marker="s", s=135, color="#374151", edgecolor="#ef4444", linewidth=0.9, zorder=2)
            ax.text(cc, cr, "X", color="#fca5a5", fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=3)

        # Route lines: blue planned path, red rival trajectory, yellow executed path
        p_path = snap["planned_path"]
        if len(p_path) > 1:
            ax.plot([p[1] for p in p_path], [p[0] for p in p_path], color="#3b82f6", linewidth=2.2, zorder=3)

        r_path = snap["rival_path"]
        if len(r_path) > 1:
            ax.plot([p[1] for p in r_path], [p[0] for p in r_path], color="#ef4444", linewidth=2.0, zorder=3)

        a_trace = [h["agent_pos"] for h in history[: st.session_state.step_cursor + 1]]
        if len(a_trace) > 1:
            ax.plot([p[1] for p in a_trace], [p[0] for p in a_trace], color="#facc15", linewidth=2.4, zorder=4)

        # Twist 2: Charging stations (cyan triangles)
        for cr, cc in base_twists["chargers"]:
            ax.scatter(cc, cr, marker="^", s=95, color="#06b6d4", edgecolor="#0f172a", linewidth=0.8, zorder=5)

        # Twist 1: Teleport tunnels (purple diamonds + dashed lines)
        drawn_pairs = set()
        for a, b in base_twists["tunnels"].items():
            pair = tuple(sorted([a, b]))
            if pair not in drawn_pairs:
                drawn_pairs.add(pair)
                ax.plot([a[1], b[1]], [a[0], b[0]], linestyle="--", color="#a855f7", alpha=0.55, linewidth=1.2, zorder=3)
                ax.scatter([a[1], b[1]], [a[0], b[0]], marker="D", s=85, color="#a855f7", edgecolor="#ffffff", linewidth=0.8, zorder=5)

        # Start Depot (blue circle 'D')
        sr, sc = inst["start"]
        ax.scatter(sc, sr, marker="o", s=190, color="#2563eb", edgecolor="#ffffff", linewidth=1.2, zorder=6)
        ax.text(sc, sr, "D", color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=7)

        # Delivery targets (green numbered circles 1..N)
        for idx, (dr, dc) in enumerate(inst["delivery_stops"]):
            ax.scatter(dc, dr, marker="o", s=190, color="#16a34a", edgecolor="#ffffff", linewidth=1.1, zorder=6)
            ax.text(dc, dr, str(idx + 1), color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=7)

        # Rival drone (red circle 'R')
        rr, rc = snap["rival_pos"]
        ax.scatter(rc, rr, marker="o", s=200, color="#dc2626", edgecolor="#ffffff", linewidth=1.3, zorder=8)
        ax.text(rc, rr, "R", color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=9)

        # Delivery agent (yellow circle 'A')
        ar, ac = snap["agent_pos"]
        ax.scatter(ac, ar, marker="o", s=210, color="#facc15", edgecolor="#111827", linewidth=1.3, zorder=9)
        ax.text(ac, ar, "A", color="#111827", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=10)

        # Fine grid border ticks
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#475569", linestyle="-", linewidth=0.45)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Step slider below board for scrubbing back and forth through history
        new_step = st.slider("Playback Timeline Step", 0, max_idx, st.session_state.step_cursor, label_visibility="collapsed")
        if new_step != st.session_state.step_cursor:
            st.session_state.step_cursor = new_step
            st.rerun()

    # Visual map legend
    with leg_col:
        st.markdown(
            """
            <div style="font-weight:700;font-size:0.9rem;color:#f8fafc;margin-bottom:8px;">Grid City</div>
            <div class="legend-item"><span class="dot" style="background:#2563eb;">D</span><span><b>Depot</b> (Start)</span></div>
            <div class="legend-item"><span class="dot" style="background:#16a34a;">P</span><span><b>Parcels / Stops</b><br>(Numbers 1–N)</span></div>
            <div class="legend-item"><span class="dot" style="background:#dc2626;">R</span><span><b>Rival Agent</b><br>(Interceptor trace)</span></div>
            <div class="legend-item"><span class="dot" style="background:#facc15;color:#111;">A</span><span><b>Autonomous Agent</b><br>(with path trace)</span></div>
            <div class="legend-item"><span class="dot" style="background:#4b5563;">■</span><span><b>Obstacles</b> (-1)</span></div>
            <div class="legend-item"><span class="dot" style="background:#94a3b8;color:#111;">▦</span><span><b>Terrain Shading</b><br>Road, grass, mud, ford costs</span></div>
            <div class="legend-item"><span class="dot" style="background:#a855f7;">◆</span><span><b>Assigned Twists</b><br>Teleport portals &amp; fuel chargers</span></div>
            <div class="legend-item"><span class="dot" style="background:#f97316;">!</span><span><b>Fragile Mud (Twist 3)</b><br>! = 2x cost (hit 1), X = collapsed (hit 2)</span></div>
            <div style="margin-top:10px;font-size:0.75rem;color:#cbd5e1;">
                <span style="color:#3b82f6;font-weight:700;">━━</span> Planned Route<br>
                <span style="color:#ef4444;font-weight:700;">━━</span> Rival Route
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==================== RIGHT COLUMN: Real-Time Stats & Logs ====================
with col_right:
    # 1. 4-Card Live KPI Grid (Parcels, Fuel, Total Cost, Replans)
    total_stops = len(inst["delivery_stops"])
    max_f = base_twists["max_fuel"]
    st.markdown(
        f"""
        <div class="panel-box">
            <div class="panel-title">Real-Time Stats &amp; Logs</div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Parcels<br>Delivered</div>
                    <div class="kpi-val">{snap['delivered']}/{total_stops}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Current<br>Fuel</div>
                    <div class="kpi-val">{snap['fuel']}/{max_f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total<br>Cost</div>
                    <div class="kpi-val">{snap['total_cost']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Replans<br>Triggered</div>
                    <div class="kpi-val">{snap['replans']}</div>
                    <div class="kpi-sub">(due to slip)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Performance Comparison Table (BFS vs A* vs Greedy vs UCS)
    st.markdown('<div class="panel-box"><div class="panel-title">Performance Comparison Panel</div>', unsafe_allow_html=True)
    comp_rows = compute_comparison_metrics()
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. LLM Dispatcher Thought -> Action -> Observation Trace with Guardrail Banner
    llm_res = st.session_state.llm_result
    trace = llm_res.get("trace", [])
    thought_txt = next((t.get("thought", "") for t in trace if "thought" in t), "Parsed constraints.")
    act_step = next((t for t in trace if "executed_tool" in t or "proposed_tool" in t), {})
    tool_str = f"{act_step.get('executed_tool', act_step.get('proposed_tool', 'plan_route'))}({act_step.get('args', act_step.get('raw_args', {}))})"
    obs_step = next((t.get("observation", t.get("result", {})) for t in trace if "observation" in t or "result" in t), {})

    val_banner = (
        f'<div class="val-pass">✅ <b>Parameter Validation Passed:</b> Arguments verified against {n}x{n} map &amp; fuel limits.</div>'
        if not llm_res.get("caught_error")
        else '<div class="val-fail">⚠️ <b>Validator Intercepted Invalid Argument:</b> Recovered safely with fallback plan.</div>'
    )

    st.markdown(
        f"""
        <div class="panel-box">
            <div class="panel-title">LLM Dispatcher &amp; Validation Trace</div>
            <div style="font-size:0.78rem;line-height:1.45;color:#e2e8f0;font-family:monospace;">
                <div><b>Thought:</b> {thought_txt}</div>
                <div style="margin-top:4px;"><b>Action:</b> {tool_str}</div>
                <div style="margin-top:4px;"><b>Observation:</b> {obs_step}</div>
            </div>
            {val_banner}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Bottom Footer Stamp
# Permanent footer stamp with student names and CMS IDs.
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="corner-row" style="margin-top:6px;">
        <span>{STAMP_LINE}</span>
        <span>{STAMP_LINE}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FILE SUMMARY: app.py
# -----------------------------------------------------------------------------
# This file provides the interactive Streamlit command-center GUI:
# 1. Dashboard Layout: Configures a 3-column dark theme command center showing
#    controls on the left, an interactive grid map in the middle, and live KPIs/logs on the right.
# 2. Map & Overlay Rendering: Visualizes terrain shading, start depot, delivery stops,
#    charging stations, teleport portals, fragile mud degradation ('!' and 'X'),
#    planned paths, and rival interceptor traces using Matplotlib.
# 3. Interactive Playback: Supports step-by-step scrubbing through simulation history
#    with dynamic replanning whenever stochastic environmental slips occur.
# 4. LLM Dispatcher Integration: Includes an interactive prompt box displaying
#    the SENSE -> REASON -> ACT trace with real-time argument validation banners.
# =============================================================================