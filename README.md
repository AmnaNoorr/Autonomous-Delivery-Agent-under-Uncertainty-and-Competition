# Autonomous Delivery Agent under Uncertainty and Competition
---

## 1. Project Overview

This repository contains the complete implementation of an autonomous delivery agent operating in a complex, competitive grid city under environmental uncertainty and adversarial interference. 

### Unique Instance Specification
- **Grid Size:** $35 \times 35$ tiles (1,225 total cells).
- **Start Depot:** $(24, 23)$.
- **Delivery Stops:** 3 parcel targets at $(16, 13)$, $(21, 29)$, and $(19, 29)$.
- **Terrain Costs:** Road ($3$), Grass ($4$), Mud ($6$), Ford ($8$), Obstacles ($-1$ / $\infty$).
- **Adversary Type:** Interceptor rival with search depth limit $6$.
- **Slip Probability:** $0.15$ ($15\%$ chance of unexpected movement slip).

### Assigned & Proposed Twists
1. **Twist 1 — Teleport Tunnels:** Bidirectional portal pairs connecting $(33, 16) \leftrightarrow (19, 27)$ and $(22, 15) \leftrightarrow (34, 29)$ with traversal cost $1.0$.
2. **Twist 2 — Fuel Limit with Charging Stations:** Max tank capacity of $87$ fuel units with replenishment stations placed at $(5, 3)$, $(19, 0)$, and $(23, 28)$.
3. **Twist 3 (Group-Proposed) — Fragile Mud Cells (Degrading Terrain):** 194 mud cells that wear down when traversed. The 1st crossing incurs $2\times$ base cost ($12$ units) and marks the tile damaged. A 2nd crossing causes the cell to collapse into a permanent obstacle ($-1$), dynamically altering the board topology.

---

## 2. Repository Structure

```text
.
├── config.py                  # Student instance generator and team stamp configuration
├── oel_instance.py            # Instructor-provided instance generation script
├── part_a_world.py            # World model, twist generation, and vectorized NumPy benchmarks
├── part_b_search.py           # From-scratch search planners (BFS, UCS, DLS, IDS, Greedy, A*) & heuristic
├── part_c_adversarial.py      # Interceptor rival model, Minimax vs Alpha-Beta, tuning & 30-game tournament
├── part_d_llm.py              # LLM dispatcher toolkit, argument validation guardrail & 20-prompt benchmark
├── run_all.py                 # Master one-command script reproducing all tables and figures
├── app.py                     # Streamlit interactive command-center dashboard
├── requirements.txt           # Project dependencies
├── outputs/                   # Generated evaluation artifacts
│   ├── table1_part_a_benchmark.csv
│   ├── table2_part_b_search_comparison.csv
│   ├── fig1_part_b_ablation.png
│   ├── table3_part_c_pruning.csv
│   ├── fig2_part_c_pruning.png
│   ├── table4_part_c_tournament_30games.csv
│   ├── table5_part_d_llm_dispatcher.csv
│   └── table6_part_e_30scenarios_ci95.csv
```

---

## 3. Installation & Setup

1. **Clone the repository and navigate into the workspace:**
   ```bash
   cd "e:/MCS/Semester 7/Artificial Intelligence/Lab/oel"
   ```

2. **Create and activate a Python virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Execution & Reproducibility

### A. One-Command Master Pipeline (Reproduce All Results)
To execute all benchmarks and recreate all report tables/figures:
```bash
python run_all.py
```
This runs Parts A through E and saves stamped CSVs and high-resolution plots to `./outputs/`.

### B. Interactive Streamlit Dashboard
<img width="917" height="407" alt="image" src="https://github.com/user-attachments/assets/4b53122c-b2d3-4c1e-8d0b-d5b19dd80e17" />

To launch the graphical command center:
```bash
streamlit run app.py
```
**Dashboard Features:**
- **Simulation Controls (Left):** Select algorithms (A*, Alpha-Beta, Minimax, Greedy, BFS, UCS, DLS, IDS), adjust slip probability, modify rival search depth, and submit natural-language dispatcher queries.
- **Grid City Visualization (Center):** Real-time color-coded board displaying terrain costs, depot ($D$), delivery targets ($1..N$), rival drone ($R$), autonomous agent ($A$), chargers ($\triangle$), teleport portals ($\diamond$), damaged mud ($!$), collapsed mud ($X$), planned routes, and executed traces.
- **Scrubber Bar:** Step backward and forward through history with turn-by-turn dynamic replanning.
- **Telemetry & Validation (Right):** Live 4-card KPI grid (Delivered, Fuel, Cost, Replans), benchmark comparison matrix, and LLM Thought $\rightarrow$ Action $\rightarrow$ Observation trace with parameter validation badges.

---

## 5. Summary of Experimental Results

1. **Part A (Vectorization Speedup):** Vectorized NumPy matrix operations achieved a **19.30x speedup** over pure Python loops ($42.89\ \mu\text{s}$ vs $827.78\ \mu\text{s}$).
2. **Part B (Heuristic & Search):** Custom tunnel-aware MST heuristic proved strictly admissible and consistent ($0$ violations across $1,000$ checked state transitions).
3. **Part C (Adversarial Pruning):** Alpha-Beta with move ordering saved **$87.05\%$ of search nodes** at depth 6 compared to standard Minimax ($141$ vs $1,089$ nodes).
4. **Part D (LLM Guardrails):** The `ArgumentValidator` successfully intercepted 100% of out-of-bound requests and hallucinated arguments, safely recovering with valid executions.
5. **Part E (Statistical Ablations):** Evaluated over 30 reachable scenarios with $95\%$ confidence intervals, confirming optimal pathing and efficiency gains.
   
<img width="482" height="261" alt="Screenshot 2026-11-23 210746" src="https://github.com/user-attachments/assets/8f96a763-fd7a-4fc1-bddb-cca3d784f0ff" />    

<img width="341" height="207" alt="Screenshot 2026-11-23 205811" src="https://github.com/user-attachments/assets/75bddd09-f7c6-4293-9687-b0206484040f" />

