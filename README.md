# Gurobi Optimization Model for Battery Electric Bus Scheduling

This Python script formulates and solves a Mixed-Integer Linear Program (MILP) for **Battery Electric Bus (BEB) scheduling** using a Spatial-Temporal-SOC network.

---

## 🚍 Problem Overview

The script reads a directed graph from a pre-generated CSV file (`FinalLinks.csv`) where each **node** represents a BEB's location, time, and SOC (State of Charge), and each **arc** corresponds to an operational activity (trip, deadhead, idle, charging, etc.).

It solves a flow-based optimization model that determines how BEBs should move through this network to:
- Cover all scheduled trips
- Respect flow conservation across intermediate nodes
- Conserve vehicle balance between depot and sink nodes
- Satisfy charging station capacity constraints

---

## 🛠️ Features

- ✔️ Flow conservation at all internal nodes
- ✔️ Trip coverage (each trip is covered exactly once)
- ✔️ Charging station time-dependent capacity constraints
- ✔️ Depot and sink flow balancing for each SOC level
- ✔️ Gurobi-based cost minimization

---

## 📁 Input

- `FinalLinks.csv`: Output from the spatial-temporal-SOC network generator containing all nodes and arcs with cost, duration, energy, and type.

---

## 🧾 Output

- `model.lp`: The optimization model in LP format
- `solution.sol`: Gurobi solution file
- `solutionNew1.txt`: Readable text file listing flow on each active arc

---

## ⚙️ Optimization Settings

- Time limit: 80,000 seconds
- MIP gap: 1%
- Heuristics: Enabled (50% time allocation)

To customize solver behavior (threads, cuts, focus, presolve), you can uncomment and tune the relevant lines in the script.

---

## ✅ Dependencies

- Python ≥ 3.7
- Gurobi
- pandas
- networkx
- matplotlib

Install them with:

```bash
pip install gurobipy pandas networkx matplotlib
```

> Note: Gurobi requires a valid license (academic licenses available at [gurobi.com](https://www.gurobi.com)).

---

## 📌 How to Run

Ensure `FinalLinks.csv` is in the same directory, then run:

```bash
python gurobi_model.py
```

---

## 👨‍💻 Author

Behnam Emami

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
