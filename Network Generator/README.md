# Spatial-Temporal-SOC Network Generator

This repository provides a complete implementation of a **Spatial-Temporal-State of Charge (SOC) network generator** for Battery Electric Bus (BEB) scheduling and fleet optimization. It constructs a time-expanded graph incorporating time, energy, and location dimensions based on concepts from Kliewer et al. (2006) and Zhang et al. (2021).

---

## 🚍 Problem Context

The generator creates a network where each node represents a location (depot, terminal, or charging station) at a specific time interval and SOC level. Arcs between nodes capture operational activities such as:

- **Trip execution**
- **Pull-out / Pull-in**
- **Deadheading (non-revenue travel)**
- **Idling**
- **Charging (with tariffs)**

This spatial-temporal-SOC network serves as the foundation for BEB scheduling models.

---

## 📂 Input Files

Place these CSV files in the same directory:

- `Bus_Trip_Timetable.csv` — BEB scheduled trips (Start/End times and terminals)
- `Terminals.csv` — Distance matrix between terminals
- `Charging stations.csv` — Distance matrix between charging stations and terminals
- `Charging_Costs_by_Time_Interval.csv` — Electricity tariffs per time interval

---

## 🛠️ How It Works

The script performs the following steps:

1. **Load and parse input data**
2. **Discretize the planning horizon into operational and non-operational intervals**
3. **Generate nodes for each location, time interval, and SOC level**
4. **Create arcs for each possible BEB activity**
5. **Export the nodes and links to CSV files**

---

## 📤 Output Files

The following files will be generated:

- `all_nodes.csv` — List of all network nodes
- `pull-In-Links.csv` — Pull-in arcs (return to depot)
- `pull-Out-Links.csv` — Pull-out arcs (start from depot)
- `Triplinknew.csv` — Trip arcs (scheduled trips)
- `idle_links.csv` — Idle arcs (waiting at terminals)
- `deadheading_links_df.csv` — Terminal-to-terminal arcs
- `deadheading_links-TC_df.csv` — Terminal-to-charging arcs
- `deadheading_links-CT_df.csv` — Charging-to-terminal arcs
- `deadheading_links-CC_df.csv` — Charging arcs
- `FinalLinks.csv` — Combined set of all arcs

---

## ✅ Dependencies

- Python ≥ 3.7
- pandas

Install with:

```bash
pip install pandas
```

---

## 📌 Notes

- Time intervals are dynamically created from the trip schedule.
- SOC is discretized into 5% intervals (e.g., 100-95, 95-90, ..., 5-0).
- Energy and cost parameters are user-defined and can be adjusted in the script.

---

## 📚 References

- Kliewer, N., et al. (2006). A Time–Space Network Based Exact Algorithm for Multi-Depot Bus Scheduling.
- Zhang, R., et al. (2021). Value-of-Charging-Time Analysis for Electric Bus Scheduling.

---

## 👨‍💻 Author

Behnam Emami  

---

## 📄 License

This project is released under the MIT License.
