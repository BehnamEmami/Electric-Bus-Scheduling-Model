"""

@author: Behnam
"""

import pandas as pd
import math
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import gurobipy as gp
from collections import defaultdict
from gurobipy import Model, GRB, quicksum


network_data = pd.read_csv("FinalLinks.csv")


links = network_data
links_list = list(links.index)

# Create a directed graph
G = nx.DiGraph()

# Add edges with attributes
for _, row in links.iterrows():
    G.add_edge(row["Start Node"], row["End Node"], 
               trip_id=row["Trip_ID"], 
               duration=row["Duration"], 
               cost=row["Cost"], 
               energy_consumption=row["Energy Consumption"], 
               type=row["Type"], 
               id=row["ID"])

# Display the graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")




# Create the optimization model
model = Model("LinkFlowOptimization")

# Add flow variables for each link in the graph
flow = model.addVars(
    G.edges,  # Links in the graph
    vtype=GRB.INTEGER,  # Flow can be continuous; change to GRB.INTEGER or GRB.BINARY if needed
    name="flow"
)

intermediate_nodes = [
    node for node in G.nodes 
    if "Depot" not in node and "Sink" not in node
]

for node in intermediate_nodes:
    model.addConstr(
        quicksum(flow[i, node] for i in G.predecessors(node)) ==  # Incoming flow
        quicksum(flow[node, j] for j in G.successors(node)),     # Outgoing flow
        name=f"FlowConservation_{node}"
    )


trip_edges = defaultdict(list)  # Dictionary to store edges by Trip_ID
for i, j, attr in G.edges(data=True):
    edge_id = str(attr.get("id", ""))  # Convert to string to ensure it is iterable
    if "Trip_" in edge_id:
        trip_edges[edge_id].append((i, j))



for trip_id, edges in trip_edges.items():
    model.addConstr(
        quicksum(flow[i, j] for i, j in edges) == 1,
        name=f"TripConstraint_{trip_id}"
    )
    
    
# Group depot and sink nodes by energy level
depot_nodes_by_energy = defaultdict(list)
sink_nodes_by_energy = defaultdict(list)

for node in G.nodes:
    if "Depot" in node:
        energy_level = node.split("+")[-1]  # Extract energy level (e.g., "45-40")
        depot_nodes_by_energy[energy_level].append(node)
    elif "Sink" in node:
        energy_level = node.split("+")[-1]  # Extract energy level (e.g., "45-40")
        sink_nodes_by_energy[energy_level].append(node)



for energy_level in depot_nodes_by_energy:
    # Get the depots and sinks for the current energy level
    depots = depot_nodes_by_energy[energy_level]
    sinks = sink_nodes_by_energy.get(energy_level, [])  # Use .get to handle missing energy levels

    # Add the constraint if both depots and sinks exist for this energy level
    if depots and sinks:
        model.addConstr(
            quicksum(flow[depot, j] for depot in depots for j in G.successors(depot)) ==
            quicksum(flow[i, sink] for sink in sinks for i in G.predecessors(sink)),
            name=f"DepotSinkFlow_{energy_level}"
        )


# Define capacity for each charging station
charging_station_capacities = {
    "1": 8,  # Capacity for Charging Station 1
    "2": 5,  # Capacity for Charging Station 2
    "3": 8,  # Capacity for Charging Station 3
    "4": 3,
    "5": 3,
    "6": 3,
    "7": 3,
    "8": 8,
    "9":5,
    "10": 3,
    "11": 3,
    "12": 3,
    "13": 3,
    # Add capacities for all other stations as needed
}

# Group charging station nodes by station number and time interval
charging_stations = defaultdict(lambda: defaultdict(list))

# Categorize nodes based on station number and time interval
for node in G.nodes:
    if "Charging station" in node:
        parts = node.split("+")
        station_number = parts[0].split()[-1]  # Extract station number (e.g., "1" from "Charging station 1")
        time_interval = parts[1]              # Extract time interval (e.g., "1")
        charging_stations[station_number][time_interval].append(node)



for station_number, time_intervals in charging_stations.items():
    for time_interval, nodes in time_intervals.items():
        # Aggregate incoming flow for this specific charging station at all energy levels
        incoming_flow = quicksum(
            flow[i, charging_station]
            for charging_station in nodes
            for i in G.predecessors(charging_station)
            if G[i][charging_station].get("type") in ["DeadheadingTC", "Charging", "Pull-In"]
        )

        # Aggregate outgoing flow for this specific charging station at all energy levels
        outgoing_flow = quicksum(
            flow[charging_station, j]
            for charging_station in nodes
            for j in G.successors(charging_station)
            if G[charging_station][j].get("type") == "DeadheadingCT"
        )

        # Retrieve the capacity for this charging station
        station_capacity = charging_station_capacities.get(station_number, float('inf'))  # Default to unlimited if not defined

        # Add the constraint for this station and time interval
        constraint = model.addConstr(
            incoming_flow - outgoing_flow <= station_capacity,
            name=f"ChargingStation_{station_number}_Time_{time_interval}_Capacity"
        )


# Objective: Minimize total cost
model.setObjective(
    quicksum(flow[i, j] * G[i][j]["cost"] for i, j in G.edges),
    GRB.MINIMIZE
)


# Time and termination
model.setParam("TimeLimit", 80000)  #  time limit
model.setParam("MIPGap", 0.01)     # 1% optimality gap

# Parallel computing
#model.setParam("Threads", 12)       # Use 8 threads
#model.setParam("ConcurrentMIP", 6) # Solve 4 MIP models concurrently

# MIP focus
#model.setParam("MIPFocus", 2)      # Focus on finding feasible solutions

# Heuristics
model.setParam("Heuristics", 0.5)  # Spend up to 50% of time on heuristics

# Cuts
#model.setParam("NodeMethod", 2)

#model.setParam("Cuts", 2)        # Aggressive cut generation

# Presolve
#model.setParam("Presolve", 2)      # Aggressive presolve

# Perform more feasibility pump passes
#model.setParam("PumpPasses", 20)

# Apply RINS more frequently
#model.setParam("RINS", 10)


model.optimize()


model.write("model.lp")
model.write("solution.sol")

        
        
        
with open("solutionNew1.txt", "w") as file:
    for i, j in flow:
        if flow[i, j].x > 0:  # Only save non-zero flows
            # Write the output to the file
            file.write(f"Flow on edge ({i} -> {j}): {flow[i, j].x}\n")

print("Solution saved to gurobi_solution.txt")