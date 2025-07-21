# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:25:03 2025

@author: Behnam
"""

import pandas as pd
import json
import math
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import gurobipy as gp
from collections import defaultdict
from gurobipy import Model, GRB, quicksum
import re



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
    vtype=GRB.INTEGER,  #
    name="flow"
)
############################################################################
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

######################################################################
trip_edges = defaultdict(list)  # Dictionary to store edges by Trip_ID
for i, j, attr in G.edges(data=True):
    edge_id = str(attr.get("id", "")) 
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
        energy_level = node.split("+")[-1]  
        sink_nodes_by_energy[energy_level].append(node)



for energy_level in depot_nodes_by_energy:
    # Get the depots and sinks for the current energy level
    depots = depot_nodes_by_energy[energy_level]
    sinks = sink_nodes_by_energy.get(energy_level, [])  

    # Add the constraint if both depots and sinks exist for this energy level
    if depots and sinks:
        model.addConstr(
            quicksum(flow[depot, j] for depot in depots for j in G.successors(depot)) ==
            quicksum(flow[i, sink] for sink in sinks for i in G.predecessors(sink)),
            name=f"DepotSinkFlow_{energy_level}"
        )


# Define capacity for each charging station
charging_station_capacities = {
    "1": 8,  
    "2": 5,
    "3": 8,  
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
}

# Group charging station nodes by station number and time interval
charging_stations = defaultdict(lambda: defaultdict(list))

# Categorize nodes based on station number and time interval
for node in G.nodes:
    if "Charging station" in node:
        parts = node.split("+")
        station_number = parts[0].split()[-1]  
        time_interval = parts[1]              
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



##################################################################
# Objective: Minimize total cost
model.setObjective(
    quicksum(flow[i, j] * G[i][j]["cost"] for i, j in G.edges),
    GRB.MINIMIZE
)

model.setParam("TimeLimit", 80000)  # 1 hour time limit
model.setParam("MIPGap", 0.035)     # 1% optimality gap

model.optimize()

####################################################################


model.write("model.lp")
model.write("solution.sol")

#################################################################
#plotting
for i, j in flow:
    if flow[i, j].x > 0:  # Only print non-zero flows
        print(f"Flow on edge ({i} -> {j}): {flow[i, j].x}")
        
        


def parse_solution_file(file_path, output_csv_path):
    # Define a list to store parsed rows
    data = []

    # Regex to parse the link information
    pattern = re.compile(
        r"Flow on edge \((.+?) -> (.+?)\): (\d+\.?\d*)"
    )

    # Read the file and parse lines
    with open(file_path, "r") as file:
        for line in file:
            match = pattern.match(line)
            if match:
                start_node = match.group(1)
                end_node = match.group(2)
                flow = float(match.group(3))

                # Extract time interval and energy level from start and end nodes
                start_parts = start_node.split("+")
                end_parts = end_node.split("+")

                # Parse start node
                if "Depot" in start_node or "Sink" in start_node:
                    start_node_name = start_parts[0]
                    start_time_interval = ""
                    start_energy_level = start_parts[1] if len(start_parts) > 1 else ""
                else:
                    start_node_name = start_parts[0]
                    start_time_interval = start_parts[1] if len(start_parts) > 1 else ""
                    start_energy_level = start_parts[2] if len(start_parts) > 2 else ""

                # Parse end node
                if "Depot" in end_node or "Sink" in end_node:
                    end_node_name = end_parts[0]
                    end_time_interval = ""
                    end_energy_level = end_parts[1] if len(end_parts) > 1 else ""
                else:
                    end_node_name = end_parts[0]
                    end_time_interval = end_parts[1] if len(end_parts) > 1 else ""
                    end_energy_level = end_parts[2] if len(end_parts) > 2 else ""

                # Append parsed data
                data.append([
                    start_node_name,
                    start_time_interval,
                    start_energy_level,
                    end_node_name,
                    end_time_interval,
                    end_energy_level,
                    flow,
                ])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=[
        "Start Node", "Start Time Interval", "Start Energy Level",
        "End Node", "End Time Interval", "End Energy Level", "Flow"
    ])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)


file_path = "solutionNew1.txt"  
output_csv_path = "Results - Links Flows1.csv"  

parse_solution_file(file_path, output_csv_path)
print(f"CSV file created at: {output_csv_path}")



file_path = 'Results - Links Flows1.csv'
data = pd.read_csv(file_path)

# Extract unique nodes from the 'Start Node' and 'End Node' columns
unique_nodes = set(data['Start Node']).union(set(data['End Node']))

# Convert the set to a sorted list
unique_nodes = sorted(unique_nodes)

# Print the list of unique nodes
print("List of unique nodes:")
print(unique_nodes)



nodes = pd.concat([data['Start Node'], data['End Node']]).unique()
charging_stations = [node for node in nodes if "Charging station" in node]

# Define the full range of time intervals
time_intervals = pd.Series(range(1, 129))  

# Initialize an empty dictionary to store results
charging_station_flows = {}

# Calculate the flow for each charging station
for station in charging_stations:
    # Filter data where the charging station is either the Start Node or End Node
    station_data = data[(data['Start Node'] == station) & (data['End Node'] == station)]
    
    # Group by 'Start Time Interval' and calculate the total flow
    flow_summary = (
        station_data.groupby('Start Time Interval')['Flow']
        .sum()
        .reset_index()
        .rename(columns={'Flow': 'Total Flow'})
    )
    
    flow_summary = (
        pd.DataFrame({'Start Time Interval': time_intervals})
        .merge(flow_summary, on='Start Time Interval', how='left')
        .fillna({'Total Flow': 0})
    )
    
    # Store the result in the dictionary
    charging_station_flows[station] = flow_summary

    # Plot the total flow vs time interval
    plt.figure(figsize=(10, 6))
    plt.plot(flow_summary['Start Time Interval'], flow_summary['Total Flow'], marker='o', linestyle='-', label=station)
    plt.xlabel('Time Interval')
    plt.ylabel('Total Flow (Number of Buses)')
    plt.title(f'Total Flow vs Time Interval for {station}')
    plt.legend()
    plt.grid(True)
    
    # Set axis limits
    plt.xlim(0, 128)  
    plt.ylim(0, 8)    
    plt.yticks(range(0, 9, 1))  
    
    # Save the plot as a PNG file
    plt.savefig(f"{station.replace(' ', '_')}_flow_plot1.png")
    plt.show()


for station, flow_summary in charging_station_flows.items():
    file_name = f"{station.replace(' ', '_')}_flow_summary1.csv"
    flow_summary.to_csv(file_name, index=False)
    print(f"Saved: {file_name}")
    
    
    
#########################################################################

nodes = pd.concat([data['Start Node'], data['End Node']]).unique()
charging_stations = [node for node in nodes if "Charging station" in node]

# Define the full range of time intervals
time_intervals = pd.Series(range(1, 129))  

# Initialize a DataFrame to store total flow for all stations
total_flow_per_interval = pd.DataFrame({'Start Time Interval': time_intervals, 'Total Flow': 0})

# Calculate the total flow across all charging stations
for station in charging_stations:
    # Filter data where the charging station is either the Start Node or End Node
    station_data = data[(data['Start Node'] == station) | (data['End Node'] == station)]
    
    # Group by 'Start Time Interval' and calculate the total flow
    flow_summary = (
        station_data.groupby('Start Time Interval')['Flow']
        .sum()
        .reset_index()
        .rename(columns={'Flow': 'Total Flow'})
    )
    
    # Merge with the full range of time intervals and fill missing values with 0
    flow_summary = (
        pd.DataFrame({'Start Time Interval': time_intervals})
        .merge(flow_summary, on='Start Time Interval', how='left')
        .fillna({'Total Flow': 0})
    )
    
    # Add the station's flow to the total flow per interval
    total_flow_per_interval['Total Flow'] += flow_summary['Total Flow']

# Calculate the charging station ratio for each time interval
total_flow_per_interval['Charging Station Ratio'] = total_flow_per_interval['Total Flow'] / 66

# Adjust x-axis positions
adjusted_x = []
for t in total_flow_per_interval['Start Time Interval']:
    if t <= 120:
        adjusted_x.append(t)
    else:
        adjusted_x.append(120 + 3 * (t - 120))


def generate_time_labels():
    time_labels = {}

    for i in range(0, 121, 6):
        time_labels[i] = f"{4 + i // 6} AM" if (4 + i // 6) <= 11 else f"{(4 + i // 6) - 12} PM"
    # After 120, label every 2 intervals with hourly time
    for i in range(122, 129, 2):
        time_labels[120 + 3 * (i - 120)] = f"{1 + (i - 122) // 2} AM"
    return time_labels

time_labels = generate_time_labels()

# Plot the charging station ratio
plt.figure(figsize=(12, 6))
plt.plot(adjusted_x, total_flow_per_interval['Charging Station Ratio'], marker='o', linestyle='-', color='red', label='Charging Station Ratio')
plt.xlabel('Time Interval')
plt.ylabel('Charging Station Ratio')
plt.title('Charging Station Ratio vs Time Interval')
plt.legend()
plt.grid(True)

# Set custom x-axis labels
plt.xticks(
    ticks=list(time_labels.keys()), 
    labels=list(time_labels.values()), 
    rotation=45
)

plt.xlim(min(adjusted_x), max(adjusted_x))  
plt.ylim(0, 1.1)  
plt.tight_layout()
plt.savefig('charging_station_ratio_scaled_x_axis_plot1.png')  
plt.show()


print("\nCharging Station Ratio:")
print(total_flow_per_interval)

