"""
Spatial-Temporal-SOC Network Generator
--------------------------------------

This script generates a time-expanded network with energy (SOC) dimensions
for Battery Electric Bus (BEB) scheduling. It includes:

- Time discretization (operational & non-operational)
- Node generation (Depot, Terminal, Charging Station, Sink)
- Arc/link generation for:
  * Pull-in/Pull-out
  * Trip arcs
  * Idle arcs
  * Deadheading (Terminal <-> Terminal/Charging)
  * Charging arcs (with tariff)

Input files required:
- Bus_Trip_Timetable.csv
- Terminals.csv
- Charging stations.csv
- Charging_Costs_by_Time_Interval.csv

Output files generated:
- all_nodes.csv
- pull-In-Links.csv
- pull-Out-Links.csv
- Triplinknew.csv
- idle_links.csv
- deadheading_links_df.csv
- deadheading_links-TC_df.csv
- deadheading_links-CT_df.csv
- deadheading_links-CC_df.csv
- FinalLinks.csv

Author: Behnam Emami
"""


from datetime import datetime, timedelta
import pandas as pd

# Load the bus trip timetable
file_path = 'Bus_Trip_Timetable.csv'  # Update the file path if necessary
bus_timetable = pd.read_csv(file_path)

# Convert Start_Time and End_Time to datetime for easier manipulation
bus_timetable['Start_Time'] = pd.to_datetime(bus_timetable['Start_Time'], format='%H:%M')
bus_timetable['End_Time'] = pd.to_datetime(bus_timetable['End_Time'], format='%H:%M')

# Find the earliest start time and latest end time
earliest_start_time = bus_timetable['Start_Time'].min()
latest_end_time = bus_timetable['End_Time'].max()

# Round earliest start time down to the closest previous hour
rounded_start_time = earliest_start_time.replace(minute=0, second=0, microsecond=0)

# Round latest end time up to the next hour
if latest_end_time.minute > 0 or latest_end_time.second > 0:
    rounded_end_time = (latest_end_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
else:
    rounded_end_time = latest_end_time

# Print the results
print("Earliest Start Time:", earliest_start_time.strftime('%H:%M'))
print("Rounded Start Time:", rounded_start_time.strftime('%H:%M'))
print("Latest End Time:", latest_end_time.strftime('%H:%M'))
print("Rounded End Time:", rounded_end_time.strftime('%H:%M'))



# Inputs
operational_interval_duration = 10  # Duration in minutes for operational period
non_operational_interval_duration = 30  # Duration in minutes for non-operational period
energy_levels = ["100-95", "95-90", "90-85", "85-80", "80-75", "75-70", 
                 "70-65", "65-60", "60-55", "55-50", "50-45", "45-40", 
                 "40-35", "35-30", "30-25", "25-20", "20-15", "15-10", 
                 "10-5", "5-0"]

# Load terminals and charging stations data
terminals_distances_path = 'Terminals.csv'  # Update path as necessary
charging_stations_path = 'Charging stations.csv'  # Update path as necessary
terminals_distances = pd.read_csv(terminals_distances_path)
charging_stations = pd.read_csv(charging_stations_path)

# Extract terminals and charging stations
terminals = terminals_distances['Terminal'].unique()
valid_charging_stations = charging_stations['ChargingStation'].unique()

# Calculate the operational and non-operational periods
operational_start_time = rounded_start_time
operational_end_time = rounded_end_time
non_operational_start_time = rounded_end_time
non_operational_end_time = rounded_start_time + timedelta(days=1)  # Wrap around midnight

# Function to generate time intervals
def generate_intervals(start_time, end_time, interval_duration, start_index, base_time, is_operational):
    intervals = [
        (index, int((current_time - base_time).total_seconds() // 60),
         int((current_time + timedelta(minutes=interval_duration) - base_time).total_seconds() // 60),
         is_operational)
        for index, current_time in enumerate(
            (start_time + timedelta(minutes=interval_duration * i) for i in range(int((end_time - start_time).total_seconds() // 60 // interval_duration))),
            start=start_index
        )
    ]
    return intervals

# Define base time as the rounded start time
base_time = operational_start_time

# Generate operational intervals
operational_intervals = generate_intervals(
    operational_start_time, operational_end_time, operational_interval_duration, start_index=1, base_time=base_time, is_operational=True
)

# Generate non-operational intervals (post-midnight period)
non_operational_intervals = generate_intervals(
    non_operational_start_time, non_operational_end_time, non_operational_interval_duration,
    start_index=len(operational_intervals) + 1, base_time=base_time, is_operational=False
)

# Combine both intervals
all_intervals = operational_intervals + non_operational_intervals

# Convert to DataFrame for better readability
intervals_df = pd.DataFrame(all_intervals, columns=['Index', 'Start_Minutes', 'End_Minutes', 'Is_Operational'])


print(intervals_df)



# Generate nodes
# Depot and Sink nodes with energy levels
depot_nodes = [f"Depot+{level}" for level in energy_levels]
sink_nodes = [f"Sink+{level}" for level in energy_levels]

# Terminal and Charging Station nodes with time and energy levels
terminal_nodes = [f"{terminal}+{interval[0]}+{level}" 
                  for terminal in terminals 
                  for interval in all_intervals 
                  for level in energy_levels]
charging_station_nodes = [f"{station}+{interval[0]}+{level}" 
                          for station in valid_charging_stations 
                          for interval in all_intervals 
                          for level in energy_levels]

# Combine all nodes
all_nodes = depot_nodes + terminal_nodes + charging_station_nodes + sink_nodes

# Create a DataFrame for the nodes
nodes_df = pd.DataFrame({'Node_Index': range(1, len(all_nodes) + 1), 'Node_Name': all_nodes})

nodes_df.to_csv("all_nodes.csv", index = False)
# Display the nodes
print(nodes_df)



# Load terminals and charging stations distances
terminals_distances_file = 'Terminals.csv'  # Update file path if necessary
charging_stations_distances_file = 'Charging stations.csv'  # Update file path if necessary

terminals_distances = pd.read_csv(terminals_distances_file)
charging_stations_distances = pd.read_csv(charging_stations_distances_file)

# Initialize the simplified distance dictionary
distance_dict = {}

# Map distances between terminals
for _, row in terminals_distances.iterrows():
    terminal_from = row['Terminal']
    for col in terminals_distances.columns[1:]:  # Skip the 'Terminal' column
        terminal_to = col
        distance_dict[(terminal_from, terminal_to)] = round(row[terminal_to])

# Map distances between charging stations and terminals
for _, row in charging_stations_distances.iterrows():
    charging_station_from = row['ChargingStation']
    for col in charging_stations_distances.columns[1:]:  # Skip the 'ChargingStation' column
        terminal_to = col
        distance_dict[(charging_station_from, terminal_to)] = round(row[terminal_to])
        distance_dict[(terminal_to, charging_station_from)] = round(row[terminal_to]) # Assuming symmetric distances

# Print a preview of the dictionary
print("Preview of the distance dictionary:")
for key, value in list(distance_dict.items())[:400]:
    print(f"{key}: {value}")




# Generate Pull-In links efficiently
intervals_map = intervals_df.set_index('Index')['Is_Operational'].to_dict()
pull_in_links = [
    {
        "Start Node": depot,
        "End Node": terminal,
        "Duration": 0,
        "Cost": 300,
        "Energy Consumption": 0,
        "Type": "Pull-In",
        "ID": "NA"
    }
    for depot in depot_nodes
    for terminal in terminal_nodes + charging_station_nodes
    if (depot.split('+')[1] == terminal.split('+')[2] and
        intervals_map.get(int(terminal.split('+')[1]), False))
]

# Convert Pull-In links to a DataFrame
pull_in_links_df = pd.DataFrame(pull_in_links)

print(pull_in_links_df)
pull_in_links_df.to_csv("pull-In-Links.csv", index=False)




# Generate Pull-Out links
pull_out_links = []

# Iterate through all terminal nodes and sink nodes
for terminal in terminal_nodes:
    terminal_parts = terminal.split('+')
    terminal_energy = terminal_parts[2]
    time_interval_index = int(terminal_parts[1])
    
    # Check if the interval is operational or non-operational
    is_operational = intervals_map.get(time_interval_index, False)
    
    for sink in sink_nodes:
        sink_energy = sink.split('+')[1]
        
        # Create link only if energy levels match
        if terminal_energy == sink_energy:
            cost = (120 - time_interval_index) if is_operational else 0
            pull_out_links.append({
                "Start Node": terminal,
                "End Node": sink,
                "Duration": 0,
                "Cost": cost,
                "Energy Consumption": 0,
                "Type": "Pull-Out",
                "ID": "NA"
            })

# Convert Pull-Out links to a DataFrame
pull_out_links_df = pd.DataFrame(pull_out_links)

print(pull_out_links_df)
pull_out_links_df.to_csv("pull-Out-Links.csv", index=False)




# Generate Trip Links

def generate_trip_links_with_unique_ids(timetable, expanded_nodes, time_intervals, energy_levels, energy_consumption_rate):
    trip_links = []

    # Map energy levels to their average values
    energy_level_map = {
        level: (int(level.split("-")[0]) + int(level.split("-")[1])) / 2 for level in energy_levels
    }
    energy_level_bounds = {
        level: (int(level.split("-")[0]), int(level.split("-")[1])) for level in energy_levels
    }

    # Parse expanded nodes to include terminal and interval index
    expanded_nodes = expanded_nodes.copy()
    expanded_nodes['Terminal'] = expanded_nodes['Node_Name'].str.split('+').str[0]
    expanded_nodes['Time Interval Index'] = pd.to_numeric(expanded_nodes['Node_Name'].str.split('+').str[1], errors='coerce')
    expanded_nodes['Energy Level'] = expanded_nodes['Node_Name'].str.split('+').str[2]

    # Iterate through the timetable
    for _, trip in timetable.iterrows():
        trip_id = trip["Trip_ID"]
        start_terminal = trip["Start_Terminal"]
        end_terminal = trip["End_Terminal"]
        start_time = pd.to_datetime(trip["Start_Time"], format="%H:%M")
        end_time = pd.to_datetime(trip["End_Time"], format="%H:%M")

        # Find the start and end time intervals using the interval generation concept
        start_interval_index = next(
            (interval[0] for interval in all_intervals if interval[1] <= int((start_time - base_time).total_seconds() // 60) < interval[2]),
            None
        )
        end_interval_index = next(
            (interval[0] for interval in all_intervals if interval[1] <= int((end_time - base_time).total_seconds() // 60) < interval[2]),
            None
        )

        if start_interval_index is None or end_interval_index is None:
            continue  # Skip if time intervals are not found

        # Get potential start nodes
        start_nodes = expanded_nodes[
            (expanded_nodes['Terminal'] == start_terminal) &
            (expanded_nodes['Time Interval Index'] == start_interval_index)
        ]

        # Iterate through all start nodes
        for _, start_node in start_nodes.iterrows():
            start_energy_level = start_node['Energy Level']
            start_energy_avg = energy_level_map[start_energy_level]

            # Calculate trip duration and energy consumption
            trip_duration = (end_time - start_time).seconds / 60  # Duration in minutes
            energy_consumption = trip_duration * energy_consumption_rate

            # Compute end energy level
            end_energy_avg = start_energy_avg - energy_consumption

            # Find the corresponding energy level interval
            end_energy_level = None
            for level, (upper, lower) in energy_level_bounds.items():
                if upper >= end_energy_avg > lower:  # Check if end_energy_avg is within the bounds
                    end_energy_level = level
                    break

            if not end_energy_level or end_energy_avg <= 10:
                continue  # Skip if energy level is invalid or below 10

            # Get potential end nodes
            end_nodes = expanded_nodes[
                (expanded_nodes['Terminal'] == end_terminal) &
                (expanded_nodes['Time Interval Index'] == end_interval_index) &
                (expanded_nodes['Energy Level'] == end_energy_level)
            ]

            # Create trip links
            for _, end_node in end_nodes.iterrows():
                unique_id = f"Trip_{trip_id}_{start_interval_index}_{end_interval_index}"
                trip_links.append({
                    "Start Node": start_node['Node_Name'],
                    "End Node": end_node['Node_Name'],
                    "Trip_ID": trip_id,
                    "Duration": trip_duration,
                    "Cost": 0,  # Cost proportional to duration
                    "Energy Consumption": energy_consumption,
                    "Type": "Trip",
                    "ID": unique_id
                })

    # Convert trip links to a DataFrame
    return pd.DataFrame(trip_links)



energy_consumption_rate = 0.2  # Example kWh per minute
trip_links = generate_trip_links_with_unique_ids(
    timetable=bus_timetable,
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
    energy_levels=energy_levels,
    energy_consumption_rate=energy_consumption_rate
)
trip_links.to_csv("Triplinknew.csv",index=False)



def generate_idle_links(expanded_nodes, time_intervals):
    idle_links = []

    # Iterate through all terminal nodes
    for _, node in expanded_nodes.iterrows():
        if "Terminal" not in node["Node_Name"]:
            continue  # Skip non-terminal nodes

        # Parse current node details
        node_name = node["Node_Name"]
        terminal = node_name.split('+')[0]
        time_interval = int(node_name.split('+')[1])
        energy_level = node_name.split('+')[2]

        # Find the corresponding next time interval
        next_interval = time_interval + 1

        # Find the start and end intervals
        start_interval = time_intervals[time_intervals["Index"] == time_interval]
        end_interval = time_intervals[time_intervals["Index"] == next_interval]

        if start_interval.empty or end_interval.empty:
            continue  # Skip if intervals are not found

        # Check if operational or non-operational
        is_operational = start_interval["Is_Operational"].iloc[0]
        duration = 10 if is_operational else 30
        cost = duration * 1.2 if is_operational else 0

        # Find the corresponding end node
        end_node = expanded_nodes[
            (expanded_nodes["Node_Name"] == f"{terminal}+{next_interval}+{energy_level}")
        ]

        if end_node.empty:
            continue  # Skip if end node is not found

        # Create idle links
        idle_links.append({
            "Start Node": node_name,
            "End Node": end_node["Node_Name"].iloc[0],
            "Duration": duration,
            "Cost": cost,
            "Energy Consumption": 0,  # Idle links consume no energy
            "Type": "Idle",
            "ID": "NA"
        })

    # Convert idle links to DataFrame
    return pd.DataFrame(idle_links)


idle_links = generate_idle_links(
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
)


idle_links.to_csv("idle_links.csv", index = False)


def generate_deadheading_links(expanded_nodes, time_intervals, distance_dict, energy_levels, energy_consumption_rate, base_time):
    """
    Generate Deadheading Links between terminal nodes only using vectorized operations.

    Parameters:
    - expanded_nodes: pd.DataFrame containing all nodes.
    - time_intervals: pd.DataFrame containing time interval information.
    - distance_dict: Dictionary containing distances between terminals.
    - energy_levels: List of energy levels.
    - energy_consumption_rate: Energy consumption rate per minute.
    - base_time: Base time (e.g., operational start time).

    Returns:
    - pd.DataFrame containing Deadheading Links.
    """
    # Map energy levels to bounds
    energy_level_bounds = {
        level: (int(level.split("-")[0]), int(level.split("-")[1])) for level in energy_levels
    }

    # Precompute valid terminal connections from distance_dict
    terminal_pairs = pd.DataFrame(
        [(start, end, duration) for (start, end), duration in distance_dict.items() if start != end],
        columns=["Start Terminal", "End Terminal", "Duration"]
    )

    # Filter terminal nodes and extract components
    terminal_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Terminal")].copy()
    terminal_nodes["Terminal"] = terminal_nodes["Node_Name"].str.split('+').str[0]
    terminal_nodes["Time Interval"] = terminal_nodes["Node_Name"].str.split('+').str[1].astype(int)
    terminal_nodes["Energy Level"] = terminal_nodes["Node_Name"].str.split('+').str[2]

    # Merge terminal nodes with terminal pairs to create all possible links
    links = terminal_nodes.merge(
        terminal_pairs, left_on="Terminal", right_on="Start Terminal", how="inner"
    )

    # Add time interval for end nodes
    links["End Time Interval"] = links["Time Interval"] + (links["Duration"] // 10).astype(int)

    # Map time intervals to operational status
    time_intervals_dict = {
        row["Index"]: (row["Start_Minutes"], row["End_Minutes"], row["Is_Operational"])
        for _, row in time_intervals.iterrows()
    }
    time_intervals_df = pd.DataFrame.from_dict(time_intervals_dict, orient="index", columns=["Start", "End", "Is_Operational"])
    time_intervals_df.reset_index(inplace=True)

    # Merge time interval info to get operational status of start and end intervals
    links = links.merge(time_intervals_df, left_on="Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "Start Operational"}, inplace=True)

    links = links.merge(time_intervals_df, left_on="End Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "End Operational"}, inplace=True)

    # Keep only operational intervals
    links = links[(links["Start Operational"]) & (links["End Operational"])]

    # Compute energy consumption and remaining energy
    links["Start Energy Upper"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][0])
    links["Start Energy Lower"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][1])
    links["Energy Consumption"] = links["Duration"] * energy_consumption_rate
    links["End Energy"] = (links["Start Energy Upper"] + links["Start Energy Lower"]) / 2 - links["Energy Consumption"]

    # Determine the end energy level
    def determine_end_energy_level(end_energy):
        for level, (upper, lower) in energy_level_bounds.items():
            if upper >= end_energy > lower and level != "5-0":
                return level
        return None

    links["End Energy Level"] = links["End Energy"].apply(determine_end_energy_level)

    # Filter out links with invalid end energy levels
    links = links[links["End Energy Level"].notnull()]

    # Explicitly filter out non-terminal end nodes
    links["End Node"] = links["End Terminal"] + "+" + links["End Time Interval"].astype(str) + "+" + links["End Energy Level"]
    valid_terminal_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Terminal")]["Node_Name"]
    links = links[links["End Node"].isin(valid_terminal_nodes)]

    # Add cost
    links["Cost"] = links["Duration"] * 1.6

    # Create final output
    deadheading_links = links[[
        "Node_Name", "End Node", "Duration", "Cost", "Energy Consumption", "Energy Level", "End Energy Level"
    ]].copy()
    deadheading_links.rename(columns={
        "Node_Name": "Start Node",
        "Energy Level": "Start Energy Level"
    }, inplace=True)
    deadheading_links["Type"] = "DeadheadingTT"
    deadheading_links["ID"] = "NA"

    return deadheading_links

deadheading_links_df = generate_deadheading_links(
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
    distance_dict=distance_dict,  # Terminal distances
    energy_levels=energy_levels,
    energy_consumption_rate=0.255,  # Example: kWh per minute
    base_time=rounded_start_time
)

deadheading_links_df.to_csv("deadheading_links_df.csv", index = False)


def generate_deadheading_links_terminal_to_charging(
    expanded_nodes,
    time_intervals,
    distance_dict,
    energy_levels,
    energy_consumption_rate,
    base_time
):
    """
    Generate Deadheading Links between terminal nodes and charging station nodes.

    Parameters:
    - expanded_nodes: pd.DataFrame containing all nodes.
    - time_intervals: pd.DataFrame containing time interval information.
    - distance_dict: Dictionary containing distances between terminals and charging stations.
    - energy_levels: List of energy levels.
    - energy_consumption_rate: Energy consumption rate per minute.
    - base_time: Base time (e.g., operational start time).

    Returns:
    - pd.DataFrame containing Deadheading Links (Terminal to Charging Stations).
    """
    # Map energy levels to bounds
    energy_level_bounds = {
        level: (int(level.split("-")[0]), int(level.split("-")[1])) for level in energy_levels
    }

    # Precompute valid terminal-to-charging connections from distance_dict
    terminal_charging_pairs = pd.DataFrame(
        [
            (start, end, duration)
            for (start, end), duration in distance_dict.items()
            if start.startswith("Terminal") and end.startswith("Charging station")
        ],
        columns=["Start Terminal", "End Charging Station", "Duration"]
    )

    # Filter terminal nodes and extract components
    terminal_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Terminal")].copy()
    terminal_nodes["Terminal"] = terminal_nodes["Node_Name"].str.split('+').str[0]
    terminal_nodes["Time Interval"] = terminal_nodes["Node_Name"].str.split('+').str[1].astype(int)
    terminal_nodes["Energy Level"] = terminal_nodes["Node_Name"].str.split('+').str[2]

    # Merge terminal nodes with terminal-to-charging pairs to create all possible links
    links = terminal_nodes.merge(
        terminal_charging_pairs, left_on="Terminal", right_on="Start Terminal", how="inner"
    )

    # Add time interval for end nodes
    links["End Time Interval"] = links["Time Interval"] + (links["Duration"] // 10).astype(int)

    # Map time intervals to operational status
    time_intervals_dict = {
        row["Index"]: (row["Start_Minutes"], row["End_Minutes"], row["Is_Operational"])
        for _, row in time_intervals.iterrows()
    }
    time_intervals_df = pd.DataFrame.from_dict(time_intervals_dict, orient="index", columns=["Start", "End", "Is_Operational"])
    time_intervals_df.reset_index(inplace=True)

    # Merge time interval info to get operational status of start and end intervals
    links = links.merge(time_intervals_df, left_on="Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "Start Operational"}, inplace=True)

    links = links.merge(time_intervals_df, left_on="End Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "End Operational"}, inplace=True)

    # Compute energy consumption and remaining energy
    links["Start Energy Upper"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][0])
    links["Start Energy Lower"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][1])
    links["Energy Consumption"] = links["Duration"] * energy_consumption_rate
    links["End Energy"] = (links["Start Energy Upper"] + links["Start Energy Lower"]) / 2 - links["Energy Consumption"]

    # Determine the end energy level
    def determine_end_energy_level(end_energy):
        for level, (upper, lower) in energy_level_bounds.items():
            if upper >= end_energy > lower and level != "5-0":
                return level
        return None

    links["End Energy Level"] = links["End Energy"].apply(determine_end_energy_level)

    # Filter out links with invalid end energy levels
    links = links[links["End Energy Level"].notnull()]

    # Construct end node names
    links["End Node"] = links["End Charging Station"] + "+" + links["End Time Interval"].astype(str) + "+" + links["End Energy Level"]

    # Explicitly filter for charging station end nodes
    valid_charging_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Charging station")]["Node_Name"]
    links = links[links["End Node"].isin(valid_charging_nodes)]

    # Compute costs based on operational status
    links["Cost"] = 10 + links["Duration"] * links["End Operational"].apply(lambda x: 1.6 if x else 1.0)

    # Create final output
    terminal_to_charging_links = links[[
        "Node_Name", "End Node", "Duration", "Cost", "Energy Consumption", "Energy Level", "End Energy Level"
    ]].copy()
    terminal_to_charging_links.rename(columns={
        "Node_Name": "Start Node",
        "Energy Level": "Start Energy Level"
    }, inplace=True)
    terminal_to_charging_links["Type"] = "DeadheadingTC"
    terminal_to_charging_links["ID"] = "NA"

    return terminal_to_charging_links

# Generate Deadheading Links from Terminals to Charging Stations
terminal_to_charging_links_df = generate_deadheading_links_terminal_to_charging(
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
    distance_dict=distance_dict,  # Terminal-to-Charging distances
    energy_levels=energy_levels,
    energy_consumption_rate=0.25,  # Example: kWh per minute
    base_time=rounded_start_time  # Base time for intervals
)



terminal_to_charging_links_df.to_csv("deadheading_links-TC_df.csv", index = False)



def read_tariff_data(file_path):
    """
    Reads tariff data from a CSV file.

    Parameters:
    - file_path: str
        Path to the tariff CSV file.

    Returns:
    - dict: A dictionary mapping time interval indices to tariffs.
    """
    # Read the CSV file into a DataFrame
    tariff_df = pd.read_csv(file_path)

    # Convert the DataFrame into a dictionary
    tariff_dict = dict(zip(tariff_df["Time Interval Index"], tariff_df["Tariff"]))
    
    return tariff_dict


tariff_file_path = "Charging_Costs_by_Time_Interval.csv"  # Replace with the actual file path
tariffs = read_tariff_data(tariff_file_path)


def generate_charging_links_with_tariffs(
    expanded_nodes,
    time_intervals,
    energy_levels,
    tariffs,
    base_time
):
    """
    Generates charging links for charging stations, ensuring only the highest energy level is chosen.

    Parameters:
    - expanded_nodes: pd.DataFrame
        DataFrame containing expanded node information.
    - time_intervals: pd.DataFrame
        DataFrame containing time intervals with columns: Index, Start_Minutes, End_Minutes, Is_Operational.
    - energy_levels: list of str
        List of energy level intervals (e.g., ["100-95", ..., "5-0"]).
    - tariffs: dict
        Dictionary mapping time interval indices to electricity tariffs.
    - base_time: datetime object
        The base operational start time.

    Returns:
    - pd.DataFrame: Charging links as a DataFrame.
    """
    from datetime import timedelta

    charging_links = []

    # Filter charging station nodes
    charging_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Charging station")].copy()
    charging_nodes["Charging Station"] = charging_nodes["Node_Name"].str.split('+').str[0]
    charging_nodes["Time Interval Index"] = charging_nodes["Node_Name"].str.split('+').str[1].astype(int)
    charging_nodes["Energy Level"] = charging_nodes["Node_Name"].str.split('+').str[2]

    for _, start_node in charging_nodes.iterrows():
        start_terminal = start_node["Charging Station"]
        start_energy_level = start_node["Energy Level"]
        start_time_interval = start_node["Time Interval Index"]

        # Skip if already at the highest energy level
        if start_energy_level == energy_levels[0]:  # "100-95"
            continue

        # Check if the current interval is operational
        is_operational = time_intervals.loc[
            time_intervals["Index"] == start_time_interval, "Is_Operational"
        ].values[0]

        # Determine the number of energy levels to connect based on time interval type
        max_energy_levels = 1 if is_operational else 3

        # Determine the duration of the interval
        duration = 10 if is_operational else 30  # 10 minutes for operational, 30 minutes for non-operational

        # Determine the next time interval
        next_time_interval = start_time_interval + 1
        if next_time_interval > time_intervals["Index"].max():
            continue  # Skip if no next interval exists

        # Find the maximum reachable energy level
        start_energy_index = energy_levels.index(start_energy_level)
        max_reachable_index = max(0, start_energy_index - max_energy_levels)
        end_energy_level = energy_levels[max_reachable_index]

        # Calculate cost
        tariff = tariffs.get(start_time_interval, 0.0)
        cost = duration * tariff + (duration * 1.6 if is_operational else duration * 0.3)

        # Add the link
        charging_links.append({
            "Start Node": start_node["Node_Name"],
            "End Node": f"{start_terminal}+{next_time_interval}+{end_energy_level}",
            "Duration": duration,
            "Cost": cost,
            "Type": "Charging",
            "ID": f"{start_terminal}_{start_time_interval}_to_{next_time_interval}_{end_energy_level}"
        })

    # Convert to DataFrame
    return pd.DataFrame(charging_links)


# Example Usage
tariffs = read_tariff_data("Charging_Costs_by_Time_Interval.csv")  # Load tariffs as a dictionary
charging_station_links_df = generate_charging_links_with_tariffs(
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
    energy_levels=energy_levels,
    tariffs=tariffs,
    base_time=rounded_start_time  # Provide the operational start time
)

charging_station_links_df.to_csv("deadheading_links-CC_df.csv", index = False)


def generate_deadheading_links_charging_to_terminal(
    expanded_nodes,
    time_intervals,
    distance_dict,
    energy_levels,
    energy_consumption_rate,
    base_time
):
    """
    Generate Deadheading Links between charging station nodes and terminal nodes.

    Parameters:
    - expanded_nodes: pd.DataFrame containing all nodes.
    - time_intervals: pd.DataFrame containing time interval information.
    - distance_dict: Dictionary containing distances between charging stations and terminals.
    - energy_levels: List of energy levels.
    - energy_consumption_rate: Energy consumption rate per minute.
    - base_time: Base time (e.g., operational start time).

    Returns:
    - pd.DataFrame containing Deadheading Links (Charging Stations to Terminals).
    """
    # Map energy levels to bounds
    energy_level_bounds = {
        level: (int(level.split("-")[0]), int(level.split("-")[1])) for level in energy_levels
    }

    # Precompute valid charging-to-terminal connections from distance_dict
    charging_terminal_pairs = pd.DataFrame(
        [
            (start, end, duration)
            for (start, end), duration in distance_dict.items()
            if start.startswith("Charging station") and end.startswith("Terminal")
        ],
        columns=["Start Charging Station", "End Terminal", "Duration"]
    )

    # Filter charging station nodes and extract components
    charging_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Charging station")].copy()
    charging_nodes["Charging Station"] = charging_nodes["Node_Name"].str.split('+').str[0]
    charging_nodes["Time Interval"] = charging_nodes["Node_Name"].str.split('+').str[1].astype(int)
    charging_nodes["Energy Level"] = charging_nodes["Node_Name"].str.split('+').str[2]

    # Merge charging station nodes with charging-to-terminal pairs to create all possible links
    links = charging_nodes.merge(
        charging_terminal_pairs, left_on="Charging Station", right_on="Start Charging Station", how="inner"
    )

    # Add time interval for end nodes
    links["End Time Interval"] = links["Time Interval"] + (links["Duration"] // 10).astype(int)

    # Map time intervals to operational status
    time_intervals_dict = {
        row["Index"]: (row["Start_Minutes"], row["End_Minutes"], row["Is_Operational"])
        for _, row in time_intervals.iterrows()
    }
    time_intervals_df = pd.DataFrame.from_dict(time_intervals_dict, orient="index", columns=["Start", "End", "Is_Operational"])
    time_intervals_df.reset_index(inplace=True)

    # Merge time interval info to get operational status of start and end intervals
    links = links.merge(time_intervals_df, left_on="Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "Start Operational"}, inplace=True)

    links = links.merge(time_intervals_df, left_on="End Time Interval", right_on="index", how="inner")
    links.rename(columns={"Is_Operational": "End Operational"}, inplace=True)

    # Compute energy consumption and remaining energy
    links["Start Energy Upper"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][0])
    links["Start Energy Lower"] = links["Energy Level"].map(lambda x: energy_level_bounds[x][1])
    links["Energy Consumption"] = links["Duration"] * energy_consumption_rate
    links["End Energy"] = (links["Start Energy Upper"] + links["Start Energy Lower"]) / 2 - links["Energy Consumption"]

    # Determine the end energy level
    def determine_end_energy_level(end_energy):
        for level, (upper, lower) in energy_level_bounds.items():
            if upper >= end_energy > lower and level != "5-0":
                return level
        return None

    links["End Energy Level"] = links["End Energy"].apply(determine_end_energy_level)

    # Filter out links with invalid end energy levels
    links = links[links["End Energy Level"].notnull()]

    # Construct end node names
    links["End Node"] = links["End Terminal"] + "+" + links["End Time Interval"].astype(str) + "+" + links["End Energy Level"]

    # Explicitly filter for terminal end nodes
    valid_terminal_nodes = expanded_nodes[expanded_nodes["Node_Name"].str.startswith("Terminal")]["Node_Name"]
    links = links[links["End Node"].isin(valid_terminal_nodes)]

    # Compute costs based on operational status
    links["Cost"] = links["Duration"] * links["End Operational"].apply(lambda x: 1.6 if x else 1.0)

    # Create final output
    charging_to_terminal_links = links[[
        "Node_Name", "End Node", "Duration", "Cost", "Energy Consumption", "Energy Level", "End Energy Level"
    ]].copy()
    charging_to_terminal_links.rename(columns={
        "Node_Name": "Start Node",
        "Energy Level": "Start Energy Level"
    }, inplace=True)
    charging_to_terminal_links["Type"] = "DeadheadingCT"
    charging_to_terminal_links["ID"] = "NA"

    return charging_to_terminal_links


# Example Usage
charging_to_terminal_links_df = generate_deadheading_links_charging_to_terminal(
    expanded_nodes=nodes_df,
    time_intervals=intervals_df,
    distance_dict=distance_dict,  # Update with the actual dictionary
    energy_levels=energy_levels,
    energy_consumption_rate=0.25,  # Example: kWh per minute
    base_time=rounded_start_time
)


charging_to_terminal_links_df.to_csv("deadheading_links-CT_df.csv", index = False)



def combine_all_links(
    charging_to_terminal_links_df,
    charging_station_links_df,
    terminal_to_charging_links_df,
    deadheading_links_df,
    pull_in_links_df,
    idle_links_df,
    pull_out_links_df,
    trip_links_df
):
    """
    Combines all the link DataFrames into one comprehensive DataFrame.

    Parameters:
    - All the individual link DataFrames.

    Returns:
    - pd.DataFrame: Combined DataFrame containing all links.
    """
    # Combine all links
    all_links_df = pd.concat(
        [
            charging_to_terminal_links_df,
            charging_station_links_df,
            terminal_to_charging_links_df,
            deadheading_links_df,
            pull_in_links_df,
            idle_links_df,
            pull_out_links_df,
            trip_links_df,
        ],
        axis=0,
        ignore_index=True
    )

    # Add a unique ID for each link
    all_links_df["Link_ID"] = all_links_df.index + 1

    return all_links_df


# Example Usage
all_links_df = combine_all_links(
    charging_to_terminal_links_df=charging_to_terminal_links_df,
    charging_station_links_df=charging_station_links_df,
    terminal_to_charging_links_df=terminal_to_charging_links_df,
    deadheading_links_df=deadheading_links_df,
    pull_in_links_df=pull_in_links_df,
    idle_links_df=idle_links,
    pull_out_links_df=pull_out_links_df,
    trip_links_df=trip_links
)

# Save to CSV or display the combined DataFrame
all_links_df.to_csv("FinalLinks.csv", index=False)

