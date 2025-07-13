Network Generator
The Network_Generator.py script is used to generate a Spatial-Temporal-State of Charge (SOC) network for electric bus scheduling. This network incorporates time intervals, energy levels, and connectivity between trips, terminals, and charging stations.

📂 Input Files
Bus_Trip_Timetable.csv
Contains the scheduled service data, listing all trips that need to be covered by electric buses.
Fields include trip ID, start time, end time, origin terminal, destination terminal, energy required, and layover time.

Terminals.csv
A matrix representing the travel time (in minutes) between terminals.

Charging_Station.csv
A matrix showing travel times (in minutes) between terminals and nearby charging stations.

Charging_Costs_by_Time_Interval.csv
Electricity tariff data varying by time of day, used for cost-efficient charging decisions.

⚙️ What the Code Does
Time Discretization:
Divides the service day into time intervals:

10-minute intervals during operational hours

30-minute intervals during nighttime hours

Energy Discretization:
Divides the state-of-charge into levels of 5% increments.

Node Duplication:
Creates multiple instances of each node (trip, terminal, depot, sink, charging station) based on the time and SOC levels.

Link Generation:
Builds different types of arcs:

Trip arcs

Layover arcs

Charging arcs

Depot and sink arcs

📦 Required Packages
from datetime import datetime, timedelta
import pandas as pd
