"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.autopilot.waypoints import WaypointsList

waypoints_file = "config/waypoints_example.wp"
waypoints_list = WaypointsList()
waypoints_list.load_from_txt(waypoints_file)

pass
