"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

from simulator.aircraft import AircraftState, ControlDeltas
from simulator.autopilot import AutopilotStatus
from simulator.autopilot.mission_control import MissionControl, RouteManager


@dataclass
class SimulationData:
    dt_sim: float
    t_sim: float
    k_sim: int
    uav_state: AircraftState
    control_deltas: ControlDeltas
    autopilot_status: AutopilotStatus
    mission_status: MissionControl
    route_manager: RouteManager
