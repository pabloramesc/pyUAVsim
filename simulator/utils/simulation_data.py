"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""
import time
from dataclasses import dataclass
from multiprocessing import Queue
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
    mission_control: MissionControl


class SimulationDataConnector:
    def __init__(self, q: Queue) -> None:
        self.q = q
        self.last_data: SimulationData = None
        self.data_buffer: list[SimulationData] = []
        self.initialize()

    def initialize(self) -> None:
        # wait for data
        while self.q.empty():
            time.sleep(0.1)
        # get all pending data
        while not self.q.empty():
            sim_data: SimulationData = self.q.get()
            self.data_buffer.append(sim_data)
        self.last_data = self.data_buffer[-1]

    def update(self) -> None:
        while not self.q.empty():
            sim_data: SimulationData = self.q.get()
            self.data_buffer.append(sim_data)
            self.last_data = sim_data

    def read_buffer(self) -> list[SimulationData]:
        buffer_copy = self.data_buffer.copy()
        self.clear_buffer()
        return buffer_copy

    def clear_buffer(self) -> None:
        self.data_buffer = []