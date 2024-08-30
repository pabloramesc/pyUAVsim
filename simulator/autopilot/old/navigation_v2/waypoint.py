"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from simulator.common.constants import DEFAULT_HOME_COORDS

from simulator.utils.geodesy import geo2ned

class WaypointCommand(Enum):
    WAYPOINT = 0 # no params
    ORBIT_TURNS = 1 # param1=direction(+-1) param2=max-turns(float)
    ORBIT_TIME = 2 # param1=direction(+-1) param2=max-time(float)
    JUMP = 3 # param1=wp-id(int) param2=max-loops(int)
    LAND_P1 = 4
    LAND_P2 = 5
    LAND_P3 = 6
    LAND_P4 = 7

@dataclass
class Waypoint:
    id: int
    latitude: float
    longitude: float
    altitude: float
    command: WaypointCommand
    radius: float = 0.0
    param1: float = None
    param2: float = None
    param3: float = None
    
    def get_coords(self) -> np.ndarray:
        return np.array([self.latitude, self.longitude, self.altitude])
    
    def get_coords_NED(self, home_coords: np.ndarray = DEFAULT_HOME_COORDS) -> np.ndarray:
        waypoint_coords = self.get_coords()
        return geo2ned(waypoint_coords, home_coords)
    
