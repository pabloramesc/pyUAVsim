"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

class Aircraft:
    def __init__(self) -> None:
        self.state = np.zeros(12) # 12 var: pn pe pd u v w roll pitch yaw p q r


    def get_ned_position(self) -> np.ndarray:
        return self.state[0:3]
    
    def get_body_velocity(self) -> np.ndarray:
        return self.state[3:6]
    
    def get_attitude_angles(self) -> np.ndarray:
        return self.state[6:9]
    
    def get_angular_rates(self) -> np.ndarray:
        return self.state[9:12]