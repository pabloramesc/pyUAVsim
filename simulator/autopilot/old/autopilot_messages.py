"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np
from simulator.actuators.servo_motor import ServoMotor


@dataclass
class DataHandler:
    pass


class DataHandlerHistory:
    handler_type: type = DataHandler

    def __init__(self) -> None:
        self.history: dict = {}
        self.history_array: dict = {}
        for field_key in self.__class__.handler_type.__dataclass_fields__.keys():
            self.history[field_key]: list = []
            self.history_array[field_key]: np.ndarray = []

    def update(self, handler: DataHandler) -> dict:
        for key, list_k in self.history.items():
            list_k.append(getattr(handler, key))
        return self.history

    def to_array(self) -> dict:
        for key, list_k in self.history.items():
            self.history_array[key] = np.array(list_k)
        return self.history_array


@dataclass
class ActuatorsDeltas(DataHandler):
    elev: float = 0.0
    ailR: float = 0.0
    ailL: float = 0.0
    rudd: float = 0.0
    throttle: float = 0.0

    def to_dict(self) -> dict:
        return dict(
            elev=self.elev,
            ailR=self.ailR,
            ailL=self.ailL,
            rudd=self.rudd,
            throttle=self.throttle,
        )


class ActuatorsDeltasHistory(DataHandlerHistory):
    handler_type: type = ActuatorsDeltas


@dataclass
class AutopilotDeltas(DataHandler):
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def to_actuators_deltas(self) -> ActuatorsDeltas:
        """
        Use this method to map autopilot deltas to actuators inputs

        ### Autopilot deltas (input) ###
        - elevator
        - aileron
        - throttle
        - rudder

        ### Actuators deltas (output) ###
        - elev <-- elevator
        - ailR <-- -aileron
        - ailL <-- +aileron
        - rudd <-- rudder
        - throttle <-- throttle

        """
        return ActuatorsDeltas(
            elev=self.elevator,
            ailR=-self.aileron,
            ailL=+self.aileron,
            rudd=self.rudder,
            throttle=self.throttle,
        )


class AutopilotDeltasHistory(DataHandlerHistory):
    handler_type: type = AutopilotDeltas


@dataclass
class AutopilotCommands(DataHandler):
    pitch: float = 0.0
    altitude: float = 0.0
    airspeed: float = 0.0
    roll: float = 0.0
    course: float = 0.0


class AutopilotCommandsHistory(DataHandlerHistory):
    handler_type: type = AutopilotCommands


@dataclass
class TransmitterData(DataHandler):
    right_vertical: float = 0.0
    right_horizontal: float = 0.0
    left_vertical: float = 0.0
    left_horizontal: float = 0.0  # from 0.0 to 1.0
    switch_3poles: float = 0.0
    switch_2poles: float = 0.0

    def from_channels(self, channels: list) -> None:
        # for transmitter channels disposition
        self.right_vertical = channels[0]
        self.right_horizontal = channels[1]
        self.left_vertical = channels[2]
        self.left_horizontal = channels[3]

    def to_commands(self) -> dict:
        # channel to control mapping
        return dict(
            elevator=self.right_vertical,
            aileron=self.right_horizontal,
            throttle=self.left_vertical,
            rudder=self.left_horizontal,
        )
