"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

from simulator.aircraft.aircraft_state import AircraftState
from simulator.math.angles import diff_angle_pi


@dataclass
class AutopilotStatus:
    """
    Class to represent and manage the status of the autopilot system.

    Attributes
    ----------
    roll : float
        Current roll angle of the aircraft.
    target_roll : float
        Desired roll angle for the aircraft.
    course : float
        Current course angle of the aircraft.
    target_course : float
        Desired course angle for the aircraft.
    beta : float
        Current sideslip angle of the aircraft.
    target_beta : float
        Desired sideslip angle for the aircraft.
    pitch : float
        Current pitch angle of the aircraft.
    target_pitch : float
        Desired pitch angle for the aircraft.
    altitude : float
        Current altitude of the aircraft.
    target_altitude : float
        Desired altitude for the aircraft.
    airspeed : float
        Current airspeed of the aircraft.
    target_airspeed : float
        Desired airspeed for the aircraft.
    """

    roll: float = 0.0
    target_roll: float = 0.0

    course: float = 0.0
    target_course: float = 0.0

    beta: float = 0.0
    target_beta: float = 0.0

    pitch: float = 0.0
    target_pitch: float = 0.0

    altitude: float = 0.0
    target_altitude: float = 0.0

    airspeed: float = 0.0
    target_airspeed: float = 0.0

    def update_aircraft_state(self, state: AircraftState) -> None:
        """
        Update the autopilot status based on the current aircraft state.

        Parameters
        ----------
        state : AircraftState
            The current state of the aircraft.
        """
        self.roll = state.roll
        self.course = state.course_angle
        self.beta = state.beta
        self.pitch = state.pitch
        self.altitude = state.altitude
        self.airspeed = state.airspeed

    def update_targets(self, **kwargs) -> None:
        """
        Update the target values for the autopilot system.

        Parameters
        ----------
        kwargs : dict
            Key-value pairs where keys are target names and values are the new targets.
            Possible keys include 'target_roll', 'target_course', 'target_beta', 'target_pitch',
            'target_altitude', and 'target_airspeed'.
        """
        valid_keys = {
            "target_roll",
            "target_course",
            "target_beta",
            "target_pitch",
            "target_altitude",
            "target_airspeed",
        }
        for key, value in kwargs.items():
            if key in valid_keys and value is not None:
                setattr(self, key, value)
            elif key not in valid_keys:
                raise ValueError(f"Invalid target: {key}")

    @property
    def roll_error(self) -> float:
        """The roll angle error in radians"""
        return diff_angle_pi(self.target_roll, self.roll)

    @property
    def course_error(self) -> float:
        """The course angle error in radians"""
        return diff_angle_pi(self.target_course, self.course)

    @property
    def beta_error(self) -> float:
        """The sideslip angle error in radians"""
        return diff_angle_pi(self.target_beta, self.beta)

    @property
    def pitch_error(self) -> float:
        """The pitch angle error in radians"""
        return diff_angle_pi(self.target_pitch, self.pitch)

    @property
    def altitude_error(self) -> float:
        """The altitude error in meters"""
        return self.target_altitude - self.altitude

    @property
    def airspeed_error(self) -> float:
        """The airspeed error in m/s"""
        return self.target_airspeed - self.airspeed

    def __str__(self) -> str:
        """
        Returns a string representation of the autopilot status.

        Returns
        -------
        str
            A formatted string displaying the current and target values for each
            autopilot parameter.
        """
        return (
            f"Autopilot Status:\n"
            f"- Roll: {self.roll:.2f} rad, Target Roll: {self.target_roll:.2f} rad, Error: {self.roll_error:.2f} rad\n"
            f"- Course: {self.course:.2f} rad, Target Course: {self.target_course:.2f} rad, Error: {self.course_error:.2f} rad\n"
            f"- Sideslip Angle: {self.beta:.2f} rad, Target Sideslip Angle: {self.target_beta:.2f} rad, Error: {self.beta_error:.2f} rad\n"
            f"- Pitch: {self.pitch:.2f} rad, Target Pitch: {self.target_pitch:.2f} rad, Error: {self.pitch_error:.2f} rad\n"
            f"- Altitude: {self.altitude:.2f} m, Target Altitude: {self.target_altitude:.2f} m, Error: {self.altitude_error:.2f} m\n"
            f"- Airspeed: {self.airspeed:.2f} m/s, Target Airspeed: {self.target_airspeed:.2f} m/s, Error: {self.airspeed_error:.2f} m/s"
        )
