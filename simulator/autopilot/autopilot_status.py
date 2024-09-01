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
    roll_target : float
        Desired roll angle for the aircraft.
    course : float
        Current course angle of the aircraft.
    course_target : float
        Desired course angle for the aircraft.
    beta : float
        Current sideslip angle of the aircraft.
    beta_target : float
        Desired sideslip angle for the aircraft.
    pitch : float
        Current pitch angle of the aircraft.
    pitch_target : float
        Desired pitch angle for the aircraft.
    altitude : float
        Current altitude of the aircraft.
    altitude_target : float
        Desired altitude for the aircraft.
    airspeed : float
        Current airspeed of the aircraft.
    airspeed_target : float
        Desired airspeed for the aircraft.
    """

    roll: float = 0.0
    roll_target: float = 0.0
    course: float = 0.0
    course_target: float = 0.0
    beta: float = 0.0
    beta_target: float = 0.0
    pitch: float = 0.0
    pitch_target: float = 0.0
    altitude: float = 0.0
    altitude_target: float = 0.0
    airspeed: float = 0.0
    airspeed_target: float = 0.0

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

    @property
    def roll_error(self) -> float:
        """The roll angle error in radians"""
        return diff_angle_pi(self.roll_target, self.roll)

    @property
    def course_error(self) -> float:
        """The course angle error in radians"""
        return diff_angle_pi(self.course_target, self.course)

    @property
    def beta_error(self) -> float:
        """The sideslip angle error in radians"""
        return diff_angle_pi(self.beta_target, self.beta)

    @property
    def pitch_error(self) -> float:
        """The pitch angle error in radians"""
        return diff_angle_pi(self.pitch_target, self.pitch)

    @property
    def altitude_error(self) -> float:
        """The altitude error in meters"""
        return self.altitude_target - self.altitude

    @property
    def airspeed_error(self) -> float:
        """The airspeed error in m/s"""
        return self.airspeed_target - self.airspeed

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
            f"- Roll: {self.roll:.2f} rad, Target Roll: {self.roll_target:.2f} rad, Error: {self.roll_error:.2f} rad\n"
            f"- Course: {self.course:.2f} rad, Target Course: {self.course_target:.2f} rad, Error: {self.course_error:.2f} rad\n"
            f"- Sideslip Angle: {self.beta:.2f} rad, Target Sideslip Angle: {self.beta_target:.2f} rad, Error: {self.beta_error:.2f} rad\n"
            f"- Pitch: {self.pitch:.2f} rad, Target Pitch: {self.pitch_target:.2f} rad, Error: {self.pitch_error:.2f} rad\n"
            f"- Altitude: {self.altitude:.2f} m, Target Altitude: {self.altitude_target:.2f} m, Error: {self.altitude_error:.2f} m\n"
            f"- Airspeed: {self.airspeed:.2f} m/s, Target Airspeed: {self.airspeed_target:.2f} m/s, Error: {self.airspeed_error:.2f} m/s"
        )
