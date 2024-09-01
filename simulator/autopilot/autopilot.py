"""
Copyright (c) 2024 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from typing import Any

import numpy as np

from simulator.aircraft.aircraft_dynamics import AircraftDynamics
from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.aircraft.control_deltas import ControlDeltas
from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.autopilot_status import AutopilotStatus
from simulator.autopilot.flight_control import FlightControl


class Autopilot:
    """
    Class to implement an autopilot system for managing aircraft control.

    Attributes
    ----------
    dt : float
        The time step for control updates.
    aircraft_params : AirframeParameters
        Parameters of the aircraft's airframe.
    aircraft_state : AircraftState
        Reference to the current state of the aircraft. This is a live reference to
        an `AircraftState` instance that is automatically updated by the physical
        simulation. Changes to the state are reflected in real-time and affect the
        autopilot's control decisions.
    config : AutopilotConfig
        Configuration settings for the autopilot.
    status : AutopilotStatus
        Current status of the autopilot.
    flight_control : FlightControl
        Controller for managing aircraft flight.
    control_deltas : ControlDeltas
        Control inputs to the aircraft.
    """

    def __init__(
        self, dt: float, params: AirframeParameters, state: AircraftState
    ) -> None:
        """
        Initialize the autopilot with the given parameters.

        Parameters
        ----------
        dt : float
            The time step of the simulation.
        params : AirframeParameters
            Parameters of the aircraft's airframe.
        state : AircraftState
            Reference to the aircraft's state object. This object is expected to be
            automatically updated by the physical simulation. The autopilot uses this
            reference to access the current state of the aircraft in real-time.
        """
        self.dt = dt
        self.aircraft_params = params
        self.aircraft_state = state

        self.config = AutopilotConfig()
        self.status = AutopilotStatus(airspeed_target=state.airspeed)

        uav = AircraftDynamics(dt, params, use_quat=True)
        x_trim, delta_trim = uav.trim(Va=state.airspeed)
        self.control_deltas = ControlDeltas(delta_trim, max_angle=np.deg2rad(45.0))
        self.config.calculate_control_gains(params, uav.state, self.control_deltas)

        self.flight_control = FlightControl(self.config)

    def control_pitch_roll(self, **kwargs: Any) -> ControlDeltas:
        """
        Compute the control inputs to achieve the desired roll, pitch, and airspeed.

        Parameters
        ----------
        **kwargs : Any
            Optional keyword arguments for setting `roll_target`, `pitch_target`,
            and `airspeed_target`. If not provided, the current targets are used.

        Returns
        -------
        ControlDeltas
            The computed control inputs for the aircraft.
        """
        self.status.update_aircraft_state(self.aircraft_state)

        self.status.roll_target = kwargs.get("roll_target", self.status.roll_target)
        self.status.pitch_target = kwargs.get("pitch_target", self.status.pitch_target)
        self.status.airspeed_target = kwargs.get(
            "airspeed_target", self.status.airspeed_target
        )

        self.control_deltas.delta_a = self.flight_control.roll_hold_with_aileron(
            self.status.roll_target, self.aircraft_state.roll, self.aircraft_state.p
        )
        self.control_deltas.delta_e = self.flight_control.pitch_hold_with_elevator(
            self.status.pitch_target, self.aircraft_state.pitch, self.aircraft_state.q
        )
        self.control_deltas.delta_r = self.flight_control.yaw_damper_with_rudder(
            self.aircraft_state.r, self.dt
        )
        self.control_deltas.delta_t = self.flight_control.airspeed_hold_with_throttle(
            self.status.airspeed_target, self.aircraft_state.airspeed, self.dt
        )

        return self.control_deltas

    def control_course_altitude(self, **kwargs: Any) -> ControlDeltas:
        """
        Compute the control inputs to achieve the desired course, altitude, and airspeed.

        Parameters
        ----------
        **kwargs : Any
            Optional keyword arguments for setting `course_target`, `altitude_target`,
            and `airspeed_target`. If not provided, the current targets are used.

        Returns
        -------
        ControlDeltas
            The computed control inputs for the aircraft.
        """
        self.status.update_aircraft_state(self.aircraft_state)

        self.status.course_target = kwargs.get(
            "course_target", self.status.course_target
        )
        self.status.altitude_target = kwargs.get(
            "altitude_target", self.status.altitude_target
        )
        self.status.airspeed_target = kwargs.get(
            "airspeed_target", self.status.airspeed_target
        )

        self.status.roll_target = self.flight_control.course_hold_with_roll(
            self.status.course_target, self.aircraft_state.course_angle, self.dt
        )
        self.control_deltas.delta_a = self.flight_control.roll_hold_with_aileron(
            self.status.roll_target, self.aircraft_state.roll, self.aircraft_state.p
        )
        self.status.pitch_target = self.flight_control.altitude_hold_with_pitch(
            self.status.altitude_target, self.aircraft_state.altitude, self.dt
        )
        self.control_deltas.delta_e = self.flight_control.pitch_hold_with_elevator(
            self.status.pitch_target, self.aircraft_state.pitch, self.aircraft_state.q
        )
        self.control_deltas.delta_r = self.flight_control.yaw_damper_with_rudder(
            self.aircraft_state.r, self.dt
        )
        self.control_deltas.delta_t = self.flight_control.airspeed_hold_with_throttle(
            self.status.airspeed_target, self.aircraft_state.airspeed, self.dt
        )
        return self.control_deltas
