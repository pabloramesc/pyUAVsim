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

AUTOPILOT_MODES = ["manual", "fbw", "cruise", "auto"]


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
        self.mode: str = None

        uav = AircraftDynamics(dt, params, use_quat=True)
        x_trim, delta_trim = uav.trim(Va=state.airspeed)
        self.control_deltas = ControlDeltas(delta_trim, max_angle=np.deg2rad(45.0))
        self.config.calculate_control_gains(params, uav.state, self.control_deltas)

        self.flight_control = FlightControl(self.config)

    def update(self, dt: float, state: AircraftState) -> ControlDeltas:
        _dt = dt or self.dt
        _state = state or self.state
        self.status.update_aircraft_state(_state)

    def set_mode(self, mode: str) -> None:
        if mode in AUTOPILOT_MODES:
            self.mode = mode
        else:
            raise ValueError("invalid autopilot mode!")

    def run_manual_mode(self, joystick: ControlDeltas) -> ControlDeltas:
        return ControlDeltas(joystick.delta, 0.25 * np.pi)

    def run_fbw_mode(
        self, dt: float, state: AircraftState, joystick: ControlDeltas
    ) -> ControlDeltas:
        pass

    def run_cruise_mode(
        self, dt: float, state: AircraftState, joystick: ControlDeltas
    ) -> ControlDeltas:
        pass

    def run_auto_mode(self, dt: float, state: AircraftState) -> ControlDeltas:
        pass

    def control_roll_pitch_airspeed(
        self,
        roll_target: float = None,
        pitch_target: float = None,
        airspeed_target: float = None,
    ) -> ControlDeltas:
        """
        Compute control inputs to achieve the desired roll, pitch, and airspeed.

        Parameters
        ----------
        roll_target : float, optional
            Desired roll angle in radians. If not provided, the current roll target from `status` is used.

        pitch_target : float, optional
            Desired pitch angle in radians. If not provided, the current pitch target from `status` is used.

        airspeed_target : float, optional
            Desired airspeed in m/s. If not provided, the current airspeed target from `status` is used.

        Returns
        -------
        ControlDeltas
            Control inputs for the aircraft, including:
            - delta_a: Aileron deflection for roll control.
            - delta_e: Elevator deflection for pitch control.
            - delta_r: Rudder deflection for yaw damping.
            - delta_t: Throttle setting for airspeed control.

        Notes
        -----
        Updates `roll_target`, `pitch_target`, and `airspeed_target` from `status` based on the provided values.
        Computes control deltas for roll, pitch, yaw, and airspeed using these updated targets.
        """
        # Update targets if new values are provided
        self.status.update_targets(
            roll_target=roll_target,
            pitch_target=pitch_target,
            airspeed_target=airspeed_target,
        )

        # Compute control inputs on roll, pitch, and airspeed targets
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

    def control_course_altitude_airspeed(
        self,
        course_target: float = None,
        altitude_target: float = None,
        airspeed_target: float = None,
    ) -> ControlDeltas:
        """
        Compute control inputs to achieve the desired course, altitude, and airspeed.

        Parameters
        ----------
        course_target : float, optional
            Desired course angle in radians. If not provided, the current course target from `status` is used.

        altitude_target : float, optional
            Desired altitude in meters. If not provided, the current altitude target from `status` is used.

        airspeed_target : float, optional
            Desired airspeed in m/s. If not provided, the current airspeed target from `status` is used.

        Returns
        -------
        ControlDeltas
            Control inputs for the aircraft, including:
            - delta_a: Aileron deflection to achieve course hold.
            - delta_e: Elevator deflection to achieve altitude hold.
            - delta_r: Rudder deflection for yaw damping.
            - delta_t: Throttle setting for airspeed control.

        Notes
        -----
        Updates `course_target`, `altitude_target`, and `airspeed_target` from `status` based on the provided values.
        Computes intermediate roll and pitch targets based on course and altitude hold requirements.
        Delegates to `control_roll_pitch_airspeed` to compute the final control inputs.
        """
        # Update targets if new values are provided
        self.status.update_targets(
            course_target=course_target,
            altitude_target=altitude_target,
            airspeed_target=airspeed_target,
        )

        # Calculate roll and pitch targets based on course and altitude
        roll_target = self.flight_control.course_hold_with_roll(
            self.status.course_target, self.aircraft_state.course_angle, self.dt
        )
        pitch_target = self.flight_control.altitude_hold_with_pitch(
            self.status.altitude_target, self.aircraft_state.altitude, self.dt
        )

        # Compute control inputs based on new roll, pitch, and airspeed targets
        self.control_roll_pitch_airspeed(roll_target, pitch_target, airspeed_target)

        return self.control_deltas
