"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.pid_controller import PIDController
from simulator.common.constants import EARTH_GRAVITY_CONSTANT as g


class FlightControl:

    def __init__(self, config: AutopilotConfig) -> None:
        """Initialize the FlightControl with the given autopilot configuration.

        Parameters
        ----------
        config : AutopilotConfig
            Configuration object containing the autopilot settings and control parameters.
        """
        self.config = config

        self.pid_course_roll = PIDController(
            kp=self.config.kp_course_roll,
            ki=self.config.ki_course_roll,
            max_output=self.config.max_roll,
            min_output=self.config.min_roll,
        )

        self.pid_sideslip_rudder = PIDController(
            kp=self.config.kp_sideslip_rudder,
            ki=self.config.ki_sideslip_rudder,
            max_output=self.config.max_rudder,
            min_output=self.config.min_rudder,
        )

        self.xi_damper = 0.0

        self.pid_turn_coord = PIDController(
            kp=self.config.kp_turn_coord,
            ki=self.config.ki_turn_coord,
            kd=self.config.kd_turn_coord,
            max_output=self.config.max_rudder,
            min_output=self.config.min_rudder,
        )

        self.pid_altitude_pitch = PIDController(
            kp=self.config.kp_altitude_pitch,
            ki=self.config.ki_altitude_pitch,
            max_output=self.config.max_pitch,
            min_output=self.config.min_pitch,
        )

        self.pid_airspeed_throttle = PIDController(
            kp=self.config.kp_airspeed_throttle,
            ki=self.config.ki_airspeed_throttle,
        )

        self.pid_airspeed_pitch = PIDController(
            kp=self.config.kp_airspeed_pitch,
            ki=self.config.ki_airspeed_pitch,
            max_output=self.config.max_pitch,
            min_output=self.config.min_pitch,
        )

    def roll_hold_with_aileron(self, roll_ref: float, roll: float, p: float) -> float:
        """Compute the aileron deflection needed to hold the desired roll angle.

        Parameters
        ----------
        roll_ref : float
            Desired roll angle (rad).
        roll : float
            Current roll angle (rad).
        p : float
            Current x-axis roll rate (rad/s).

        Returns
        -------
        float
            The computed aileron deflection (rad), clipped to the allowed range.
        """
        delta_a = (
            self.config.kp_roll_aileron * (roll_ref - roll)
            - self.config.kd_roll_aileron * p
        )
        return np.clip(delta_a, self.config.min_aileron, self.config.max_aileron)

    def pitch_hold_with_elevator(
        self, pitch_ref: float, pitch: float, q: float
    ) -> float:
        """Compute the elevator deflection needed to hold the desired pitch angle.

        Parameters
        ----------
        pitch_ref : float
            Desired pitch angle (rad).
        pitch : float
            Current pitch angle (rad).
        q : float
            Current y-axis pitch rate (rad/s).

        Returns
        -------
        float
            The computed elevator deflection (rad), clipped to the allowed range.
        """
        delta_e = (
            self.config.kp_pitch_elevator * (pitch_ref - pitch)
            - self.config.kd_pitch_elevator * q
        )
        return np.clip(delta_e, self.config.min_elevator, self.config.max_elevator)

    @staticmethod
    def wrap_course(course_ref: float, course: float):
        """Wrap the reference course angle to the range [-pi, pi].

        Parameters
        ----------
        course_ref : float
            Desired course angle (rad).
        course : float
            Current course angle (rad).

        Returns
        -------
        float
            The wrapped reference course angle (rad).
        """
        while course_ref - course > +np.pi:
            course_ref -= 2 * np.pi
        while course_ref - course < -np.pi:
            course_ref += 2 * np.pi
        return course_ref

    def course_hold_with_roll(
        self, course_ref: float, course: float, dt: float
    ) -> float:
        """Compute the roll angle needed to maintain the desired course.

        Parameters
        ----------
        course_ref : float
            Desired course angle (rad).
        course : float
            Current course angle (rad).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed roll angle (rad).
        """
        course_ref = self.wrap_course(course_ref, course)
        roll_ref = self.pid_course_roll.update(course_ref, course, dt)
        return roll_ref

    def sideslip_hold_with_rudder(self, beta: float, dt: float) -> float:
        """Compute the rudder deflection needed to correct the sideslip angle.

        Parameters
        ----------
        beta : float
            Current sideslip angle (rad).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed rudder deflection (rad).
        """
        delta_r = self.pid_sideslip_rudder.update(0.0, beta, dt)
        return delta_r

    def yaw_damper_with_rudder(self, r: float, dt: float) -> float:
        """Compute the rudder deflection needed for yaw damping.

        Parameters
        ----------
        r : float
            Current z-axis yaw rate (rad/s).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed rudder deflection (rad).
        """
        Ts = self.config.Ts_damper or 5.0 * dt
        self.xi_damper = self.xi_damper + Ts * (
            -self.config.pwo_damper * self.xi_damper + self.config.kr_damper * r
        )
        delta_r = -self.config.pwo_damper * self.xi_damper + self.config.kr_damper * r
        return delta_r

    def turn_coordination_with_rudder(
        self, yaw_rate: float, Vg: float, roll: float, dt: float
    ) -> float:
        """
        Compute rudder deflection for coordinated turns.

        Parameters
        ----------
        yaw_rate : float
            Current body yaw rate (rad/s).
        Vg : float
            Ground speed (m/s).
        roll : float
            Current roll angle (rad).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed rudder deflection (rad) for turn coordination.
        """
        yaw_rate_ref = g * roll / Vg
        delta_r = self.pid_turn_coord.update(yaw_rate_ref, yaw_rate, dt)
        return -delta_r  # rudder positive deflection for left turn

    def altitude_hold_with_pitch(
        self, altitude_ref: float, altitude: float, dt: float
    ) -> float:
        """Compute the pitch angle needed to maintain the desired altitude.

        Parameters
        ----------
        altitude_ref : float
            Desired altitude (m).
        altitude : float
            Current altitude (m).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed pitch angle (rad).
        """
        pitch_ref = self.pid_altitude_pitch.update(altitude_ref, altitude, dt)
        return pitch_ref

    def airspeed_hold_with_throttle(
        self, Va_ref: float, Va: float, dt: float, delta_t_trim: float = 0.0
    ) -> float:
        """Compute the throttle setting needed to maintain the desired airspeed.

        Parameters
        ----------
        Va_ref : float
            Desired airspeed (m/s).
        Va : float
            Current airspeed (m/s).
        dt : float
            Time step (s).
        delta_t_trim : float, optional
            Trimmed throttle setting to adjust the output (default is 0.0).

        Returns
        -------
        float
            The computed throttle setting (percentage), clipped to the allowed range.
        """
        delta_t = self.pid_airspeed_throttle.update(Va_ref, Va, dt)
        delta_t = np.clip(
            delta_t + delta_t_trim, self.config.min_throttle, self.config.max_throttle
        )
        return delta_t

    def airspeed_hold_with_pitch(self, Va_ref: float, Va: float, dt: float) -> float:
        """Compute the pitch angle needed to maintain the desired airspeed.

        Parameters
        ----------
        Va_ref : float
            Desired airspeed (m/s).
        Va : float
            Current airspeed (m/s).
        dt : float
            Time step (s).

        Returns
        -------
        float
            The computed pitch angle (rad).
        """
        pitch_ref = self.pid_airspeed_pitch.update(Va_ref, Va, dt)
        return pitch_ref
