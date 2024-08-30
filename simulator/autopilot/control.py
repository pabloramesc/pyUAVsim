import numpy as np

from simulator.aircraft.aircraft_state import AircraftState
from simulator.aircraft.airframe_parameters import AirframeParameters
from simulator.autopilot.autopilot_config import AutopilotConfig
from simulator.autopilot.pid_controller import PIDController


class AutopilotControl:

    def __init__(self, config: AutopilotConfig) -> None:
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


    def roll_hold_with_aileron(
        self, roll_ref: float, roll: float, roll_rate: float
    ) -> float:
        delta_a = (
            self.config.kp_roll_aileron * (roll_ref - roll)
            - self.config.kd_roll_aileron * roll_rate
        )
        return np.clip(delta_a, self.config.min_aileron, self.config.max_aileron)

    @staticmethod
    def wrap_course(course_ref: float, course: float):
        while course_ref - course > +np.pi:
            course_ref -= 2 * np.pi
        while course_ref - course < -np.pi:
            course_ref += 2 * np.pi
        return course_ref

    def course_hold_with_roll(
        self, course_ref: float, course: float, dt: float
    ) -> float:
        course_ref = self.wrap_course(course_ref, course)
        roll_ref = self.pid_course_roll.update(course_ref, course, dt)
        return roll_ref

    def sideslip_hold_with_rudder(self, beta: float, dt: float) -> float:
        delta_r = self.pid_sideslip_rudder.update(0.0, beta, dt)
        return delta_r

    def yaw_damper_with_rudder(self, yaw_rate: float, dt: float) -> float:
        if self.config.Ts_damper is None:
            Ts = 5.0 * dt
        else:
            Ts = self.config.Ts_damper
        self.xi_damper = self.xi_damper + Ts * (
            -self.config.p_wo * self.xi_damper + self.config.kr * yaw_rate
        )
        delta_r = -self.config.p_wo * self.xi_damper + self.config.kr * yaw_rate
        return delta_r

    def pitch_hold_with_elevator(
        self, pitch_ref: float, pitch: float, pitch_rate: float
    ) -> float:
        delta_e = (
            self.config.kp_pitch_elevator * (pitch_ref - pitch)
            - self.config.kd_pitch_elevator * pitch_rate
        )
        return np.clip(delta_e, self.config.min_elevator, self.config.max_elevator)

    def altitude_hold_with_pitch(self, altitude_ref: float, altitude: float, dt: float) -> float:
            pitch_ref = self.pid_altitude_pitch.update(altitude_ref, altitude, dt)
            return pitch_ref
    
    def airspeed_hold_with_throttle(
        self, Va_ref: float, Va: float, dt: float, delta_t_trim: float = 0.0
    ) -> float:
        delta_t = self.pid_airspeed_throttle.update(Va_ref, Va, dt)
        delta_t = np.clip(delta_t + delta_t_trim, self.config.min_throttle, self.config.max_throttle)
        return delta_t

    def airspeed_hold_with_pitch(self, Va_ref: float, Va: float, dt: float) -> float:
        pitch_ref = self.pid_airspeed_pitch.update(Va_ref, Va, dt)
        return pitch_ref