import numpy as np
from .estimation_filter import EstimationFilter, EstimatedState
from ..sensors.sensor_system import SensorReadings
from .alpha_filter import AlphaFilter
from ..math.angles import wrap_angle_2pi

class ModelInversionFilter(EstimationFilter):
    def __init__(self, dt: float) -> None:
        """
        Model inversion filter for estimating aircraft state from sensor readings.

        Args:
            dt (float): Filter sampling period in seconds.
        """
        super().__init__(dt)

        # Low-pass filters for sensors
        self.accel_lpf = AlphaFilter(tau=0.05, dt=dt)
        self.gyro_lpf = AlphaFilter(tau=0.05, dt=dt)
        self.baro_lpf = AlphaFilter(tau=0.1, dt=dt)
        self.airspeed_lpf = AlphaFilter(tau=0.1, dt=dt)
        self.compass_lpf = AlphaFilter(tau=0.2, dt=dt)
        self.gps_lpf = AlphaFilter(tau=0.0, dt=dt)

    def update(self, readings: SensorReadings) -> EstimatedState:
        """Update the estimated state based on new sensor readings."""

        # --- Gyroscope: estimate body angular rates ---
        p, q, r = self._estimate_angular_rates(readings.gyro)

        # --- Barometer & Airspeed: estimate altitude and airspeed ---
        h_baro, Va = self._estimate_altitude_airspeed(readings.baro, readings.airspeed)

        # --- Accelerometer: estimate roll and pitch ---
        roll, pitch = self._estimate_attitude_from_accel(readings.accel)

        # --- Compass: estimate yaw/heading ---
        heading = self._estimate_heading_from_compass(readings.compass)

        # --- GPS: estimate position and ground velocity/course ---
        pn, pe, h_gps, Vg, course = self._estimate_gps(readings.gps)

        return EstimatedState(
            roll=roll,
            pitch=pitch,
            yaw=wrap_angle_2pi(heading),
            p=p,
            q=q,
            r=r,
            Va=Va,
            h=h_baro,
            pn=pn,
            pe=pe,
            pd=-h_gps,
            Vg=Vg,
            course=wrap_angle_2pi(course),
        )

    # ---------------- Helper methods ----------------

    def _estimate_angular_rates(
        self, gyro_readings: np.ndarray
    ) -> tuple[float, float, float]:
        """Convert gyroscope readings to rad/s and apply LPF."""
        p, q, r = np.deg2rad(self.gyro_lpf.update(gyro_readings))
        return p, q, r

    def _estimate_altitude_airspeed(
        self, baro: float, airspeed: float
    ) -> tuple[float, float]:
        """Estimate altitude from barometer and airspeed from dynamic pressure."""
        g = 9.81  # m/s^2
        rho = 1.225  # kg/m^3

        # Barometer: pressure to altitude
        h_baro = (101325 - self.baro_lpf.update(baro) * 1e3) / (rho * g)

        # Airspeed: dynamic pressure to velocity
        Va = np.sqrt(2 / rho * self.airspeed_lpf.update(airspeed) * 1e3)
        return h_baro.item(), Va.item()

    def _estimate_attitude_from_accel(self, accel: np.ndarray) -> tuple[float, float]:
        """Estimate roll and pitch from accelerometer with normalization."""
        acc = self.accel_lpf.update(accel)
        acc_norm = np.linalg.norm(acc) + 1e-6
        ax, ay, az = acc / acc_norm
        roll = np.arctan2(-ay, -az)
        pitch = np.arcsin(ax)
        return roll, pitch

    def _estimate_heading_from_compass(self, compass_deg: float) -> float:
        """Estimate yaw/heading from compass with LPF handling wrap-around."""
        psi = np.deg2rad(compass_deg)
        psi_sin, psi_cos = np.sin(psi), np.cos(psi)

        # Filter sin and cos components separately to handle wrap-around
        psi_sin_lpf, psi_cos_lpf = self.compass_lpf.update(np.array([psi_sin, psi_cos]))
        heading = np.arctan2(psi_sin_lpf, psi_cos_lpf)
        return heading

    def _estimate_gps(
        self, gps_data: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """
        Estimate filtered position, ground speed, and course from GPS.

        gps_data: (pn, pe, h, Vg, course_deg)
        """
        pn, pe, h_gps, Vg, course_deg = gps_data
        course_rad = np.deg2rad(course_deg)
        vn, ve = Vg * np.cos(course_rad), Vg * np.sin(course_rad)

        # Apply LPF to position and velocity
        pn, pe, h_gps, vn, ve = self.gps_lpf.update(np.array([pn, pe, h_gps, vn, ve]))

        Vg_lpf = np.sqrt(vn**2 + ve**2)
        course_lpf = np.arctan2(ve, vn)
        return pn, pe, h_gps, Vg_lpf, course_lpf
