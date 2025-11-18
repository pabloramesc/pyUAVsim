import numpy as np

from .estimation_filter import EstimationFilter, EstimatedState

from ..sensors.sensor_system import SensorReadings
from .alpha_filter import AlphaFilter


class ModelInversionFilter(EstimationFilter):
    def __init__(self) -> None:
        self.accel_lpf = AlphaFilter(alpha=0.95)
        self.gyro_lpf = AlphaFilter(alpha=0.99)
        self.baro_lpf = AlphaFilter(alpha=0.90)
        self.airspeed_lpf = AlphaFilter(alpha=0.90)
        self.compass_lpf = AlphaFilter(alpha=0.0)
        self.gps_lpf = AlphaFilter(alpha=0.1)

    def update(self, readings: SensorReadings) -> EstimatedState:
        # Estimate angular rates from gyroscope
        p, q, r = np.deg2rad(self.gyro_lpf.update(readings.gyro))

        # Estimate altitude and airspeed from barometer and airspeed sensor
        g = 9.81  # m/s^2
        rho = 1.225  # kg/m^3
        h_baro = (101325 - self.baro_lpf.update(readings.baro) * 1e3) / (
            rho * g
        )  # Convert pressure (in kPa) to altitude
        Va = np.sqrt(
            2 / rho * self.airspeed_lpf.update(readings.airspeed) * 1e3
        )  # Convert dynamic pressure (in kPa) to airspeed

        # Estimate roll and pitch from accelerometer
        acc = self.accel_lpf.update(readings.accel)
        acc_norm = np.linalg.norm(acc) + 1e-6  # Prevent division by zero
        ax, ay, az = acc / acc_norm # Normalize acceleration
        roll = np.arctan2(-ay, -az)
        pitch = np.asin(ax)
        
        # Estimate yaw from compass
        yaw = np.deg2rad(self.compass_lpf.update(readings.compass))

        # Estimate position and ground speed from GPS
        pn, pe, h_gps, Vg, heading = self.gps_lpf.update(readings.gps)

        return EstimatedState(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            p=p,
            q=q,
            r=r,
            Va=Va,
            h=h_baro,
            pn=pn,
            pe=pe,
            pd=-h_gps,
            Vg=Vg,
            heading=heading,
        )
