import numpy as np

from ..sensors.sensor_system import SensorReadings
from .alpha_filter import AlphaFilter


class ModelInversionFilter:
    def __init__(self) -> None:
        self.accel_lpf = AlphaFilter(alpha=0.7)
        self.gyro_lpf = AlphaFilter(alpha=0.7)
        self.baro_lpf = AlphaFilter(alpha=0.9)
        self.airspeed_lpf = AlphaFilter(alpha=0.7)
        self.gps_lpf = AlphaFilter(alpha=0.1)

    def update(self, readings: SensorReadings):
        # Estimate angular rates from gyroscope
        p, q, r = self.gyro_lpf.update(readings.gyro)
        
        # Estimate altitude and airspeed from barometer and airspeed sensor
        g = 9.81  # m/s^2
        rho = 1.225  # kg/m^3
        h = self.baro_lpf.update(readings.baro) / (
            rho * g
        )  # Convert pressure to altitude
        Va = np.sqrt(
            2 / rho * self.airspeed_lpf.update(readings.airspeed)
        )  # Convert dynamic pressure to airspeed
        
        # Estimate roll and pitch from accelerometer
        ax, ay, az = self.accel_lpf.update(readings.accel)
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        # Estimate position and ground speed from GPS
        pn, pe, pd, Vg, heading = self.gps_lpf.update(readings.gps)