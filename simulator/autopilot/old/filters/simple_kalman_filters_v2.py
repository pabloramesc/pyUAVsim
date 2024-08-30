"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

########################################################################################################################
##### KALMAN FILTER PROTOTIPE ##########################################################################################
########################################################################################################################
class KalmanFilter:
    def __init__(self) -> None:
        self.x = None  # state array
        self.u = None  # control input array
        self.z = None  # measurements array

        self.F = None  # state transition matrix
        self.G = None  # control matrix
        self.H = None  # measurements matrix

        self.Q = None  # process noise covariance matrix
        self.R = None  # measurements noise covariance matrix

        self.P = None  # error matrix
        self.K = None  # kalman gain
        self.I = None  # P sized identity matrix
        
        self.Dt = 0.0        
        self.gate_thr = 1e9

    def initialize(self, x0: np.ndarray) -> None:
        self.x = x0

    def prediction(self, uk: np.ndarray = None) -> np.ndarray:
        # state propagation
        if self.G is None or uk is None:
            self.x = self.F.dot(self.x)
        else:
            self.x = self.F.dot(self.x) + self.G.dot(uk)

        # error propagation
        Ad = self.I + self.F * self.Dt + self.F @ self.F * self.Dt**2
        self.P = Ad @ self.P @ Ad.T + self.Q * self.Dt**2

        return self.x

    def correction(self, zk: np.ndarray) -> np.ndarray:
        # Kalman gain calculation
        S_inv = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        resid = zk - self.H @ self.x
        if resid.T @ S_inv @ resid < self.gate_thr:
            # kalman gain calculation
            self.K = self.P @ self.H.T @ S_inv
            # state correction
            self.x = self.x + self.K @ resid
            # error correction
            I_KH = (self.I - self.K @ self.H)
            self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

        return self.x
    
########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
##### 1D KALMAN FILTER :: STATE-P MEASURE-P INPUT-V #############################################################
########################################################################################################################
class Kalman_1D_P_P_V(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray, gate_th: float = 0.2) -> None:
        """
        Kalman 1D P-P-V Filter
        P-P-V model stands for state: x=[p], measure: z=[p], input: u=[v]
        (propagate 1D state, measures the 1D state, and takes as input for propagation the differential of the state)

        Parameters
        ----------
        Dt : float
            update period in seconds
            
        z_noise : np.ndarray
            1-size array with standard deviation for measurement vector
            
        x_err : np.ndarray
            1-size array estimated error for state propagation
            
        gate_th : float, optional
            gate threshold to discard noisy measures, by default 0.2
        """
        super().__init__()

        self.Dt = Dt
        self.sigma_baro = z_noise  # measurement error
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(1)  # state array
        self.u = np.zeros(1)  # control inpu
        self.z = np.zeros(1)  # measurements array

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.G = self._get_control_matrix(Dt)  # control matrix
        self.H = np.array([[1.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2) # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(1)  # P sized identity matrix
        
        self.gate_thr = gate_th

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1
        Fk = np.eye(1)
        return Fk

    def _get_control_matrix(self, dt: float) -> np.ndarray:
        # dt
        Gk = np.diag([dt])
        return Gk

    def prediction(self, uk: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Propagate state

        Parameters
        ----------
        uk : np.ndarray
            1-size array with input as time differential of the state
             
        dt : float, optional
            time period, by default None

        Returns
        -------
        np.ndarray
            1-size array with state prediction
        """
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)
            self.G = self._get_control_matrix(dt)

        return super().prediction(uk)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        """
        Correct the propagated state

        Parameters
        ----------
        zk : np.ndarray
            1-size array with measure of the state

        Returns
        -------
        np.ndarray
            1-size array with state corrected
        """
        return super().correction(zk)


########################################################################################################################
##### 1D KALMAN FILTER :: STATE-PV MEASURE-P INPUT- ##############################################################
########################################################################################################################
class Kalman_1D_PV_P(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray, gate_th: float = 2.0) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_baro = z_noise  # measurement error
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(2)  # state array (alt vspeed)
        self.z = np.zeros(1)  # measurements array (alt_baro)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.array([[1.0, 0.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(2)  # P sized identity matrix
        
        self.gate_thr = gate_th

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1  dt
        #  0  1
        Fk = np.eye(2)
        Fk[0, 1] = dt
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### 1D KALMAN FILTER :: STATE-PVA MEASURE-P INPUT- #############################################################
########################################################################################################################
class Kalman_1D_PVA_P(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_baro = z_noise  # measurement error
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(3)  # state array (alt vspeed vacc)
        self.z = np.zeros(1)  # measurements array (alt_baro)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.array([[1.0, 0.0, 0.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(3)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1   dt  1/2*dt^2
        #  0   1   dt
        #  0   0   1
        Fk = np.eye(3)
        Fk[0, 1] = dt
        Fk[1, 2] = dt
        Fk[0, 2] = 0.5 * dt**2
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### 1D KALMAN FILTER :: STATE-PVA MEASURE-PV INPUT- ############################################################
########################################################################################################################
class Kalman_1D_PVA_PV(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray, gate_thr: float = 20.0) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_baro = z_noise  # measurement error (alt, vspeed)
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(3)  # state array (alt vspeed vacc)
        self.z = np.zeros(2)  # measurements array (alt_baro, vspeed)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(3)  # P sized identity matrix
        
        self.gate_thr = gate_thr

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1   dt  1/2*dt^2
        #  0   1   dt
        #  0   0   1
        Fk = np.eye(3)
        Fk[0, 1] = dt
        Fk[1, 2] = dt
        Fk[0, 2] = 0.5 * dt**2
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### 1D KALMAN FILTER :: STATE-PV MEASURE-P INPUT-A #############################################################
########################################################################################################################
class Kalman_1D_PV_P_A(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = z_noise  # measurement error
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(2)  # state array (alt vspeed)
        self.u = np.zeros(1)  # control input (vacc_imu)
        self.z = np.zeros(1)  # measurements array (alt_baro)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.G = self._get_control_matrix(Dt)  # control matrix
        self.H = np.array([[1.0, 0.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2) # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(2)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1  dt
        #  0  1
        Fk = np.eye(2)
        Fk[0, 1] = dt
        return Fk

    def _get_control_matrix(self, dt: float) -> np.ndarray:
        Gk = np.array([[0.5 * dt**2, dt]]).T
        return Gk

    def prediction(self, uk: np.ndarray, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)
            self.G = self._get_control_matrix(dt)

        return super().prediction(uk)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)
    
    
########################################################################################################################
##### 1D KALMAN FILTER :: STATE-PV MEASURE-PV INPUT-A ############################################################
########################################################################################################################
class Kalman_1D_PV_PV_A(KalmanFilter):
    def __init__(self, Dt: float, z_noise: np.ndarray, x_err: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = z_noise  # measurement error
        self.sigma_prop = x_err  # state propagation error

        self.x = np.zeros(2)  # state array (alt vspeed)
        self.u = np.zeros(1)  # control input (vacc_imu)
        self.z = np.zeros(2)  # measurements array (alt_baro vspeed)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.G = self._get_control_matrix(Dt)  # control matrix
        self.H = np.array([[1.0, 0.0], [0.0, 1.0]])  # measurements matrix

        self.Q = np.diag(x_err**2)  # process noise covariance matrix
        self.R = np.diag(z_noise**2)  # measurements noise covariance matrix

        self.P = np.diag(x_err**2) # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(2)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        #  1  dt
        #  0  1
        Fk = np.eye(2)
        Fk[0, 1] = dt
        return Fk

    def _get_control_matrix(self, dt: float) -> np.ndarray:
        Gk = np.array([[0.5 * dt**2, dt]]).T
        return Gk

    def prediction(self, uk: np.ndarray, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)
            self.G = self._get_control_matrix(dt)

        return super().prediction(uk)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)
    
########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
##### HORIZONTAL POSITION KALMAN FILTER :: STATE-PV MEASURE-P INPUT- ###################################################
########################################################################################################################
class Kalman_2D_PV_P(KalmanFilter):
    def __init__(self, Dt: float, sigma_gps: np.ndarray, sigma_prop: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = sigma_gps  # gps horizontal position measurement error
        self.sigma_prop = sigma_prop  # state propagation error

        self.x = np.zeros(4)  # state array (PN PE VN VE)
        self.z = np.zeros(2)  # measurements array (PN_gps PE_gps)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.eye(2, 4)  # measurements matrix

        self.Q = np.diag(sigma_prop**2)  # process noise covariance matrix
        self.R = np.diag(sigma_gps**2)  # measurements noise covariance matrix

        self.P = np.diag(sigma_prop**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(4)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        # I2  I2*dt
        # 02  I2
        Fk = np.eye(4)
        Fk[0:2, 2:4] = np.eye(2) * dt
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### HORIZONTAL POSITION KALMAN FILTER :: STATE-PVA MEASURE-P INPUT- ##################################################
########################################################################################################################
class Kalman_2D_PVA_P(KalmanFilter):
    def __init__(self, Dt: float, sigma_gps: np.ndarray, sigma_prop: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = sigma_gps  # gps horizontal position measurement error
        self.sigma_prop = sigma_prop  # state propagation error

        self.x = np.zeros(6)  # state array (PN PE VN VE AN AE)
        self.z = np.zeros(2)  # measurements array (PN_gps PE_gps)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.eye(2, 6)  # measurements matrix

        self.Q = np.diag(sigma_prop**2)  # process noise covariance matrix
        self.R = np.diag(sigma_gps**2)  # measurements noise covariance matrix

        self.P = np.diag(sigma_prop**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(6)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        # I2     I2*dt  I2*(1/2)*dt^2
        # 02     I2     I2*dt
        # 02     02     I2
        Fk = np.eye(6)
        Fk[0:2, 2:4] = np.eye(2) * dt
        Fk[2:4, 4:6] = np.eye(2) * dt
        Fk[0:2, 4:6] = np.eye(2) * 0.5 * dt**2
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### HORIZONTAL POSITION KALMAN FILTER :: STATE-PVA MEASURE-PV INPUT- #################################################
########################################################################################################################
class Kalman_2D_PVA_PV(KalmanFilter):
    def __init__(self, Dt: float, sigma_gps: np.ndarray, sigma_prop: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = sigma_gps  # gps measurement error sigma-(pn pe vn ve) (4-size numpy.ndarray)
        self.sigma_prop = sigma_prop  # state propagation error

        self.x = np.zeros(6)  # state array (PN PE VN VE AN AE)
        self.z = np.zeros(4)  # measurements array (PN_gps PE_gps VN_gps VE_gps)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.H = np.eye(4, 6)  # measurements matrix

        self.Q = np.diag(sigma_prop**2)  # process noise covariance matrix
        self.R = np.diag(sigma_gps**2)  # measurements noise covariance matrix

        self.P = np.diag(sigma_prop**2)  # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(6)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        # I2     I2*dt  I2*(1/2)*dt^2
        # 02     I2     I2*dt
        # 02     02     I2
        Fk = np.eye(6)
        Fk[0:2, 2:4] = np.eye(2) * dt
        Fk[2:4, 4:6] = np.eye(2) * dt
        Fk[0:2, 4:6] = np.eye(2) * 0.5 * dt**2
        return Fk

    def prediction(self, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)

        return super().prediction(uk=None)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)


########################################################################################################################
##### HORIZONTAL POSITION KALMAN FILTER :: STATE-PV MEASURE-PV INPUT-A #################################################
########################################################################################################################
class Kalman_2D_PV_PV_A(KalmanFilter):
    def __init__(self, Dt: float, sigma_gps: np.ndarray, sigma_prop: np.ndarray) -> None:
        super().__init__()

        self.Dt = Dt
        self.sigma_gps = sigma_gps  # gps measurement error sigma-(pn pe vn ve) (4-size numpy.ndarray)
        self.sigma_prop = sigma_prop  # state propagation error

        self.x = np.zeros(4)  # state array (PN PE VN VE)
        self.u = np.zeros(2)  # control input array (AN_imu AE_imu)
        self.z = np.zeros(4)  # measurements array (PN_gps PE_gps VN_gps VE_gps)

        self.F = self._get_transition_matrix(Dt)  # state transition matrix
        self.G = self._get_control_matrix(Dt)  # control matrix
        self.H = np.eye(4)  # measurements matrix

        self.Q = np.diag(sigma_prop**2)  # process noise covariance matrix
        self.R = np.diag(sigma_gps**2)  # measurements noise covariance matrix

        self.P = np.diag(sigma_prop**2) # error matrix
        self.K = None  # kalman gain
        self.I = np.eye(4)  # P sized identity matrix

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        # I2  I2*dt
        # 02  I2
        Fk = np.eye(4)
        Fk[0:2, 2:4] = np.eye(2) * dt
        return Fk

    def _get_control_matrix(self, dt: float) -> np.ndarray:
        Gk = np.zeros((4, 2))
        Gk[0:2, 0:2] = np.eye(2) * 0.5 * dt**2
        Gk[2:4, 0:2] = np.eye(2) * dt
        return Gk

    def prediction(self, uk: np.ndarray, dt: float = None) -> np.ndarray:
        if dt != self.Dt and not dt is None:
            self.Dt = dt
            self.F = self._get_transition_matrix(dt)
            self.G = self._get_control_matrix(dt)

        return super().prediction(uk)

    def correction(self, zk: np.ndarray) -> np.ndarray:
        return super().correction(zk)

########################################################################################################################
########################################################################################################################
########################################################################################################################