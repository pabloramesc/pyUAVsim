import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray
from ..math.numeric_differentiation import jacobian

FloatArray = NDArray[np.floating]
Array2Array = Callable[[FloatArray, FloatArray], FloatArray]


class ExtendedKalmanFilter:
    """Extended Kalman Filter (EKF) for nonlinear systems.

    This class implements a standard Extended Kalman Filter using nonlinear
    state-transition and measurement functions:

        x_k = x_{k-1} + f(x_{k-1}, u_k) dt + w_k
        z_k = h(x_k, u_k) + v_k

    where w_k ~ N(0, Q) is process noise and v_k ~ N(0, R) is measurement noise.

    The EKF linearizes the nonlinear functions around the current state estimate
    using either user-provided analytic Jacobians or numerical Jacobians.

    Attributes:
        f (callable): Nonlinear state-transition function f(x, u) → (n, 1).
        h (callable): Nonlinear observation function h(x, u) → (p, 1).
        Q (np.ndarray): Process noise covariance matrix (n × n).
        R (np.ndarray): Measurement noise covariance matrix (p × p).
        P (np.ndarray): State estimate covariance matrix (n × n).
        x (np.ndarray): Current state estimate (n × 1).
        F_jac (callable | None): Optional analytic Jacobian of f.
        H_jac (callable | None): Optional analytic Jacobian of h.
    """

    def __init__(
        self,
        dt: float,
        f: Array2Array,
        h: Array2Array,
        Q: FloatArray,
        R: FloatArray,
        P0: Optional[FloatArray] = None,
        x0: Optional[FloatArray] = None,
        F_jac: Optional[Array2Array] = None,
        H_jac: Optional[Array2Array] = None,
    ):
        """Initialize the Extended Kalman Filter.

        Args:
            dt (float): Time step size.
            f (callable): Nonlinear state-transition function f(x, u) -> x_dot.
            h (callable): Nonlinear observation function h(x, u) -> z.
            Q (np.ndarray): Process noise covariance matrix (n × n).
            R (np.ndarray): Measurement noise covariance matrix (p × p).
            P (np.ndarray): Initial estimate error covariance matrix (n × n).
                If None, initializes to identity.
            x0 (np.ndarray | None): Initial state estimate (n × 1). If None,
                initializes to zero.
            F_jac (callable | None): Optional analytic Jacobian of f with
                signature F_jac(x, u) → (n × n). If None, numerical differentiation
                is used.
            H_jac (callable | None): Optional analytic Jacobian of h with
                signature H_jac(x, u) → (p × n). If None, numerical differentiation
                is used.
        """
        self.dt = dt
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

        self.P = P0 if P0 is not None else np.eye(Q.shape[0])
        self.x = x0 if x0 is not None else np.zeros((Q.shape[0], 1))

        self.F_jac = F_jac
        self.H_jac = H_jac

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Perform the EKF prediction (time update) step.

        The nonlinear prediction is:

            x_pred = x_prev + f(x_prev, u_k) Δt

        The covariance is updated using the Jacobian of f:

            F = ∂f/∂x evaluated at (x, u)
            F_d = I + F Δt + 0.5 F^2 Δt^2
            P_k = F_d P_{k-1} F_d^T + Q Δt^2

        Args:
            u (np.ndarray): Control input vector (m × 1).

        Returns:
            np.ndarray: Predicted state estimate x_k (n × 1).
        """
        # Nonlinear state prediction
        self.x = self.x + self.f(self.x, u) * self.dt

        # Jacobian of f
        if self.F_jac is not None:
            F = self.F_jac(self.x, u)
        else:
            F = jacobian(lambda x: self.f(x, u), self.x)

        # Covariance prediction
        Fd = np.eye(F.shape[0]) + F * self.dt + 0.5 * (F @ F) * self.dt**2
        self.P = Fd @ self.P @ Fd.T + self.Q * self.dt**2

        return self.x

    def update(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Perform the EKF measurement update (correction step).

        The nonlinear measurement update is:

            y = z - h(x_k, u_k)          (innovation)
            H = ∂h/∂x at x_k             (Jacobian)
            S = H P H^T + R              (innovation covariance)
            K = P H^T S^{-1}             (Kalman gain)
            x_k = x_k + K y              (state correction)

        Covariance is updated using the numerically stable Joseph form:

            P = (I - K H) P (I - K H)^T + K R K^T

        Args:
            z (np.ndarray): Measurement vector (p × 1).
            u (np.ndarray): Control input vector (m × 1).

        Returns:
            np.ndarray: Updated state estimate x_{k|k} (n × 1).
        """
        # Jacobian of h
        if self.H_jac is not None:
            H = self.H_jac(self.x, u)
        else:
            H = jacobian(lambda x: self.h(x, u), self.x)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation (measurement residual)
        y = z - self.h(self.x, u)

        # State correction
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        return self.x
