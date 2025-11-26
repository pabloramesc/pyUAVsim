import numpy as np


class KalmanFilter:
    """Abstract base class for linear Kalman filters.

    This class implements the common structure and state for a discrete-time
    linear Kalman filter with additive Gaussian noise. It stores the system
    matrices and provides a standard state representation, while leaving
    the `predict` and `update` methods abstract so that subclasses can
    customize behavior (e.g., different noise models or constraints).

    The system model is assumed to be:

        x_k   = A x_{k-1} + B u_k + w_k
        z_k   = H x_k     + v_k

    where w_k ~ N(0, Q) is process noise and v_k ~ N(0, R) is measurement noise.

    Attributes:
        A (np.ndarray): State transition matrix of shape (n, n).
        B (np.ndarray): Control input matrix of shape (n, m).
        H (np.ndarray): Observation (measurement) matrix of shape (p, n).
        Q (np.ndarray): Process noise covariance matrix of shape (n, n).
        R (np.ndarray): Measurement noise covariance matrix of shape (p, p).
        P (np.ndarray): Estimate error covariance matrix of shape (n, n).
        x (np.ndarray): Current state estimate vector of shape (n, 1).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P0: np.ndarray | None = None,
        x0: np.ndarray | None = None,
    ):
        """Initialize the Kalman filter.

        Args:
            A (np.ndarray): State transition matrix of shape (n, n).
            B (np.ndarray): Control input matrix of shape (n, m).
            H (np.ndarray): Observation (measurement) matrix of shape (p, n).
            Q (np.ndarray): Process noise covariance matrix of shape (n, n).
            R (np.ndarray): Measurement noise covariance matrix of shape (p, p).
            P (np.ndarray): Initial estimate error covariance matrix of shape (n, n).
                If None, it is initialized to the identity matrix.
            x0 (np.ndarray | None, optional): Initial state estimate vector of
                shape (n, 1). If None, it is initialized to a zero vector.
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        self.P = P0 if P0 is not None else np.eye(Q.shape[0])
        self.x = x0 if x0 is not None else np.zeros((A.shape[0], 1))

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Perform the prediction (time update) step.

        This method should implement the state and covariance prediction:

            x_{k|k-1} = A x_{k-1|k-1} + B u_k
            P_{k|k-1} = A P_{k-1|k-1} A^T + Q

        Implementations in subclasses may modify or extend this behavior
        (e.g., time-varying matrices, additional process models).

        Args:
            u (np.ndarray): Control input vector of shape (m, 1).

        Returns:
            np.ndarray: Predicted state estimate vector x_{k|k-1} of shape (n, 1).
        """
        # State prediction
        self.x = self.A @ self.x + self.B @ u
        
        # Covariance prediction
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """Perform the correction (measurement update) step.

        This method should implement the measurement update:

            S  = H P_{k|k-1} H^T + R
            K  = P_{k|k-1} H^T S^{-1}
            y  = z_k - H x_{k|k-1}
            x_{k|k} = x_{k|k-1} + K y

        And then update the covariance, typically using either:

            P_{k|k} = (I - K H) P_{k|k-1}          (standard form)

        or the more numerically stable Joseph form:

            P_{k|k} = (I - K H) P_{k|k-1} (I - K H)^T + K R K^T

        This base implementation uses the Joseph form.

        Args:
            z (np.ndarray): Measurement vector of shape (p, 1).

        Returns:
            np.ndarray: Updated state estimate vector x_{k|k} of shape (n, 1).
        """
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # State update
        self.x = self.x + K @ y

        # Covariance update using Joseph form for numerical stability
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        return self.x
