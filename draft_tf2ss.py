import numpy as np
import matplotlib.pyplot as plt


def tf2ss(num: np.ndarray, den: np.ndarray):
    """
    Convert a transfer function to state-space representation.

    Parameters
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.

    Returns
    -------
    A : np.ndarray
        n-by-n state matrix (system dynamics matrix).
    B : np.ndarray
        n-by-1 input matrix.
    C : np.ndarray
        1-by-n output matrix.
    D : np.ndarray
        1-by-1 feedthrough matrix.
    """
    _den = np.array(den)
    n = len(den) - 1  # Order of the system
    _num = np.zeros(len(den))
    _num[len(den)-len(num):] = np.array(num)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    C = np.zeros((1, n))
    D = np.zeros((1, 1))

    A[:-1, 1:] = np.eye(n - 1)  # Shift identity matrix
    A[-1, :] = -_den[1:][::-1]  # Last row from den coefficients

    B[-1, 0] = 1.0 # Assuming single input

    C[0, :] = _num[1:][::-1] - _den[1:][::-1] * _num[0]

    D[0, 0] = _num[0]

    return A, B, C, D


def rk4(dy, t, y, dt):
    """
    Runge-Kutta 4th order integration.

    Parameters
    ----------
    dy : callable
        Function that returns the derivatives of y.
    t : float
        Current time.
    y : np.ndarray
        Current state vector.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        Updated state vector after one time step.
    """
    k1 = dt * dy(t, y)
    k2 = dt * dy(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * dy(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * dy(t + dt, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# # Example 1: 1st Order Transfer Function
# tau = 1.0  # Time constant
# num = [1.0]  # Numerator coefficients
# den = [tau, 1.0]  # Denominator coefficients

# Example 2: 2nd Order Transfer Function
wn = 10.0  # Natural frequency
zeta = 0.5  # Damping ratio
num = [wn**2]  # Numerator coefficients
den = [1.0, 2 * zeta * wn, wn**2]  # Denominator coefficients

# Convert to state-space representation
A, B, C, D = tf2ss(num, den)

# Time settings
N = 1000  # Number of time steps
t = np.linspace(0, 1.0, N)  # Time vector
dt = np.ptp(t)/N  # Time step
u = np.ones((1, N))  # Step input

# Initialize state variables and output
x = np.zeros((len(den)-1, N))  # State vector (2 states for a 2nd order system)
y = np.zeros((1, N))  # Output vector


# Integrate using RK4
for k in range(1, N):
    dx = lambda t, x: A @ x + B @ u[:, k]  # State dynamics function for RK4
    x[:, k] = rk4(dx, t[k - 1], x[:, k - 1], dt)  # Update state
    y[:, k] = C @ x[:, k] + D @ u[:, k]  # Calculate output

# Final output calculation
y[:, -1] = C @ x[:, -1] + D @ u[:, -1]

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(t, y[0], label="Output y(t)", color="blue")
plt.title("Response of 2nd Order System")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid()
plt.legend()
plt.show()
