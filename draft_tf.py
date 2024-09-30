from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from simulator.math.numeric_integration import rk4


class TransferFunction:
    """
    A class to represent a transfer function.

    Attributes
    ----------
    num : np.ndarray
        N-size array representing the numerator coefficients of the transfer function.
    den : np.ndarray
        N-size array representing the denominator coefficients of the transfer function.
    order : int
        The order of the transfer function, which is the degree of the denominator.
    """

    def __init__(self, num: np.ndarray, den: np.ndarray):
        """
        Initialize the Transfer Function with numerator and denominator coefficients.

        Parameters
        ----------
        num : np.ndarray
            N-by-1 array representing the numerator coefficients of the transfer function.
        den : np.ndarray
            N-by-1 array representing the denominator coefficients of the transfer function.

        Raises
        ------
        ValueError
            If the numerator or denominator array is empty.
        """
        if len(num) == 0 or len(den) == 0:
            raise ValueError("Numerator and Denominator cannot be empty.")

        _num, _den = TransferFunction.normalize_tf(num, den)
        self.num = np.array(_num)
        self.den = np.array(_den)
        self.order = len(_den) - 1  # Order of the system

    @staticmethod
    def normalize_tf(num, den):
        if den[0] != 1:
            factor = den[0]
            den = [coeff / factor for coeff in den]
            num = [coeff / factor for coeff in num]
        return num, den

    def to_ccf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the transfer function to Controllable Canonical Form (CCF).

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

        Notes
        -----
        In Controllable Canonical Form (CCF), the system's states are controllable from the input.
        """
        n = self.order

        # num and den coefs
        a = np.array(self.den)
        b = np.zeros(len(self.den))
        b[len(self.den)-len(self.num):] = np.array(self.num)

        # A matrix (System dynamics matrix in CCF)
        A = np.zeros((n, n))
        A[:-1, 1:] = np.eye(n - 1)
        A[-1, :] = -a[1:][::-1]

        # B matrix (Input matrix)
        B = np.zeros((n, 1))
        B[-1, 0] = 1.0

        # C matrix (Output matrix)
        C = np.zeros((1, n))
        C[0, :] = b[1:][::-1] - a[1:][::-1] * b[0]

        # D matrix (Feedthrough matrix)
        D = np.zeros((1, 1))
        D[0, 0] = b[0]

        return A, B, C, D

    def to_ocf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the transfer function to Observable Canonical Form (OCF).

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

        Notes
        -----
        In Observable Canonical Form (OCF), the system's states are observable from the output.
        """
        n = self.order

        # num and den coefs
        a = np.array(self.den)
        b = np.zeros(len(self.den))
        b[len(self.den)-len(self.num):] = np.array(self.num)

        # A matrix (System dynamics matrix in OCF)
        A = np.zeros((n, n))
        A[1:, :-1] = np.eye(n - 1)
        A[:, -1] = -a[1:][::-1]

        # B matrix (Input matrix)
        B = np.zeros((n, 1))
        B[:, 0] = b[1:][::-1] - a[1:][::-1] * b[0]

        # C matrix (Output matrix)
        C = np.zeros((1, n))
        C[0, -1] = 1.0

        # D matrix (Feedthrough matrix)
        D = np.zeros((1, 1))
        D[0, 0] = b[0]

        return A, B, C, D

if __name__ == "__main__":

    dt = 0.01

    wn = 2*np.pi
    zeta = 0.7

    num = [wn**2]
    den = [1, 2 * zeta * wn, wn**2]

    tf = TransferFunction(num, den)
    A_ccf, B_ccf, C_ccf, D_ccf = tf.to_ccf()
    A_ocf, B_ocf, C_ocf, D_ocf = tf.to_ocf()

    dx_ccf = lambda x, u: A_ccf @ x + B_ccf @ u
    dx_ocf = lambda x, u: A_ocf @ x + B_ocf @ u

    N = 100
    t = np.linspace(0.0, 10.0, N)
    u = np.ones((1, N))
    x_ccf = np.zeros((2, N))  # State for CCF
    x_ocf = np.zeros((2, N))  # State for OCF
    y_ccf = np.zeros((1, N))  # Output for CCF
    y_ocf = np.zeros((1, N))  # Output for OCF

    for k in range(N)[1:]:
        tk = t[k]
        uk = u[:, k]
        xk_ccf = x_ccf[:, k - 1]
        xk_ocf = x_ocf[:, k - 1]

        # Calculate states
        dy_ccf = lambda t, y: dx_ccf(y, uk)
        dy_ocf = lambda t, y: dx_ocf(y, uk)
        x_ccf[:, k] = xk_ccf + rk4(dy_ccf, tk, xk_ccf, dt)
        x_ocf[:, k] = xk_ocf + rk4(dy_ocf, tk, xk_ocf, dt)

        # Calculate outputs
        y_ccf[:, k] = C_ccf @ x_ccf[:, k] + D_ccf @ uk
        y_ocf[:, k] = C_ocf @ x_ocf[:, k] + D_ocf @ uk

        print(f"k: {k}, t: {tk:11.4e} s, u: {uk[0]:11.4e}")
        print(f"CCF: x1: {x_ccf[0,k]:11.4e}, x2: {x_ccf[1,k]:11.4e}, y: {y_ccf[0,k]:11.4e}")
        print(f"OCF: x1: {x_ocf[0,k]:11.4e}, x2: {x_ocf[1,k]:11.4e}, y: {y_ocf[0,k]:11.4e}")
        print()

    # Plotting the comparison of outputs
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_ccf.flatten(), label="Output CCF")
    plt.plot(t, y_ocf.flatten(), label="Output OCF")
    plt.title("Output Comparison: Controllable vs Observable Canonical Form")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.grid()
    plt.legend()
    plt.show()
