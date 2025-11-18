import numpy as np
from simulator.utils.types import FloatLike, FloatArray


class AlphaFilter:
    def __init__(self, alpha: float):
        """
        Simple first-order low-pass filter.
        Higher alpha means more smoothing (less responsive).

        Args:
            alpha (float): Smoothing factor between 0 and 1.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = float(alpha)
        self.value: np.ndarray | None = None

    def update(self, x: FloatLike | FloatArray) -> np.ndarray:
        sample = np.array(x, dtype=float)
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * sample
        return self.value
