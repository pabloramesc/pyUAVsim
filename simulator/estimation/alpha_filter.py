import numpy as np
from simulator.utils.types import FloatLike, FloatArray
from typing import Optional


class AlphaFilter:
    def __init__(
        self,
        alpha: Optional[float] = None,
        dt: Optional[float] = None,
        tau: Optional[float] = None,
    ):
        """
        Simple first-order low-pass filter.
        Higher alpha means more smoothing (less responsive).

        Args:
            alpha (float, optional): Smoothing factor between 0 and 1. If provided, dt and tau are ignored.
            dt (float, optional): Sampling period in seconds (required if alpha is None).
            tau (float, optional): Time constant in seconds (required if alpha is None).
        """
        if alpha is not None:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("Alpha must be between 0 and 1.")
            self.alpha = float(alpha)
        else:
            if dt is None or tau is None:
                raise ValueError(
                    "If alpha is not provided, both dt and tau must be given to compute alpha."
                )
            if dt <= 0:
                raise ValueError("Sampling period dt must be positive.")
            if tau < 0:
                raise ValueError("Time constant tau must be positive.")
            self.alpha = float(np.exp(-dt / tau)) if tau > 0 else 0.0

        self.value: np.ndarray | None = None

    def update(self, x: FloatLike | FloatArray) -> np.ndarray:
        sample = np.array(x, dtype=float)
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * sample
        return self.value
