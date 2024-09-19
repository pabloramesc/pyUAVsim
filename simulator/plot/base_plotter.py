"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from simulator.math.rotation import ned2xyz


class BasePlotter(ABC):

    def __init__(self, ax: plt.Axes = None, is_3d: bool = True) -> None:
        if ax is not None:
            self.ax = ax
        else:
            fig = plt.figure(figsize=(6, 6))
            self.ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        self.is_3d = is_3d

    def plot_horizontal_circle(
        self, center: np.ndarray, radius: float, style: str = "r--"
    ) -> None:
        N = 100
        ang = np.linspace(-np.pi, +np.pi, N)
        circle = np.zeros((N, 3)) if self.is_3d else np.zeros((N, 2))
        circle[:, 0] = radius * np.cos(ang) + center[0]
        circle[:, 1] = radius * np.sin(ang) + center[1]
        if self.is_3d:
            circle[:, 2] = center[2] * np.ones_like(ang)
        xyz = ned2xyz(circle)
        self.ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2] if self.is_3d else None, style)

    def show(self) -> None:
        plt.show()